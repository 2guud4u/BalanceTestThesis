#!/usr/bin/env python3
"""
Hyperparameter search for MS-TCN2, BiLSTM, and ASFormer segmentors.

Uses Optuna (Bayesian TPE + MedianPruner) over:
  - segmentor architecture params  (num_layers, hidden_size, num_f_maps, …)
  - shared trainer params          (lr, weight_decay, lambda_smooth, batch_size)

Each trial trains fold_0 for N epochs (default 150, patience 10) and reports
the best val_F1 achieved.  The study is persisted to SQLite so crashes don't
lose progress.

Usage (run from project root or src/):
    cd /code/jjiang23/BalanceTestThesis
    python scripts/tune.py \
        --segmentor mstcn \
        --data      configs/data/MPW.yml \
        --splits    /code/jjiang23/pathml/aim2_balanceV2/data/splits.json \
        [--encoder  configs/encoder/MAMP.yml] \
        [--fold     fold_0] \
        [--n_trials 60] \
        [--n_epochs 150] \
        [--device   cuda] \
        [--study_db tune_results/mstcn_mpw.db]
"""

import os
import sys
import json
import yaml
import shutil
import argparse
import importlib.util
import tempfile
import copy

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# train.py lives in src/, some imports expect src/ on the path
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.PoseDataset import PoseDataset
from Trainer import Trainer  # src/Trainer.py


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_module_from_path(file_path):
    spec = importlib.util.spec_from_file_location("_dyn_mod", file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Per-segmentor search spaces ───────────────────────────────────────────────

def suggest_mstcn(trial):
    """Sample MS-TCN2 architecture.

    num_layers_PG is capped at 7 so max dilation (2^6=64) fits inside
    window_size=120.  num_f_maps must divide evenly by r1/r2 in the attention
    path, but MSTCN has no attention so any value is fine.
    """
    num_layers_PG = trial.suggest_int("num_layers_PG", 4, 7)
    num_layers_R  = trial.suggest_int("num_layers_R",  4, 8)
    num_R         = trial.suggest_int("num_R",         1, 4)
    num_f_maps    = trial.suggest_categorical("num_f_maps", [32, 64, 128])
    return {
        "num_layers_PG": num_layers_PG,
        "num_layers_R":  num_layers_R,
        "num_R":         num_R,
        "num_f_maps":    num_f_maps,
        "num_classes":   2,
        "init_segmentor_path": os.path.join(
            PROJECT_ROOT, "initializers", "segementor", "MSTCN.py"
        ),
    }


def suggest_bilstm(trial):
    num_layers  = trial.suggest_int("num_layers",  1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
    # PyTorch LSTM ignores dropout when num_layers==1 but warns; force 0.
    dropout     = trial.suggest_float("dropout", 0.0, 0.5) if num_layers > 1 else 0.0
    trial.set_user_attr("dropout_effective", dropout)
    return {
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
        "dropout":     dropout,
        "num_classes": 2,
        "init_segmentor_path": os.path.join(
            PROJECT_ROOT, "initializers", "segementor", "biLSTM.py"
        ),
    }


def suggest_asformer(trial):
    """Sample ASFormer architecture.

    r1 and r2 must divide num_f_maps (they shrink Q/K dims via integer division),
    so we pick num_f_maps first and then constrain the reduction factors.
    """
    num_f_maps = trial.suggest_categorical("num_f_maps", [32, 64, 128])
    num_layers  = trial.suggest_int("num_layers",  6, 12)
    num_decoders = trial.suggest_int("num_decoders", 1, 4)

    # r1,r2 ∈ {2,4} both must divide num_f_maps
    valid_r = [r for r in [2, 4] if num_f_maps % r == 0]
    r1 = trial.suggest_categorical("r1", valid_r)
    r2 = trial.suggest_categorical("r2", valid_r)

    channel_masking_rate = trial.suggest_float("channel_masking_rate", 0.1, 0.5)
    return {
        "num_classes":          2,
        "num_f_maps":           num_f_maps,
        "num_layers":           num_layers,
        "num_decoders":         num_decoders,
        "r1":                   r1,
        "r2":                   r2,
        "channel_masking_rate": channel_masking_rate,
        "att_type":             "block_att",
        "init_segmentor_path":  os.path.join(
            PROJECT_ROOT, "initializers", "segementor", "ASFormer.py"
        ),
    }


SEGMENTOR_SUGGESTERS = {
    "mstcn":    suggest_mstcn,
    "bilstm":   suggest_bilstm,
    "asformer": suggest_asformer,
}


# ── Trainer search space (shared across all segmentors) ───────────────────────

def suggest_trainer(trial, base_t_cfg):
    t_cfg = copy.deepcopy(base_t_cfg)
    t_cfg["lr"]             = trial.suggest_float("lr",            1e-4, 5e-3, log=True)
    t_cfg["weight_decay"]   = trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True)
    t_cfg["lambda_smooth"]  = trial.suggest_float("lambda_smooth", 0.0,  0.3)
    t_cfg["batch_size"]     = trial.suggest_categorical("batch_size", [8, 16, 32])
    t_cfg["time_alignment"] = trial.suggest_categorical(
        "time_alignment", ["downsample_labels", "upsample_preds"]
    )
    return t_cfg


# ── Core: train one fold, return best val_f1 ─────────────────────────────────

def run_fold(
    d_cfg, e_cfg, s_cfg, t_cfg,
    train_files, val_files,
    encoder,                # pre-built, shared across trials
    n_epochs, device, trial=None,
):
    """
    Train for one fold.  Returns best val_f1 (float) or None on failure.

    `trial` is an Optuna trial object used for pruning; pass None to disable.
    """
    tmpdir = tempfile.mkdtemp(prefix="tune_trial_")
    try:
        joint_indices = d_cfg.get("joint_indices")

        train_ds = PoseDataset(
            train_files,
            window_size=d_cfg["window_size"],
            featureH5Key=d_cfg["h5_key"],
            stride=d_cfg["stride"],
            augment=True,
            joint_indices=joint_indices,
        )
        val_ds = PoseDataset(
            val_files,
            window_size=d_cfg["window_size"],
            featureH5Key=d_cfg["h5_key"],
            stride=d_cfg["stride"],
            augment=False,
            joint_indices=joint_indices,
        )

        class_weights = train_ds.compute_class_weights(device=device)

        # Load segmentor initializer fresh each trial
        seg_init = getattr(
            load_module_from_path(s_cfg["init_segmentor_path"]),
            "initialize_segmentor",
        )
        segmentor = seg_init(
            s_cfg,
            encoder,
            class_weights=class_weights,
            lambda_smooth=t_cfg.get("lambda_smooth", 0.01),
            time_alignment=t_cfg.get("time_alignment", "downsample_labels"),
        )

        trainer = Trainer(
            encoder=encoder,
            segmentor=segmentor,
            early_stop_patience=t_cfg.get("early_stop_patience", 10),
            early_stop_min_delta=t_cfg.get("early_stop_min_delta", 0.001),
            early_stop_monitor=t_cfg.get("early_stop_monitor", "val_f1"),
        )

        # Monkey-patch Trainer.train to report intermediate values and prune.
        # We wrap the inner epoch loop via a callback instead of patching.
        best_val_f1 = [None]

        # Run training — we override n_epochs to the tuning budget
        t_cfg_run = copy.deepcopy(t_cfg)
        t_cfg_run["epochs"] = n_epochs

        # We need pruning hooks inside the epoch loop.
        # Strategy: subclass / wrap Trainer._early_stop_step to also call
        # trial.report / trial.should_prune.
        original_early_stop = trainer._early_stop_step

        epoch_counter = [0]

        def patched_early_stop(val_metric):
            epoch_counter[0] += 1
            if val_metric is not None:
                best_val_f1[0] = val_metric
                if trial is not None:
                    trial.report(val_metric, epoch_counter[0])
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            return original_early_stop(val_metric)

        trainer._early_stop_step = patched_early_stop

        trainer.train(
            save_dir=tmpdir,
            batch_gen=train_ds,
            val_batch_gen=val_ds,
            num_epochs=n_epochs,
            batch_size=t_cfg_run["batch_size"],
            learning_rate=float(t_cfg_run["lr"]),
            weight_decay=float(t_cfg_run.get("weight_decay", 0.0)),
            device=device,
        )

        result = trainer._best_val if trainer._best_val is not None else best_val_f1[0]

        del segmentor, trainer, train_ds, val_ds
        torch.cuda.empty_cache()
        return result

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial failed: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(
    segmentor_type, d_cfg, e_cfg, base_t_cfg, splits, fold_name,
    encoder, n_epochs, device,
):
    """
    If fold_name is None, the objective averages val_f1 across ALL folds
    (statistically clean: no fold is used for both HP selection and final eval).
    If fold_name is set, only that single fold is used (faster but fold_name's
    val set is then contaminated for final reporting).
    """
    suggest_segmentor = SEGMENTOR_SUGGESTERS[segmentor_type]

    def objective(trial):
        s_cfg = suggest_segmentor(trial)
        t_cfg = suggest_trainer(trial, base_t_cfg)

        fold_names = list(splits.keys()) if fold_name is None else [fold_name]
        scores = []
        for i, fn in enumerate(fold_names):
            fold = splits[fn]
            # Only pass trial (for pruning) on the first fold to avoid
            # conflicting intermediate reports across folds.
            val_f1 = run_fold(
                d_cfg, e_cfg, s_cfg, t_cfg,
                fold["train"], fold["val"],
                encoder, n_epochs, device,
                trial=trial if i == 0 else None,
            )
            if val_f1 is None:
                raise optuna.TrialPruned()
            scores.append(val_f1)

        return float(np.mean(scores))

    return objective


# ── Save best config ──────────────────────────────────────────────────────────

def save_best_configs(study, segmentor_type, out_dir, base_t_cfg):
    os.makedirs(out_dir, exist_ok=True)
    best = study.best_trial

    # Rebuild s_cfg and t_cfg from best params
    seg_keys = {
        "mstcn":    ["num_layers_PG", "num_layers_R", "num_R", "num_f_maps"],
        "bilstm":   ["hidden_size", "num_layers"],
        "asformer": ["num_f_maps", "num_layers", "num_decoders", "r1", "r2",
                     "channel_masking_rate"],
    }[segmentor_type]
    trainer_keys = ["lr", "weight_decay", "lambda_smooth", "batch_size", "time_alignment"]

    s_cfg_out = {k: best.params[k] for k in seg_keys if k in best.params}
    s_cfg_out["num_classes"] = 2
    s_cfg_out["init_segmentor_path"] = os.path.join(
        PROJECT_ROOT, "initializers", "segementor",
        {"mstcn": "MSTCN.py", "bilstm": "biLSTM.py",
         "asformer": "ASFormer.py"}[segmentor_type],
    )
    if segmentor_type == "bilstm":
        s_cfg_out["dropout"] = best.user_attrs.get(
            "dropout_effective", best.params.get("dropout", 0.3)
        )
    if segmentor_type == "asformer":
        s_cfg_out["att_type"] = "block_att"

    t_cfg_out = copy.deepcopy(base_t_cfg)
    for k in trainer_keys:
        if k in best.params:
            t_cfg_out[k] = best.params[k]

    seg_path = os.path.join(out_dir, f"{segmentor_type}_best.yml")
    t_path   = os.path.join(out_dir, "trainer_best.yml")

    with open(seg_path, "w") as f:
        yaml.dump(s_cfg_out, f, sort_keys=False)
    with open(t_path, "w") as f:
        yaml.dump(t_cfg_out, f, sort_keys=False)

    print(f"\nBest trial #{best.number}  val_f1={best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"  Saved segmentor config → {seg_path}")
    print(f"  Saved trainer config   → {t_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter tuning via Optuna")
    p.add_argument("--segmentor",  required=True, choices=list(SEGMENTOR_SUGGESTERS),
                   help="Which segmentor to tune: mstcn | bilstm | asformer")
    p.add_argument("--data",       required=True, help="Path to data config YAML")
    p.add_argument("--splits",     required=True, help="Path to splits JSON")
    p.add_argument("--encoder",    default=None,  help="Path to encoder config YAML (omit for Identity)")
    p.add_argument("--trainer",    default=os.path.join(PROJECT_ROOT, "configs", "trainer", "downsamp.yml"),
                   help="Base trainer config YAML (lr/wd/lambda_smooth will be overridden)")
    p.add_argument("--fold",       default="fold_0",
                   help="Which fold to use for HP search (default: fold_0). "
                        "IMPORTANT: exclude this fold from final evaluation — its val set "
                        "was used to select hyperparameters and is therefore contaminated. "
                        "Pass 'all' to average across all folds (contaminates every fold; "
                        "only valid when you have a separate held-out test set).")
    p.add_argument("--n_trials",   type=int, default=60,
                   help="Number of Optuna trials (default: 60)")
    p.add_argument("--n_epochs",   type=int, default=150,
                   help="Max epochs per trial (default: 150)")
    p.add_argument("--device",     default="cuda",
                   help="PyTorch device (default: cuda)")
    p.add_argument("--study_db",   default=None,
                   help="SQLite path for persistent study, e.g. tune_results/mstcn.db")
    p.add_argument("--out_dir",    default=None,
                   help="Output dir for best configs (default: tune_results/<segmentor>_<data>)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load configs ─────────────────────────────────────────────────────────
    d_cfg      = load_yaml(os.path.abspath(args.data))
    base_t_cfg = load_yaml(os.path.abspath(args.trainer))
    e_cfg      = load_yaml(os.path.abspath(args.encoder)) if args.encoder else None

    # Override patience for faster tuning
    base_t_cfg["early_stop_patience"]  = 10
    base_t_cfg["early_stop_min_delta"] = 0.001
    base_t_cfg["early_stop_monitor"]   = "val_f1"
    base_t_cfg["device"]               = args.device

    with open(os.path.abspath(args.splits)) as f:
        splits = json.load(f)

    tune_fold = None if args.fold == "all" else args.fold
    if tune_fold is not None and tune_fold not in splits:
        raise ValueError(f"Fold '{tune_fold}' not found in splits. "
                         f"Available: {list(splits.keys())}")
    if tune_fold is None:
        print(
            "\nWARNING: --fold all contaminates every fold's val set. "
            "Only use this mode if you have a separate held-out test set. "
            "Without one, use --fold fold_0 and exclude fold_0 from final evaluation.\n"
        )

    # ── Build encoder once — shared across all trials ────────────────────────
    if e_cfg is not None:
        enc_init = getattr(
            load_module_from_path(os.path.abspath(e_cfg["init_encoder_path"])),
            "initialize_encoder",
        )
    else:
        _id_path = os.path.join(PROJECT_ROOT, "initializers", "encoder", "Identity.py")
        enc_init = getattr(load_module_from_path(_id_path), "initialize_encoder")

    encoder = enc_init(d_cfg, e_cfg)
    encoder.to(args.device).eval()
    print(f"Encoder built: out_dim={encoder.out_dim}")

    # ── Set up output dir ────────────────────────────────────────────────────
    data_tag = os.path.splitext(os.path.basename(args.data))[0]
    out_dir  = args.out_dir or os.path.join(
        PROJECT_ROOT, "tune_results", f"{args.segmentor}_{data_tag}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── Create / load Optuna study ────────────────────────────────────────────
    storage = None
    if args.study_db:
        db_path = os.path.abspath(args.study_db)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        storage = f"sqlite:///{db_path}"

    study_name = f"{args.segmentor}_{data_tag}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5),
        storage=storage,
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}")
    print(f"  Segmentor : {args.segmentor}")
    print(f"  Data      : {args.data}")
    print(f"  Fold      : {'all (averaged)' if tune_fold is None else tune_fold}")
    print(f"  Trials    : {args.n_trials}  (epochs/trial: {args.n_epochs})")
    print(f"  Output    : {out_dir}\n")

    # ── Run optimization ──────────────────────────────────────────────────────
    objective = make_objective(
        segmentor_type=args.segmentor,
        d_cfg=d_cfg,
        e_cfg=e_cfg,
        base_t_cfg=base_t_cfg,
        splits=splits,
        fold_name=tune_fold,
        encoder=encoder,
        n_epochs=args.n_epochs,
        device=args.device,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Completed {len(study.trials)} trials")
    print(f"Best val_f1: {study.best_value:.4f}")
    save_best_configs(study, args.segmentor, out_dir, base_t_cfg)

    # Summary CSV
    import pandas as pd
    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, "all_trials.csv")
    df.to_csv(csv_path, index=False)
    print(f"All trials saved → {csv_path}")


if __name__ == "__main__":
    main()
