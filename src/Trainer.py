import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
import numpy as np
def compute_f1(predictions, targets, mask, num_classes):
    """
    Compute per-class F1 and macro-F1 over valid (masked) frames.

    Args:
        predictions: (B, T) int tensor of predicted class indices
        targets:     (B, T) int tensor of ground-truth class indices
        mask:        (B, T) float tensor, 1 for valid frames, 0 for padding
    Returns:
        per_class_f1: dict  {class_idx: f1}
        macro_f1:     float
    """
    m = mask.bool().view(-1)
    preds_flat = predictions.view(-1)[m]
    targs_flat = targets.view(-1)[m]

    per_class_f1 = {}
    f1_sum = 0.0
    valid_classes = 0

    for c in range(num_classes):
        tp = ((preds_flat == c) & (targs_flat == c)).sum().float()
        fp = ((preds_flat == c) & (targs_flat != c)).sum().float()
        fn = ((preds_flat != c) & (targs_flat == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Only count classes that actually appear in ground truth
        if (targs_flat == c).any():
            per_class_f1[c] = f1.item()
            f1_sum += f1.item()
            valid_classes += 1
        else:
            per_class_f1[c] = float('nan')

    macro_f1 = f1_sum / valid_classes if valid_classes > 0 else 0.0
    return per_class_f1, macro_f1


class Trainer:
    def __init__(
        self,
        encoder,
        segmentor,
        early_stop_patience=30,
        early_stop_min_delta=0.0,
        early_stop_monitor="val_f1",  # "val_loss" or "val_f1"
    ):
        self.encoder = encoder
        self.segmentor = segmentor  # owns its own loss
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_monitor = early_stop_monitor
        self._best_val = None
        self._patience_counter = 0
        self._best_saved = None

        if not getattr(Trainer, "_stdout_sink_added", False):
            logger.add(sys.stdout, colorize=True, format="{message}")
            Trainer._stdout_sink_added = True

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        device,
        learning_rate=0.0005,
        val_batch_gen=None,
        weight_decay=0.0,
    ):
        self.segmentor.train()
        self.segmentor.to(device)  # class_weights buffer moves here too
        self.encoder.to(device)

        self._best_val = None
        self._patience_counter = 0
        self._best_saved = None

        os.makedirs(save_dir, exist_ok=True)
        log_handler_id = logger.add(
            os.path.join(save_dir, "train.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
        )

        trainable_params = [
            p
            for p in list(self.encoder.parameters()) + list(self.segmentor.parameters())
            if p.requires_grad
        ]
        optimizer = optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

        for epoch in range(num_epochs):
            # ---- Train ----
            epoch_loss, n_batches, correct, total = 0, 0, 0, 0
            all_preds, all_targets, all_masks = [], [], []

            while batch_gen.has_next():
                x, y, mask = batch_gen.next_batch(batch_size)
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                optimizer.zero_grad()

                raw = self.segmentor(self.encoder(x))
                loss, preds, y_al, mask_al = self.segmentor.compute_loss(raw, y, mask)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                correct += ((preds == y_al).float() * mask_al[:, 0, :]).sum().item()
                total += mask_al[:, 0, :].sum().item()
                all_preds.append(preds.detach().cpu())
                all_targets.append(y_al.detach().cpu())
                all_masks.append(mask_al[:, 0, :].detach().cpu())

            batch_gen.reset()
            train_loss = epoch_loss / n_batches
            train_acc = correct / total if total > 0 else 0.0
            _, train_f1 = compute_f1(
                torch.cat(all_preds),
                torch.cat(all_targets),
                torch.cat(all_masks),
                self.segmentor.num_classes,
            )

            # ---- Validation ----
            val_loss = val_acc = val_f1 = None
            if val_batch_gen is not None:
                self.encoder.eval()
                self.segmentor.eval()
                vl, vn, vc, vt = 0, 0, 0, 0
                vp, vgt, vm = [], [], []

                with torch.no_grad():
                    while val_batch_gen.has_next():
                        x, y, mask = val_batch_gen.next_batch(batch_size)
                        x, y, mask = x.to(device), y.to(device), mask.to(device)

                        raw = self.segmentor(self.encoder(x))
                        loss, preds, y_al, mask_al = self.segmentor.compute_loss(
                            raw, y, mask
                        )

                        vl += loss.item()
                        vn += 1
                        vc += ((preds == y_al).float() * mask_al[:, 0, :]).sum().item()
                        vt += mask_al[:, 0, :].sum().item()
                        vp.append(preds.cpu())
                        vgt.append(y_al.cpu())
                        vm.append(mask_al[:, 0, :].cpu())

                val_batch_gen.reset()
                val_loss = vl / vn
                val_acc = vc / vt if vt > 0 else 0.0
                _, val_f1 = compute_f1(
                    torch.cat(vp),
                    torch.cat(vgt),
                    torch.cat(vm),
                    self.segmentor.num_classes,
                )
                self.encoder.train()
                self.segmentor.train()

            # ---- Save best ----
            if val_batch_gen is not None:
                es_metric = val_f1 if self.early_stop_monitor == "val_f1" else val_loss
                higher = self.early_stop_monitor == "val_f1"
                is_best = (
                    self._best_saved is None
                    or (higher and es_metric > self._best_saved)
                    or (not higher and es_metric < self._best_saved)
                )
                if is_best:
                    self._best_saved = es_metric
                    torch.save(
                        self.segmentor.state_dict(),
                        os.path.join(save_dir, "best_segmentor.pt"),
                    )
                    torch.save(
                        self.encoder.state_dict(),
                        os.path.join(save_dir, "best_encoder.pt"),
                    )
                    logger.info(
                        f"  ↳ Saved best ({self.early_stop_monitor}={es_metric:.4f})"
                    )

            # ---- Log ----
            if val_loss is None:
                logger.info(
                    f"[epoch {epoch + 1}] loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f}"
                )
            else:
                logger.info(
                    f"[epoch {epoch + 1}] loss={train_loss:.4f}  acc={train_acc:.4f}  f1={train_f1:.4f}"
                    f"  | val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
                )

            # ---- Early stop ----
            if val_batch_gen is not None:
                es_metric = val_f1 if self.early_stop_monitor == "val_f1" else val_loss
                if self._early_stop_step(es_metric):
                    logger.info(
                        f"Early stop at epoch {epoch + 1} (best={self._best_val:.4f})"
                    )
                    break

        if val_batch_gen is None:
            torch.save(
                self.segmentor.state_dict(), os.path.join(save_dir, "best_segmentor.pt")
            )
            torch.save(
                self.encoder.state_dict(), os.path.join(save_dir, "best_encoder.pt")
            )

        try:
            logger.remove(log_handler_id)
        except ValueError:
            pass

    def predict(
        self,
        model_dir,
        results_dir,
        features_path,
        vid_list_file,
        actions_dict,
        device,
        sample_rate,
        checkpoint_name="best",
    ):
        os.makedirs(results_dir, exist_ok=True)
        self.segmentor.to(device)
        self.encoder.to(device)
        self.segmentor.load_state_dict(
            torch.load(
                os.path.join(model_dir, f"{checkpoint_name}_segmentor.pt"),
                map_location=device,
            )
        )
        self.encoder.load_state_dict(
            torch.load(
                os.path.join(model_dir, f"{checkpoint_name}_encoder.pt"),
                map_location=device,
            )
        )
        self.segmentor.eval()
        self.encoder.eval()
        idx_to_action = {v: k for k, v in actions_dict.items()}

        with torch.no_grad():
            for vid in open(vid_list_file).read().splitlines():
                features = np.load(
                    os.path.join(features_path, vid.split(".")[0] + ".npy")
                )
                features = (
                    features[:, ::sample_rate]
                    if features.shape[0] < features.shape[1]
                    else features[::sample_rate]
                )
                x = torch.tensor(features, dtype=torch.float, device=device)
                if x.dim() == 2 and x.shape[0] < x.shape[1]:
                    x = x.T
                x = x.unsqueeze(0)

                raw = self.segmentor(self.encoder(x))
                predicted = self.segmentor.predict(raw).squeeze(0).cpu().numpy()

                recognition = [
                    idx_to_action[int(c)] for c in predicted for _ in range(sample_rate)
                ]
                fname = vid.split("/")[-1].split(".")[0]
                with open(os.path.join(results_dir, fname), "w") as f:
                    f.write("### Frame level recognition: ###\n")
                    f.write(" ".join(recognition))
