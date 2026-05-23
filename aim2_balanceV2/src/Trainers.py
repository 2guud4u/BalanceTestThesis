import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
import numpy as np

sys.path.append("..")
from models.segmentors.MSTCNplus import MS_TCN2
from scripts.time_align import TimeAligner


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
            per_class_f1[c] = float("nan")

    macro_f1 = f1_sum / valid_classes if valid_classes > 0 else 0.0
    return per_class_f1, macro_f1


class Simple_MSTCN2_Trainer:
    def __init__(
        self,
        encoder,
        num_layers_PG=11,
        num_layers_R=10,
        num_R=3,
        num_f_maps=64,
        num_classes=2,
        class_weights=None,
        early_stop_patience=30,
        early_stop_min_delta=0.0,
        time_alignment="upsample_preds",
        early_stop_monitor="val_f1",
    ):
        """
        Args:
            time_alignment: "upsample_preds" (default) or "downsample_labels"
                - upsample_preds: interpolate predictions up to frame-level, compare with frame labels
                - downsample_labels: majority-vote labels down to patch-level, compare directly
            early_stop_monitor: "val_loss" (lower is better) or "val_f1" (higher is better)
        """
        self.encoder = encoder
        self.model = MS_TCN2(
            num_layers_PG, num_layers_R, num_R, num_f_maps, encoder.out_dim, num_classes
        )

        # Handle class weights
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)

        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes
        self.aligner = TimeAligner(strategy=time_alignment, num_classes=num_classes)

        # Early stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_monitor = early_stop_monitor  # "val_loss" or "val_f1"
        self._best_val = None
        self._patience_counter = 0
        self._best_saved = None

        # Add stdout sink once per process; remove any prior stdout sink first.
        if not getattr(Simple_MSTCN2_Trainer, "_stdout_sink_added", False):
            logger.add(sys.stdout, colorize=True, format="{message}")
            Simple_MSTCN2_Trainer._stdout_sink_added = True

    def _early_stop_step(self, val_metric):
        """
        Monitor validation metric. Returns True if training should stop.
        Supports both 'val_loss' (lower is better) and 'val_f1' (higher is better).
        """
        if val_metric is None:
            return False

        higher_is_better = self.early_stop_monitor == "val_f1"

        if higher_is_better:
            improved = self._best_val is None or val_metric > (
                self._best_val + self.early_stop_min_delta
            )
        else:
            improved = self._best_val is None or val_metric < (
                self._best_val - self.early_stop_min_delta
            )

        if improved:
            self._best_val = val_metric
            self._patience_counter = 0
            return False

        self._patience_counter += 1
        return self._patience_counter >= self.early_stop_patience

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        device,
        learning_rate=0.0005,
        val_batch_gen=None,
        lambda_smooth=0.15,
        weight_decay=0.0,
    ):
        self.model.train()
        self.model.to(device)
        self.encoder.to(device)

        # Reset early-stop / best-checkpoint state so multiple train() calls
        # on the same trainer instance don't carry over stale bests.
        self._best_val = None
        self._patience_counter = 0
        self._best_saved = None

        # Add file sink for this run, and capture handler id so we can remove it
        # at the end (otherwise repeated train() calls leak loguru handlers).
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, "train.log")
        log_handler_id = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
        )
        logger.info(f"class weight is {self.class_weights}")

        # Move class weights to device if they exist
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
            self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=self.class_weights)

        # Only optimize trainable parameters
        trainable_params = [
            p
            for p in list(self.encoder.parameters()) + list(self.model.parameters())
            if p.requires_grad
        ]
        optimizer = optim.Adam(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_train_batches = 0
            correct = 0
            total = 0
            train_all_preds = []
            train_all_targets = []
            train_all_masks = []
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                optimizer.zero_grad()

                # -------- Encoder forward pass --------
                # input B,T,D
                # Encoder outputs (B, D, T) directly for MS_TCN2 compatibility
                encoder_features = self.encoder(batch_input)  # (B, D, T)
                # -------- MS_TCN2 forward pass --------
                predictions = self.model(encoder_features)  # list[(B, C, T)]

                predictions_aligned, target_aligned, mask_aligned = self.aligner(
                    predictions, batch_target, mask
                )

                loss = 0
                for p in predictions_aligned:
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        target_aligned.view(-1),
                    )
                    loss += lambda_smooth * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=16,
                        )
                        * mask_aligned[:, :, 1:]
                    )

                epoch_loss += loss.item()
                num_train_batches += 1
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions_aligned[-1], 1)
                correct += (
                    (
                        (predicted == target_aligned).float()
                        * mask_aligned[:, 0, :].squeeze(1)
                    )
                    .sum()
                    .item()
                )
                total += torch.sum(mask_aligned[:, 0, :]).item()
                train_all_preds.append(predicted.detach().cpu())
                train_all_targets.append(target_aligned.detach().cpu())
                train_all_masks.append(mask_aligned[:, 0, :].squeeze(1).detach().cpu())

            batch_gen.reset()
            train_acc = float(correct) / total if total > 0 else 0.0
            _, train_f1 = compute_f1(
                torch.cat(train_all_preds),
                torch.cat(train_all_targets),
                torch.cat(train_all_masks),
                self.num_classes,
            )

            # -------- Validation --------
            val_loss = None
            val_acc = None
            val_f1 = None
            if val_batch_gen is not None:
                self.encoder.eval()
                self.model.eval()
                val_epoch_loss = 0
                num_val_batches = 0
                val_correct = 0
                val_total = 0
                val_all_preds = []
                val_all_targets = []
                val_all_masks = []

                with torch.no_grad():
                    while val_batch_gen.has_next():
                        val_input, val_target, val_mask = val_batch_gen.next_batch(
                            batch_size
                        )
                        val_input, val_target, val_mask = (
                            val_input.to(device),
                            val_target.to(device),
                            val_mask.to(device),
                        )

                        val_encoder_features = self.encoder(val_input)  # (B, D, T)
                        val_predictions = self.model(
                            val_encoder_features
                        )  # list[(B, C, T)]

                        val_preds_aligned, val_target_aligned, val_mask_aligned = (
                            self.aligner(val_predictions, val_target, val_mask)
                        )

                        val_loss_batch = 0
                        for p in val_preds_aligned:
                            val_loss_batch += self.ce(
                                p.transpose(2, 1)
                                .contiguous()
                                .view(-1, self.num_classes),
                                val_target_aligned.view(-1),
                            )
                            val_loss_batch += lambda_smooth * torch.mean(
                                torch.clamp(
                                    self.mse(
                                        F.log_softmax(p[:, :, 1:], dim=1),
                                        F.log_softmax(p.detach()[:, :, :-1], dim=1),
                                    ),
                                    min=0,
                                    max=16,
                                )
                                * val_mask_aligned[:, :, 1:]
                            )

                        val_epoch_loss += val_loss_batch.item()
                        num_val_batches += 1

                        _, val_predicted = torch.max(val_preds_aligned[-1], 1)
                        val_correct += (
                            (
                                (val_predicted == val_target_aligned).float()
                                * val_mask_aligned[:, 0, :].squeeze(1)
                            )
                            .sum()
                            .item()
                        )
                        val_total += torch.sum(val_mask_aligned[:, 0, :]).item()
                        val_all_preds.append(val_predicted.cpu())
                        val_all_targets.append(val_target_aligned.cpu())
                        val_all_masks.append(val_mask_aligned[:, 0, :].squeeze(1).cpu())

                val_batch_gen.reset()
                val_loss = (
                    val_epoch_loss / num_val_batches if num_val_batches > 0 else 0.0
                )
                val_acc = val_correct / val_total if val_total > 0 else 0.0
                _, val_f1 = compute_f1(
                    torch.cat(val_all_preds),
                    torch.cat(val_all_targets),
                    torch.cat(val_all_masks),
                    self.num_classes,
                )
                self.encoder.train()
                self.model.train()

            # -------- Save best model only --------
            if val_batch_gen is not None:
                es_metric = val_f1 if self.early_stop_monitor == "val_f1" else val_loss
                higher_is_better = self.early_stop_monitor == "val_f1"
                is_best = (
                    self._best_saved is None
                    or (higher_is_better and es_metric > self._best_saved)
                    or (not higher_is_better and es_metric < self._best_saved)
                )
                if is_best:
                    self._best_saved = es_metric
                    torch.save(self.model.state_dict(), save_dir + "/best.model")
                    torch.save(
                        self.encoder.state_dict(), save_dir + "/best_encoder.model"
                    )
                    logger.info(
                        f"  ↳ Saved best model ({self.early_stop_monitor}={es_metric:.4f})"
                    )

            # -------- Logging --------
            train_loss = (
                epoch_loss / num_train_batches if num_train_batches > 0 else 0.0
            )
            if val_loss is None:
                logger.info(
                    "[epoch %d]: train_loss = %f,   train_acc = %f,   train_f1 = %f"
                    % (epoch + 1, train_loss, train_acc, train_f1)
                )
            else:
                logger.info(
                    "[epoch %d]: train_loss = %f,   train_acc = %f,   train_f1 = %f,   val_loss = %f,   val_acc = %f,   val_f1 = %f"
                    % (
                        epoch + 1,
                        train_loss,
                        train_acc,
                        train_f1,
                        val_loss,
                        val_acc,
                        val_f1,
                    )
                )

            # -------- Early stopping --------
            if val_batch_gen is not None:
                es_metric = val_f1 if self.early_stop_monitor == "val_f1" else val_loss
                if self._early_stop_step(es_metric):
                    logger.info(
                        f"Early stopping at epoch {epoch + 1} "
                        f"({self.early_stop_monitor} did not improve for {self.early_stop_patience} epochs, "
                        f"best={self._best_val:.4f})"
                    )
                    break

        # If we ran without a val set, no "best.model" was saved during training.
        # Save the final-epoch weights so predict()/downstream eval has something to load.
        if val_batch_gen is None:
            torch.save(self.model.state_dict(), save_dir + "/best.model")
            torch.save(self.encoder.state_dict(), save_dir + "/best_encoder.model")
            logger.info("Saved final-epoch model (no val set provided).")

        # Clean up the per-run file handler so repeated train() calls don't leak.
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
        """
        Run inference on a list of videos and write per-frame recognitions.

        Args:
            model_dir:       directory containing the saved checkpoints
            results_dir:     directory to write per-video recognition files
            features_path:   directory of per-video .npy feature files
            vid_list_file:   text file with one video filename per line
            actions_dict:    {action_name: class_index}
            device:          torch device
            sample_rate:     temporal stride applied to features at inference
            checkpoint_name: filename stem for checkpoints. Loads
                             {checkpoint_name}.model + {checkpoint_name}_encoder.model
                             (defaults to "best", matching what train() saves).
        """
        os.makedirs(results_dir, exist_ok=True)

        # Load model + encoder weights saved by train()
        model_path = os.path.join(model_dir, f"{checkpoint_name}.model")
        encoder_path = os.path.join(model_dir, f"{checkpoint_name}_encoder.model")
        self.model.to(device)
        self.encoder.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.model.eval()
        self.encoder.eval()

        # Reverse lookup: class_index -> action_name
        idx_to_action = {v: k for k, v in actions_dict.items()}

        with torch.no_grad():
            with open(vid_list_file, "r") as file_ptr:
                list_of_vids = [line for line in file_ptr.read().split("\n") if line]

            for vid in list_of_vids:
                features = np.load(
                    os.path.join(features_path, vid.split(".")[0] + ".npy")
                )
                # Subsample temporally. Encoder expects (B, T, D); features may be (D, T).
                features = (
                    features[:, ::sample_rate]
                    if features.shape[0] < features.shape[1]
                    else features[::sample_rate, :]
                )
                input_x = torch.tensor(features, dtype=torch.float, device=device)
                # Ensure (B, T, D) layout for the encoder
                if input_x.dim() == 2 and input_x.shape[0] < input_x.shape[1]:
                    input_x = input_x.transpose(0, 1)  # (D, T) -> (T, D)
                input_x = input_x.unsqueeze(0)  # (1, T, D)

                # Forward through encoder, then MS-TCN — same path as training.
                encoder_features = self.encoder(input_x)  # (1, D, T)
                predictions = self.model(encoder_features)  # list[(1, C, T)]
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze(0).cpu().numpy()

                recognition = []
                for cls_idx in predicted:
                    recognition.extend([idx_to_action[int(cls_idx)]] * sample_rate)

                f_name = vid.split("/")[-1].split(".")[0]
                with open(os.path.join(results_dir, f_name), "w") as f_ptr:
                    f_ptr.write("### Frame level recognition: ###\n")
                    f_ptr.write(" ".join(recognition))
