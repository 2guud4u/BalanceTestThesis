
class MAMP_MSTCN2(nn.Module):
    """
    Full model: frozen MAMP encoder + MS-TCN++.
    """
    def __init__(
        self,
        mamp_encoder,
        num_layers_PG: int,
        num_layers_R: int,
        num_R: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.encoder = mamp_encoder
        self.mstcn = MS_TCN2(
            num_layers_PG,
            num_layers_R,
            num_R,
            num_f_maps,
            dim,
            num_classes,
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)           # (B, D, T_patches)
        preds = self.mstcn(feats)         # list[(B, C, T_patches)]
        return preds


class MAMP_MSTCN2_Trainer:
    def __init__(
        self,
        model: MAMP_MSTCN2,
        num_classes: int,
        dataset: str,
        split: str,
        class_weights: None,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 0.0,
        lambda_smooth: float = 0.005,
        label_downsample: str = "majority",  # "majority" or "nearest"
        interp_mode_preds: str = "nearest",  # only used if you upsample preds
    ):
        self.model = model
        self.num_classes = num_classes
        self.lambda_smooth = lambda_smooth
        self.label_downsample = label_downsample
        self.interp_mode_preds = interp_mode_preds

        if class_weights is not None:
            class_weights = class_weights.float()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        self.mse = nn.MSELoss(reduction="none")

        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self._best_val = None
        self._patience_counter = 0

        logger.add(f"logs/{dataset}_{split}_{{time}}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def _early_stop_step(self, val_metric):
        if val_metric is None:
            return False
        if self._best_val is None or val_metric < (self._best_val - self.early_stop_min_delta):
            self._best_val = val_metric
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        return self._patience_counter > self.early_stop_patience

    def _downsample_labels_majority(self, target: torch.Tensor, mask: torch.Tensor, T_pred: int):
        """
        target: (B, T_frames)
        mask:   (B, 1, T_frames)
        returns:
            target_ds: (B, T_pred)
            mask_ds:   (B, 1, T_pred)
        """
        B, T_frames = target.shape
        if T_frames % T_pred != 0:
            # fallback: nearest
            target_ds = F.interpolate(
                target.unsqueeze(1).float(),
                size=T_pred,
                mode="nearest",
            ).squeeze(1).long()
            mask_ds = F.interpolate(mask.float(), size=T_pred, mode="nearest")
            return target_ds, mask_ds

        k = T_frames // T_pred  # frames per patch

        tgt = target.view(B, T_pred, k)                # (B, T_pred, k)
        m = mask[:, 0, :].view(B, T_pred, k).float()   # (B, T_pred, k)

        target_ds = torch.full((B, T_pred), -100, dtype=torch.long, device=target.device)

        # vectorized majority vote
        tgt_valid = tgt.masked_fill(m == 0, 0)
        one_hot = F.one_hot(tgt_valid.clamp(min=0), num_classes=self.num_classes).float()
        one_hot = one_hot * m.unsqueeze(-1)
        counts = one_hot.sum(dim=2)                    # (B, T_pred, num_classes)
        target_ds = counts.argmax(dim=-1)              # (B, T_pred)

        # patch valid if most frames in patch are valid
        mask_ds = (m.mean(dim=-1, keepdim=True) > 0.5).float()  # (B, T_pred, 1)
        mask_ds = mask_ds.permute(0, 2, 1).contiguous()         # (B, 1, T_pred)

        return target_ds, mask_ds

    def _align_time(self, predictions, target, mask):
        """
        Align target/mask to the temporal length of predictions[-1].
        """
        T_frames = target.shape[1]
        T_pred = predictions[-1].shape[-1]
        if T_frames == T_pred:
            return predictions, target, mask

        if self.label_downsample == "majority":
            target_aligned, mask_aligned = self._downsample_labels_majority(target, mask, T_pred)
        else:
            target_aligned = F.interpolate(
                target.unsqueeze(1).float(),
                size=T_pred,
                mode="nearest",
            ).squeeze(1).long()
            mask_aligned = F.interpolate(mask.float(), size=T_pred, mode="nearest")

        return predictions, target_aligned, mask_aligned

    def _compute_loss(self, predictions, target, mask):
        """
        predictions: list[(B, C, T)]
        target: (B, T)
        mask: (B, 1, T)
        """
        loss = 0.0

        for p in predictions:
            ce_loss = self.ce(
                p.permute(0, 2, 1).reshape(-1, self.num_classes),
                target.reshape(-1),
            )

            log_p = F.log_softmax(p, dim=1)
            smooth = self.mse(
                log_p[:, :, 1:],
                log_p.detach()[:, :, :-1],
            ).mean(dim=1)  # (B, T-1)

            smooth = smooth * mask[:, 0, 1:]
            smooth_loss = torch.mean(torch.clamp(smooth, 0, 16))

            loss += ce_loss + self.lambda_smooth * smooth_loss

        return loss / len(predictions)

    def train(
        self,
        save_dir,
        batch_gen,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        val_batch_gen=None,
    ):
        self.model.to(device)
        self.model.train()

        optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            train_correct = 0.0
            train_total = 0.0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)

                batch_input = batch_input.to(device)   # (B, T, V, 3)
                batch_target = batch_target.to(device) # (B, T)
                mask = mask.to(device)                 # (B, 1, T)

                optimizer.zero_grad()

                predictions = self.model(batch_input)  # list[(B, C, T_patch)]
                predictions, batch_target_aligned, mask_aligned = self._align_time(
                    predictions=predictions,
                    target=batch_target,
                    mask=mask,
                )

                loss = self._compute_loss(predictions, batch_target_aligned, mask_aligned)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    5.0,
                )
                optimizer.step()

                epoch_loss += loss.item()

                with torch.no_grad():
                    final_pred = predictions[-1].argmax(dim=1)   # (B, T_patch)
                    m = mask_aligned[:, 0, :]
                    train_correct += ((final_pred == batch_target_aligned).float() * m).sum().item()
                    train_total += m.sum().item()

            batch_gen.reset()
            train_acc = train_correct / train_total if train_total > 0 else 0.0

            val_loss = None
            val_acc = None
            if val_batch_gen is not None:
                self.model.eval()
                val_correct = 0.0
                val_total = 0.0
                val_loss_total = 0.0
                val_loss_count = 0

                with torch.no_grad():
                    while val_batch_gen.has_next():
                        v_input, v_target, v_mask = val_batch_gen.next_batch(batch_size)
                        v_input = v_input.to(device)
                        v_target = v_target.to(device)
                        v_mask = v_mask.to(device)

                        preds = self.model(v_input)
                        preds, v_target_aligned, v_mask_aligned = self._align_time(
                            predictions=preds,
                            target=v_target,
                            mask=v_mask,
                        )

                        batch_loss = self._compute_loss(preds[-1:], v_target_aligned, v_mask_aligned)
                        val_loss_total += batch_loss.item()
                        val_loss_count += 1

                        final_pred = preds[-1].argmax(dim=1)
                        m = v_mask_aligned[:, 0, :]
                        val_correct += ((final_pred == v_target_aligned).float() * m).sum().item()
                        val_total += m.sum().item()

                    val_batch_gen.reset()

                val_loss = val_loss_total / max(1, val_loss_count)
                val_acc = val_correct / max(1.0, val_total)
                self.model.train()

            if val_loss is None:
                logger.info(f"[Epoch {epoch+1}] loss={epoch_loss:.4f}, train_acc={train_acc:.4f}")
            else:
                logger.info(
                    f"[Epoch {epoch+1}] loss={epoch_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )

            # save checkpoint
            torch.save(self.model.state_dict(), f"{save_dir}/epoch-{epoch+1}.model")
            torch.save(optimizer.state_dict(), f"{save_dir}/epoch-{epoch+1}.opt")

            # early stopping
            if val_batch_gen is not None and self._early_stop_step(val_loss):
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(val_loss did not improve for {self.early_stop_patience} epochs)"
                )
                break