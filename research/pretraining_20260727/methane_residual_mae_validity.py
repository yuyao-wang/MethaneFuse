"""Validity-aware, parameter-compatible Residual-MAE adapter.

This module deliberately leaves the reference implementation in
``/home/yuyao/NormWear`` unchanged.  The subclass adds no parameters or
buffers, so its state-dict keys and tensor shapes are identical to
``MethaneResidualMAE``.  It is selected only by the runner's explicit
``--validity-masked-reconstruction`` flag.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from modules.methane_residual_mae import MethaneResidualMAE


VALIDITY_OBJECTIVE_VERSION = "masked_valid_element_per_sample_stream_v1"
VALIDITY_SEMANTICS = (
    "per_channel_native_validity;"
    "tiff_finite_nonzero;"
    "s5p_finite;"
    "residual_validity_intersection;"
    "resize_area_down_nearest_up;"
    f"{VALIDITY_OBJECTIVE_VERSION}"
)


class ValidityMaskedMethaneResidualMAE(MethaneResidualMAE):
    """Residual-MAE whose reconstruction loss excludes invalid elements."""

    def _prepare_target_validity(
        self,
        residuals: Mapping[str, torch.Tensor],
        validity_by_stream: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        residual_keys = set(residuals)
        validity_keys = set(validity_by_stream)
        if residual_keys != validity_keys:
            raise ValueError(
                "Validity keys must exactly match residual stream keys: "
                f"residual_only={sorted(residual_keys - validity_keys)}, "
                f"validity_only={sorted(validity_keys - residual_keys)}"
            )

        targets: Dict[str, torch.Tensor] = {}
        for stream, images in residuals.items():
            validity = validity_by_stream[stream]
            if tuple(validity.shape) != tuple(images.shape):
                raise ValueError(
                    f"{stream} validity shape {tuple(validity.shape)} does not "
                    f"match input shape {tuple(images.shape)}"
                )
            validity = validity.to(device=images.device, dtype=torch.float32)
            if not bool(torch.isfinite(validity).all().item()):
                raise ValueError(f"{stream} validity contains non-finite values")
            targets[stream] = self.patchify(validity.clamp(0.0, 1.0))
        return targets

    def forward_encoder_pretrain(
        self,
        residuals: Mapping[str, torch.Tensor],
        validity_by_stream: Mapping[str, torch.Tensor],
        sensor_present: Optional[torch.Tensor] = None,
    ):
        present_by_stream = self.stream_present_dict(residuals, sensor_present)
        target_validity = self._prepare_target_validity(
            residuals,
            validity_by_stream,
        )
        tokens_by_stream = {}
        masks = {}
        ids_restore = {}
        targets = {}

        for stream, images in residuals.items():
            targets[stream] = self.patchify(images)
            validity_present = (
                target_validity[stream].sum(dim=(1, 2)) > 0
            )
            present_by_stream[stream] = (
                present_by_stream[stream].bool() & validity_present
            )
            patch_tokens = self.patch_embed(stream, images)
            patch_tokens = patch_tokens + self.pos_embed[:, 1:, :]
            (
                patch_tokens,
                masks[stream],
                ids_restore[stream],
            ) = self.random_masking(
                patch_tokens,
                self.mask_ratio,
            )
            tokens_by_stream[stream] = self.prepend_cls(stream, patch_tokens)

        latent = self.encode_tokens(tokens_by_stream, present_by_stream)
        return (
            latent,
            masks,
            ids_restore,
            targets,
            target_validity,
            present_by_stream,
        )

    def forward_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        present_by_stream: Mapping[str, torch.Tensor],
        target_validity: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        if target_validity is None:
            return super().forward_loss(
                preds,
                targets,
                masks,
                present_by_stream,
            )

        stream_losses = []
        stream_has_valid = []
        metrics = {}
        all_valid_legacy_equivalent = True

        for stream, prediction in preds.items():
            target = targets[stream]
            validity = target_validity[stream].to(
                device=prediction.device,
                dtype=torch.float32,
            )
            if tuple(validity.shape) != tuple(prediction.shape):
                raise ValueError(
                    f"{stream} patchified validity shape "
                    f"{tuple(validity.shape)} does not match prediction shape "
                    f"{tuple(prediction.shape)}"
                )

            reconstruction_mask = masks[stream].to(
                device=prediction.device,
                dtype=torch.float32,
            ).unsqueeze(-1)
            sample_present = present_by_stream[stream].to(
                device=prediction.device,
                dtype=torch.float32,
            ).view(-1, 1, 1)
            all_valid_legacy_equivalent = (
                all_valid_legacy_equivalent
                and bool((validity == 1).all().item())
                and bool((sample_present == 1).all().item())
            )
            effective_validity = (
                validity.clamp(0.0, 1.0)
                * reconstruction_mask
                * sample_present
            )

            squared_error = (
                prediction.float() - target.float()
            ).pow(2)
            sample_numerator = (
                squared_error * effective_validity
            ).sum(dim=(1, 2))
            sample_denominator = effective_validity.sum(dim=(1, 2))
            sample_has_valid = sample_denominator > 0
            sample_loss = sample_numerator / sample_denominator.clamp_min(1.0)
            valid_sample_count = sample_has_valid.float().sum()
            batch_sample_count = prediction.new_tensor(
                float(prediction.shape[0]),
                dtype=torch.float32,
            )
            stream_loss = (
                sample_loss * sample_has_valid.float()
            ).sum() / valid_sample_count.clamp_min(1.0)
            has_valid = (valid_sample_count > 0).to(dtype=stream_loss.dtype)

            stream_losses.append(stream_loss)
            stream_has_valid.append(has_valid)
            metrics[f"loss_{stream}"] = stream_loss.detach()
            metrics[f"valid_fraction_{stream}"] = (
                validity.mean().detach()
            )
            metrics[f"valid_masked_elements_{stream}"] = (
                sample_denominator.sum().detach()
            )
            metrics[f"valid_samples_{stream}"] = (
                valid_sample_count.detach()
            )
            metrics[f"excluded_samples_{stream}"] = (
                batch_sample_count - valid_sample_count
            ).detach()
            metrics[f"batch_samples_{stream}"] = (
                batch_sample_count.detach()
            )
            metrics[f"empty_stream_{stream}"] = (
                1.0 - has_valid
            ).detach()

        if not stream_losses:
            raise ValueError("No residual stream predictions were produced.")

        stream_loss_tensor = torch.stack(stream_losses)
        stream_valid_tensor = torch.stack(stream_has_valid)
        valid_stream_count = stream_valid_tensor.sum()
        if not bool((valid_stream_count > 0).item()):
            raise ValueError(
                "Validity-masked reconstruction batch has no valid masked "
                "elements in any stream."
            )
        loss = (
            stream_loss_tensor * stream_valid_tensor
        ).sum() / valid_stream_count
        metrics["valid_streams"] = valid_stream_count.detach()
        metrics["empty_streams"] = (
            stream_valid_tensor.numel() - valid_stream_count
        ).detach()
        metrics["total_streams"] = stream_loss_tensor.new_tensor(
            float(stream_loss_tensor.numel())
        ).detach()
        if all_valid_legacy_equivalent:
            # Preserve the legacy objective bit-for-bit in the all-valid
            # regime.  The mathematically equivalent per-element reduction
            # above can differ by one float32 ULP because its summation order
            # is different.
            legacy_loss, legacy_metrics = super().forward_loss(
                preds,
                targets,
                masks,
                present_by_stream,
            )
            metrics.update(legacy_metrics)
            loss = legacy_loss
        return loss, metrics

    def forward(
        self,
        residuals: Mapping[str, torch.Tensor],
        sensor_present: Optional[torch.Tensor] = None,
        *,
        validity_by_stream: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        if not self.is_pretrain:
            return self.forward_features(residuals, sensor_present)
        if validity_by_stream is None:
            raise ValueError(
                "ValidityMaskedMethaneResidualMAE pretraining requires "
                "validity_by_stream."
            )

        (
            latent,
            masks,
            ids_restore,
            targets,
            target_validity,
            present_by_stream,
        ) = self.forward_encoder_pretrain(
            residuals,
            validity_by_stream,
            sensor_present,
        )
        preds = {
            stream: self.forward_decoder(
                stream,
                tokens,
                ids_restore[stream],
            )
            for stream, tokens in latent.items()
        }
        loss, metrics = self.forward_loss(
            preds,
            targets,
            masks,
            present_by_stream,
            target_validity,
        )
        return loss, preds, masks, metrics
