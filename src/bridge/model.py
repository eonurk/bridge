from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _build_mlp(input_dim: int, hidden_dim: int, latent_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, latent_dim))
    return nn.Sequential(*layers)


def _build_decoder(latent_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(latent_dim, hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = query.unsqueeze(1)
        k = key_value.unsqueeze(1)
        v = key_value.unsqueeze(1)
        attended, _ = self.attn(q, k, v, need_weights=False)
        return attended.squeeze(1)


class BridgeModule(nn.Module):
    """Checkpoint-compatible bridge module for inference and latent encoding."""

    def __init__(
        self,
        rna_dim: int,
        meth_dim: int,
        latent_dim: int,
        hidden_dim: int,
        lr: float,
        weight_decay: float,
        self_recon_weight: float,
        cross_weight: float,
        latent_align_weight: float,
        dropout: float,
        use_attention: bool,
        attention_heads: int,
        contrastive_weight: float,
        contrastive_temperature: float,
        rna_feature_dropout: float,
        meth_feature_dropout: float,
    ):
        super().__init__()

        self.rna_dim = int(rna_dim)
        self.meth_dim = int(meth_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_attention = bool(use_attention)

        # Preserve the old .hparams access pattern for compatibility.
        self.hparams = SimpleNamespace(
            rna_dim=rna_dim,
            meth_dim=meth_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            weight_decay=weight_decay,
            self_recon_weight=self_recon_weight,
            cross_weight=cross_weight,
            latent_align_weight=latent_align_weight,
            dropout=dropout,
            use_attention=use_attention,
            attention_heads=attention_heads,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            rna_feature_dropout=rna_feature_dropout,
            meth_feature_dropout=meth_feature_dropout,
        )

        self.encoder_rna = _build_mlp(rna_dim, hidden_dim, latent_dim, dropout)
        self.encoder_meth = _build_mlp(meth_dim, hidden_dim, latent_dim, dropout)
        self.decoder_rna = _build_decoder(latent_dim, hidden_dim, rna_dim, dropout)
        self.decoder_meth = _build_decoder(latent_dim, hidden_dim, meth_dim, dropout)

        if self.use_attention:
            self.cross_attn_rna = CrossAttention(latent_dim, attention_heads, dropout)
            self.cross_attn_meth = CrossAttention(latent_dim, attention_heads, dropout)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: Optional[torch.device | str] = None,
        strict: bool = True,
    ) -> "BridgeModule":
        """Load a model from a PyTorch Lightning .ckpt without importing Lightning."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=map_location)

        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError(
                "Unsupported checkpoint format. Expected a Lightning-style dict with 'state_dict'."
            )

        hparams = checkpoint.get("hyper_parameters")
        if not isinstance(hparams, dict):
            raise ValueError(
                "Checkpoint is missing 'hyper_parameters'. "
                "Cannot reconstruct BridgeModule architecture without it."
            )

        sig = inspect.signature(cls.__init__)
        init_kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in hparams:
                init_kwargs[name] = hparams[name]
            elif param.default is not inspect._empty:
                init_kwargs[name] = param.default
            else:
                raise ValueError(
                    f"Checkpoint hyper_parameters missing required field '{name}'."
                )

        model = cls(**init_kwargs)
        state_dict = checkpoint["state_dict"]
        incompatible = model.load_state_dict(state_dict, strict=strict)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Checkpoint state_dict mismatch. "
                f"Missing keys: {incompatible.missing_keys}, "
                f"Unexpected keys: {incompatible.unexpected_keys}"
            )

        model.eval()
        return model

    def forward(self, rna_batch: torch.Tensor, meth_batch: torch.Tensor) -> dict[str, torch.Tensor]:
        z_rna = self.encoder_rna(rna_batch)
        z_meth = self.encoder_meth(meth_batch)
        outputs: dict[str, torch.Tensor] = {
            "z_rna_raw": z_rna,
            "z_meth_raw": z_meth,
        }
        if self.use_attention:
            z_rna = z_rna + self.cross_attn_rna(z_rna, z_meth)
            z_meth = z_meth + self.cross_attn_meth(z_meth, z_rna)
        outputs.update(
            {
                "z_rna": z_rna,
                "z_meth": z_meth,
                "rna_self": self.decoder_rna(z_rna),
                "meth_self": self.decoder_meth(z_meth),
                "rna_cross": self.decoder_rna(z_meth),
                "meth_cross": self.decoder_meth(z_rna),
            }
        )
        return outputs

    def encode_pair(self, rna_batch: torch.Tensor, meth_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.encoder_rna(rna_batch), self.encoder_meth(meth_batch)

    @staticmethod
    def _nearest_reference(query_latents: torch.Tensor, reference_latents: torch.Tensor) -> torch.Tensor:
        query_norm = F.normalize(query_latents, p=2, dim=1)
        ref_norm = F.normalize(reference_latents, p=2, dim=1)
        sim = query_norm @ ref_norm.t()
        top_idx = torch.argmax(sim, dim=1)
        return reference_latents[top_idx]

    def encode_rna(
        self,
        rna_batch: torch.Tensor,
        reference_meth_latents: Optional[torch.Tensor] = None,
        use_attention: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_rna = self.encoder_rna(rna_batch)
            if (
                use_attention
                and self.use_attention
                and reference_meth_latents is not None
                and reference_meth_latents.numel() > 0
            ):
                ref = reference_meth_latents.to(z_rna.device)
                partner = self._nearest_reference(z_rna, ref)
                z_rna = z_rna + self.cross_attn_rna(z_rna, partner)
            return z_rna

    def encode_meth(
        self,
        meth_batch: torch.Tensor,
        reference_rna_latents: Optional[torch.Tensor] = None,
        use_attention: bool = False,
    ) -> torch.Tensor:
        with torch.no_grad():
            z_meth = self.encoder_meth(meth_batch)
            if (
                use_attention
                and self.use_attention
                and reference_rna_latents is not None
                and reference_rna_latents.numel() > 0
            ):
                ref = reference_rna_latents.to(z_meth.device)
                partner = self._nearest_reference(z_meth, ref)
                z_meth = z_meth + self.cross_attn_meth(z_meth, partner)
            return z_meth
