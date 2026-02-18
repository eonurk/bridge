from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

from .bundle_source import env_path, resolve_bundle_path

if TYPE_CHECKING:
    from .predictor import BridgePredictor


def _add_model_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Path to bridge inference bundle (.bundle). Env fallback: BRIDGE_BUNDLE",
    )
    parser.add_argument(
        "--bundle-url",
        help="Bundle URL to download and cache. Env fallback: BRIDGE_BUNDLE_URL",
    )
    parser.add_argument(
        "--bundle-cache-dir",
        type=Path,
        help=(
            "Cache directory for --bundle-url (default: ~/.cache/bridge). "
            "Env fallback: BRIDGE_BUNDLE_CACHE_DIR"
        ),
    )
    parser.add_argument(
        "--bundle-sha256",
        help="Optional checksum validation for bundle path/url. Env fallback: BRIDGE_BUNDLE_SHA256",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to Bridge checkpoint (.ckpt). Env fallback: BRIDGE_CHECKPOINT",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to preprocessing metadata (.joblib). Env fallback: BRIDGE_METADATA",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        help="Path to latent classifier (.joblib). Env fallback: BRIDGE_CLASSIFIER",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto/cpu/cuda/mps (default: auto)",
    )


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument(
        "--use-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable attention-based encoder pathway when available (default: False)",
    )


def _write_frame(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output, index=False)
    elif suffix in {".tsv", ".txt"}:
        df.to_csv(output, sep="\t", index=False)
    else:
        df.to_csv(output, index=False)


def _validate_model_source(
    *,
    bundle: Path | None,
    checkpoint: Path | None,
    metadata: Path | None,
    classifier: Path | None,
    require_classifier: bool,
) -> None:
    if bundle is not None:
        if checkpoint is not None or metadata is not None or classifier is not None:
            raise ValueError(
                "When --bundle is provided, do not also pass --checkpoint/--metadata/--classifier."
            )
        return

    if checkpoint is None:
        raise ValueError("Provide either --bundle or --checkpoint.")
    if require_classifier and classifier is None:
        raise ValueError("predict-rna requires --classifier when --bundle is not used.")


def _build_predictor(args: argparse.Namespace, *, require_classifier: bool) -> BridgePredictor:
    from .predictor import BridgePredictor

    bundle = resolve_bundle_path(
        bundle=args.bundle or env_path("BRIDGE_BUNDLE"),
        bundle_url=args.bundle_url or os.getenv("BRIDGE_BUNDLE_URL"),
        bundle_cache_dir=args.bundle_cache_dir or env_path("BRIDGE_BUNDLE_CACHE_DIR"),
        bundle_sha256=args.bundle_sha256 or os.getenv("BRIDGE_BUNDLE_SHA256"),
    )
    checkpoint = args.checkpoint or env_path("BRIDGE_CHECKPOINT")
    metadata = args.metadata or env_path("BRIDGE_METADATA")
    classifier = args.classifier or env_path("BRIDGE_CLASSIFIER")

    _validate_model_source(
        bundle=bundle,
        checkpoint=checkpoint,
        metadata=metadata,
        classifier=classifier,
        require_classifier=require_classifier,
    )
    return BridgePredictor(
        checkpoint=checkpoint,
        metadata=metadata,
        classifier=classifier,
        bundle=bundle,
        device=args.device,
    )


def _cmd_predict_rna(args: argparse.Namespace) -> int:
    predictor = _build_predictor(args, require_classifier=True)
    try:
        result = predictor.predict_rna(
            args.rna_counts,
            batch_size=args.batch_size,
            use_attention=args.use_attention,
        )
    finally:
        predictor.close()

    _write_frame(result.combined, args.combined_out)
    if args.predictions_out is not None:
        _write_frame(result.predictions, args.predictions_out)
    if args.probabilities_out is not None:
        _write_frame(result.probabilities, args.probabilities_out)
    if args.latents_out is not None:
        _write_frame(result.latents, args.latents_out)
    print(f"Wrote {len(result.combined)} predictions to {args.combined_out}")
    return 0


def _cmd_encode_rna(args: argparse.Namespace) -> int:
    predictor = _build_predictor(args, require_classifier=False)
    try:
        latents = predictor.encode_rna(
            args.rna_counts,
            batch_size=args.batch_size,
            use_attention=args.use_attention,
        )
    finally:
        predictor.close()

    _write_frame(latents, args.output)
    print(f"Wrote {len(latents)} RNA latents to {args.output}")
    return 0


def _cmd_encode_methylation(args: argparse.Namespace) -> int:
    predictor = _build_predictor(args, require_classifier=False)
    try:
        latents = predictor.encode_methylation(
            args.methylation_values,
            batch_size=args.batch_size,
            use_attention=args.use_attention,
        )
    finally:
        predictor.close()

    _write_frame(latents, args.output)
    print(f"Wrote {len(latents)} methylation latents to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bridge",
        description="Bridge CLI for RNA/methylation latent encoding and RNA class prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict-rna",
        help="Predict labels from an RNA count matrix",
    )
    _add_model_source_args(predict)
    _add_inference_args(predict)
    predict.add_argument("rna_counts", type=Path, help="RNA count matrix (.csv/.tsv/.parquet)")
    predict.add_argument(
        "--combined-out",
        type=Path,
        required=True,
        help="Combined prediction output path (.csv/.tsv/.parquet)",
    )
    predict.add_argument(
        "--predictions-out",
        type=Path,
        help="Optional per-sample predictions output path",
    )
    predict.add_argument(
        "--probabilities-out",
        type=Path,
        help="Optional per-class probabilities output path",
    )
    predict.add_argument(
        "--latents-out",
        type=Path,
        help="Optional latent features output path",
    )
    predict.set_defaults(func=_cmd_predict_rna)

    encode_rna = subparsers.add_parser(
        "encode-rna",
        help="Encode an RNA count matrix into Bridge RNA latents",
    )
    _add_model_source_args(encode_rna)
    _add_inference_args(encode_rna)
    encode_rna.add_argument("rna_counts", type=Path, help="RNA count matrix (.csv/.tsv/.parquet)")
    encode_rna.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for RNA latents (.csv/.tsv/.parquet)",
    )
    encode_rna.set_defaults(func=_cmd_encode_rna)

    encode_meth = subparsers.add_parser(
        "encode-methylation",
        help="Encode a methylation matrix into Bridge methylation latents",
    )
    _add_model_source_args(encode_meth)
    _add_inference_args(encode_meth)
    encode_meth.add_argument(
        "methylation_values",
        type=Path,
        help="Methylation matrix (.csv/.tsv/.parquet)",
    )
    encode_meth.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for methylation latents (.csv/.tsv/.parquet)",
    )
    encode_meth.set_defaults(func=_cmd_encode_methylation)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        parser.exit(2, f"Error: {exc}\n")
