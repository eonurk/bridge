import os
import shlex

from snakemake.exceptions import WorkflowError


def _bridge_cfg():
    if "bridge" not in config:
        raise WorkflowError("Missing 'bridge' section in config.yaml")
    return config["bridge"]


def _bundle_args() -> str:
    cfg = _bridge_cfg()
    bundle = os.getenv("BRIDGE_BUNDLE")
    bundle_url = os.getenv("BRIDGE_BUNDLE_URL")
    bundle_sha256 = os.getenv("BRIDGE_BUNDLE_SHA256")
    cache_dir = cfg.get("bundle_cache_dir", ".cache/bridge")

    if bundle and bundle_url:
        raise WorkflowError("Set only one of BRIDGE_BUNDLE or BRIDGE_BUNDLE_URL.")
    if bundle:
        return f"--bundle {shlex.quote(bundle)}"
    if bundle_url:
        parts = [
            f"--bundle-url {shlex.quote(bundle_url)}",
            f"--bundle-cache-dir {shlex.quote(str(cache_dir))}",
        ]
        if bundle_sha256:
            parts.append(f"--bundle-sha256 {shlex.quote(bundle_sha256)}")
        return " ".join(parts)

    raise WorkflowError(
        "Provide BRIDGE_BUNDLE or BRIDGE_BUNDLE_URL in the environment for bridge rules."
    )


def _attention_flag() -> str:
    enabled = bool(_bridge_cfg().get("use_attention", False))
    return "--use-attention" if enabled else "--no-use-attention"


BRIDGE_INPUT_PATTERN = str(_bridge_cfg()["input_pattern"])
BRIDGE_OUTPUT_PATTERN = f"{_bridge_cfg()['output_dir']}" + "/{sample}.predictions.csv"


rule bridge_predict_rna:
    input:
        BRIDGE_INPUT_PATTERN
    output:
        BRIDGE_OUTPUT_PATTERN
    threads:
        lambda wc: int(_bridge_cfg().get("threads", 1))
    params:
        bundle_args=lambda wc: _bundle_args(),
        batch_size=lambda wc: int(_bridge_cfg().get("batch_size", 256)),
        device=lambda wc: str(_bridge_cfg().get("device", "auto")),
        attention=lambda wc: _attention_flag(),
    shell:
        r"""
        python -m bridge predict-rna \
          {params.bundle_args} \
          --batch-size {params.batch_size} \
          --device {params.device:q} \
          {params.attention} \
          {input:q} \
          --combined-out {output:q}
        """
