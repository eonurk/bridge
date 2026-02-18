# Bridge

`Bridge` is a standalone Python package for applying the Bridge model to predict Acute Leukemia class to new RNA-seq count matrices.

## Access to bundle

The inference bundle (`bridge_inference.bundle`) is not distributed publicly in this repository.
If you want to run the package, please email us (the project maintainers) to request bundle access.

## Install

```bash
pip install -e ./Bridge
```

Conda package (published):

```bash
conda install -c eonurk -c conda-forge -c pytorch bridge
```

Package page: `https://anaconda.org/eonurk/bridge`

## Quick start (RNA count matrix -> predictions)

```python
from bridge import BridgePredictor

predictor = BridgePredictor(
    bundle="path/to/bridge_inference.bundle",
)

result = predictor.predict_rna("path/to/rna_count_matrix.csv")
print(result.combined.head())
predictor.close()
```

`result.combined` includes predictions, per-class probabilities, and RNA latents.
The bundle is provided by the model team and contains all required inference artifacts.

## CLI usage

After install, the package exposes a `bridge` command:

```bash
bridge --help
```

Predict labels from an RNA matrix:

```bash
bridge predict-rna \
  --bundle path/to/bridge_inference.bundle \
  path/to/rna_count_matrix.csv \
  --combined-out outputs/predictions.csv
```

Export RNA latents:

```bash
bridge encode-rna \
  --bundle path/to/bridge_inference.bundle \
  path/to/rna_count_matrix.csv \
  --output outputs/rna_latents.parquet
```

Export methylation latents:

```bash
bridge encode-methylation \
  --bundle path/to/bridge_inference.bundle \
  path/to/methylation_matrix.csv \
  --output outputs/meth_latents.csv
```

If you are not using a bundle, pass `--checkpoint` (plus `--metadata` if needed) and for
`predict-rna` also pass `--classifier`.

### Pipeline integration for large bundles

For existing pipelines, avoid storing large bundles in your repo by passing a URL and cache dir:

```bash
bridge predict-rna \
  --bundle-url https://example.org/path/bridge_inference.bundle \
  --bundle-cache-dir /shared/cache/bridge \
  --bundle-sha256 <expected_sha256> \
  input/rna_counts.csv \
  --combined-out output/predictions.csv
```

The bundle is downloaded once and reused from cache in later runs.

All model-source options support environment variable fallbacks:
- `BRIDGE_BUNDLE`, `BRIDGE_BUNDLE_URL`, `BRIDGE_BUNDLE_CACHE_DIR`, `BRIDGE_BUNDLE_SHA256`
- `BRIDGE_CHECKPOINT`, `BRIDGE_METADATA`, `BRIDGE_CLASSIFIER`

## License and use

- License: `CC-BY-NC-4.0`
- Non-commercial use only.
- For any commercial use (including commercial research, product integration, or deployment), you must email us to request a separate commercial-use license.
- See `LICENSE` for details.

The package auto-detects whether features are in rows or columns and aligns input features to the training metadata.

## Snakemake template

A drop-in Snakemake example is included at:

- `Bridge/examples/snakemake/Snakefile`
- `Bridge/examples/snakemake/rules/bridge.smk`
- `Bridge/examples/snakemake/config.yaml`
- `Bridge/examples/snakemake/README.md`

It is designed for private bundle delivery via environment variables (`BRIDGE_BUNDLE` or
`BRIDGE_BUNDLE_URL`) and keeps bundle paths/URLs out of committed config.

## Tests

Install test dependencies and run:

```bash
cd Bridge
pip install -e ".[test]"
PYTHONPATH=src pytest -q
```
