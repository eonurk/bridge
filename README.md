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

## License and use

- License: `CC-BY-NC-4.0`
- Non-commercial use only.
- For any commercial use (including commercial research, product integration, or deployment), you must email us to request a separate commercial-use license.
- See `LICENSE` for details.

## Additional API usage

Export RNA latents:

```python
rna_latents = predictor.encode_rna(
    "path/to/rna_count_matrix.csv"
)
```

The package auto-detects whether features are in rows or columns and aligns input features to the training metadata.
