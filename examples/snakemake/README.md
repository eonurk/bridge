# Bridge Snakemake Template

This template shows how to call the `bridge` CLI from Snakemake without
committing model bundles to the repository.

## Files

- `Snakefile`: entrypoint that includes Bridge rules.
- `rules/bridge.smk`: reusable Bridge prediction rule.
- `config.yaml`: non-secret pipeline settings.

## Secrets and bundle access

Set exactly one of these environment variables before running Snakemake:

1. Local private bundle path:

```bash
export BRIDGE_BUNDLE=/secure/path/bridge_inference.bundle
```

2. Private bundle URL (recommended for shared clusters):

```bash
export BRIDGE_BUNDLE_URL="https://private-storage.example/bridge_inference.bundle"
export BRIDGE_BUNDLE_SHA256="<expected_sha256>"
```

`BRIDGE_BUNDLE_SHA256` is optional but strongly recommended.

## Run

```bash
cd Bridge/examples/snakemake
snakemake -j 4
```

## Quick smoke run with local bundle

```bash
cd Bridge/examples/snakemake
export BRIDGE_BUNDLE=/Users/onur-lumc/Desktop/AML-bridge/artifacts/bridge_inference.bundle
export PYTHONPATH=/Users/onur-lumc/Desktop/AML-bridge/Bridge/src
snakemake -j 1
```

This example includes `input/demo1.csv`, a tiny single-sample RNA matrix.
