# Data Directory

This directory does NOT contain third-party raw data (C4 / DOLLY / Wiki).

download_datasets.py        # Download/construct datasets from the original providers (see DATA.md)
download_baseline_assets.py # Download third-party baseline weights/statistics (official URL + SHA256)

## Paper-Generated / Detection Data Package (records + scores)

The data package **IASW-data.zip** is released on Zenodo (sanitized, without
third-party human_text):

- DOI: https://doi.org/10.5281/zenodo.23601338
- Record page: https://zenodo.org/records/23601338

After downloading, unzip it next to this repository (e.g., create `../data/`
beside the `IASW/` repository root), then run
`analysis/paper/build_paper.py --data-root ../data` to rebuild the paper
tables/figures. See the repository-root `DATA.md` for the detailed layout and
reconstruction rules.
