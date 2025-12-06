# AIMarkerFinder

**AIMarkerFinder** is a computational tool designed for the analysis of high-dimensional biomedical data (e.g., metabolomic or genomic datasets) to identify minimal sets of informative biomarkers. The method combines a **denoising autoencoder with an attention mechanism (DAE)** and a **Kolmogorov–Arnold Network (KAN)** to perform feature selection and build interpretable classification models. This approach effectively addresses the "curse of dimensionality" and feature redundancy, enhancing both model accuracy and interpretability.

> Preprint: [https://www.preprints.org/manuscript/202511.1705](https://www.preprints.org/manuscript/202511.1705)

---

## Installation

### Requirements
- Python ≥ 3.10
- [Poetry](https://python-poetry.org/) for dependency management

### Setup
Install dependencies using Poetry:

```bash
poetry install
```

### Quick Start

View all available arguments:
```bash
poetry run aimf -h
```

Basic command syntax:
```bash
poetry run aimf <input_file> [options]
```
Supported input formats: .csv, .tsv, .xls, .xlsx  
Class labels: Specify with -CC; if omitted, all non-numeric columns are used by default.
    
Example usage with an input file example/test_data.xlsx:
```bash
poetry run aimf example/test_data.tsv -balance -CC Diagnose
```


