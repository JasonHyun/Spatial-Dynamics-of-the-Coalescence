# A coalescent-spatial model to uncover the architectural principles of colorectal cancer

A computational pipeline for analyzing spatial dynamics of tumor evolution in colorectal cancer using coalescent theory. This project processes whole exome sequencing (WXS) data from cumulated datasets to reconstruct coalescent trees, test for neutral evolution, and generate spatial representations of tumor lineage dynamics.

## Overview

The pipeline integrates population genetics principles with cancer genomics to:
- **Test for neutral evolution** using the 1/f power law framework
- **Reconstruct coalescent trees** from somatic mutation data using Kingman's coalescent
- **Generate spatial mappings** of lineage distributions using Gaussian diffusion
- **Calculate spatial metrics** including Moran's I, within-lineage spread (α), and founder separation (β)

## Quick Start

### Requirements

- Python 3.10+
- Conda (recommended)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd thesis

# Create conda environment
conda env create -f code/environment.yml
conda activate thesis-coalescent
```

### Data Setup

**Note**: The somatic mutation information data has been downloaded from the GDC Data Portal using the data transfer tool and the included manifest file under snv maf. All data  obtained from the NCI Genomic Data Commons.


### Running the Pipeline

```bash
cd code
bash pipeline.sh
```

Run individual steps:
```bash
bash pipeline.sh --preprocess-only      # Step 1: Preprocessing
bash pipeline.sh --neutrality-only      # Step 2: Neutrality testing
bash pipeline.sh --coalescent-only       # Step 3: Coalescent reconstruction
bash pipeline.sh --spatial-only         # Step 4: Spatial mapping
```

## Pipeline Steps

1. **Preprocessing**: Filters and prepares MAF files
2. **Neutrality Testing**: Tests for neutral evolution using 1/f power law (R² ≥ 0.98 threshold)
3. **Coalescent Tree Reconstruction**: Builds coalescent-compliant trees with VAF-based N_e estimation
4. **Spatial Mapping**: Applies Gaussian diffusion to generate spatial representations and calculate metrics

## Output Structure

```
results/
├── neutrality_tests/          # Neutrality test results and plots
├── coalescent_trees/          # Coalescent tree reconstructions
└── spatial_mapping/          # Spatial mappings and metrics
```

## Key Features

- **Coalescent-compliant tree reconstruction** following Kingman's coalescent principles
- **Deterministic reproducibility** using MD5 hash-based seeding
- **Spatial autocorrelation analysis** via Moran's I
- **Lineage partitioning** and spatial metric calculation
- **Batch processing** of 800+ patient samples

## Project Structure

```
thesis/
├── code/              # Pipeline scripts
├── data/              # MAF data directory 
└── results/           # Analysis outputs
```


