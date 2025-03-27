# Using DVC with the SEIR Model

This guide demonstrates how to use Data Version Control (DVC) with the SEIR
compartmental model for epidemiology. DVC helps you version data and models,
create reproducible pipelines, track experiments, and manage large files
efficiently.

## Table of Contents

1. [Introduction to DVC](#introduction-to-dvc)
2. [Setting Up DVC](#setting-up-dvc)
3. [Tracking Data and Models](#tracking-data-and-models)
4. [Creating Reproducible Pipelines](#creating-reproducible-pipelines)
5. [Tracking Experiments](#tracking-experiments)
6. [Comparing Results](#comparing-results)
7. [Best Practices](#best-practices)

## Introduction to DVC

DVC (Data Version Control) is an open-source tool that helps data scientists
and ML engineers manage:

- Large data files and datasets
- ML models and their versions
- Reproducible pipelines
- Experiment tracking and comparison

DVC works alongside Git, extending its capabilities to handle large files
efficiently without storing them directly in your Git repository. Instead, DVC
stores references to these files in Git while the actual data is stored in
remote storage (like S3, Google Drive, or local storage).

## Setting Up DVC

### Prerequisites

- Python 3.6+
- Git repository initialized
- SEIR model code and data

### Installation

```bash
pipx install dvc
```

### Initialize DVC

Navigate to your project directory and initialize DVC:

```bash
# Assuming you already have a Git repository
git init  # If you don't already have a Git repo
dvc init
git commit -m "Initialize DVC"
```

### Configure Azure File Storage as Remote Storage

DVC supports Azure Blob Storage and Azure File Storage for remote storage. For
this guide, we'll use Azure File Storage.

First, install the Azure dependencies:

```bash
pip install dvc[azure]
```

Next, set up Azure File Storage as your remote storage:

```bash
# Configure Azure File Storage as remote
dvc remote add -d azure azure://myshare/path

# Set Azure connection string
dvc remote modify azure connection_string "DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"

# Or use environment variables (recommended for security)
# export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"

git add .dvc/config
git commit -m "Configure Azure File Storage for DVC remote storage"
```

Note: Replace `<account_name>` and `<account_key>` with your Azure Storage
account name and key. For security, it's recommended to use environment
variables rather than storing credentials in the DVC config file.

## Tracking Data and Models

### Track Vaccination Data

```bash
# Add the vaccination data file to DVC
dvc add data/covid19vax_trends_us.parquet

# Add the .dvc file to Git
git add data/covid19vax_trends_us.parquet.dvc
git commit -m "Track vaccination data with DVC"
```

### Track SEIR Model

```bash
# Add the model file to DVC
dvc add models/seir_model.pkl
dvc add models/vax_trends.pkl

# Add the .dvc files to Git
git add models/*.dvc
git commit -m "Track SEIR model and vaccination trends model with DVC"
```

### Push Data to Azure File Storage

Push your tracked data and models to Azure File Storage:

```bash
# Push all DVC-tracked files to Azure
dvc push

# To push specific files
dvc push data/covid19vax_trends_us.parquet.dvc
dvc push models/seir_model.pkl.dvc
```

If you're using environment variables for authentication, make sure they're set
before running the push command:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"
dvc push
```

## Creating Reproducible Pipelines

DVC allows you to define reproducible pipelines that transform data and
generate models. Let's create a pipeline for the SEIR model:

### Define Pipeline Stages

Create a `dvc.yaml` file:

```yaml
stages:
  prepare_vax_data:
    cmd: python scripts/prepare_vax_data.py
    deps:
      - scripts/prepare_vax_data.py
      - raw_data/covid19vax_trends.parquet
    outs:
      - data/covid19vax_trends_us.parquet
  
  create_vax_model:
    cmd: python scripts/create_vax_model.py
    deps:
      - scripts/create_vax_model.py
      - data/covid19vax_trends_us.parquet
    outs:
      - models/vax_trends.pkl
  
  run_seir_simulation:
    cmd: python scripts/run_seir_simulation.py --beta ${beta} --sigma ${sigma} --gamma ${gamma} --vax_eff ${vax_eff}
    deps:
      - scripts/run_seir_simulation.py
      - models/vax_trends.pkl
    params:
      - params.yaml:
          - model.beta
          - model.sigma
          - model.gamma
          - model.vax_eff
    outs:
      - models/seir_model.pkl
    metrics:
      - metrics.json:
          cache: false
```

### Create Parameter File

Create a `params.yaml` file:

```yaml
model:
  beta: 0.2    # Infection rate
  sigma: 0.2   # Incubation rate
  gamma: 0.05  # Recovery rate
  vax_eff: 0.5 # Vaccine efficacy
```

### Create Script Files

Create the necessary Python scripts:

1. `scripts/prepare_vax_data.py`: Script to prepare vaccination data
2. `scripts/create_vax_model.py`: Script to create the vaccination trends model
3. `scripts/run_seir_simulation.py`: Script to run the SEIR simulation

These scripts should contain the same logic as in the Jupyter notebook but
adapted for command-line execution.

### Run the Pipeline

```bash
dvc repro
```

This command will run all stages of the pipeline in the correct order, skipping
stages that don't need to be rerun.

## Tracking Experiments

DVC allows you to track different experiments with varying parameters:

### Modify Parameters

```bash
# Edit params.yaml to change parameters
# For example, change beta to 0.3
```

### Run Experiment

```bash
dvc exp run --name "higher_beta" --set-param model.beta=0.3
```

### List Experiments

```bash
dvc exp list
```

### Apply Experiment

```bash
dvc exp apply higher_beta
```

## Comparing Results

DVC provides tools to compare experiment results:

### Compare Metrics

```bash
dvc metrics show
```

### Compare Experiments

```bash
dvc exp diff
```

### Visualize Results

```bash
dvc plots show
```

## Best Practices

1. **Organize Your Project Structure**:
   ```
   seir_demo/
   ├── data/                  # Processed data
   ├── raw_data/              # Raw data
   ├── models/                # Saved models
   ├── scripts/               # Python scripts
   ├── notebooks/             # Jupyter notebooks
   ├── dvc.yaml               # Pipeline definition
   ├── params.yaml            # Parameters
   └── .gitignore             # Git ignore file
   ```

2. **Version Control Best Practices**:
   - Commit `.dvc` files to Git
   - Use meaningful commit messages
   - Create branches for different experiments

3. **Pipeline Design**:
   - Break down your workflow into logical stages
   - Make each stage idempotent (can be run multiple times with the same result)
   - Use parameters for values that might change

4. **Experiment Management**:
   - Use descriptive names for experiments
   - Document the purpose of each experiment
   - Clean up experiments that are no longer needed

5. **Collaboration with Azure File Storage**:
   - Push both Git commits and DVC data when sharing work
   - Pull both Git commits and DVC data when receiving work
   - Use a shared Azure File Storage for team collaboration
   - Consider setting up Azure Active Directory (AAD) for more secure access control
   - Use environment variables for Azure credentials to avoid exposing them in your code or config files

## Example: Complete Workflow

Here's a complete example workflow using DVC with the SEIR model:

```bash
# Initialize Git and DVC
git init
dvc init
git add .dvc/config
git commit -m "Initialize DVC"

# Configure Azure File Storage as remote
pip install dvc[azure]
dvc remote add -d azure azure://myshare/path
dvc remote modify azure connection_string "DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"
git add .dvc/config
git commit -m "Configure Azure File Storage for DVC remote storage"

# Track data and models
dvc add data/covid19vax_trends_us.parquet
dvc add models/seir_model.pkl
dvc add models/vax_trends.pkl
git add *.dvc
git commit -m "Track data and models with DVC"

# Push data to Azure File Storage
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=<account_name>;AccountKey=<account_key>;EndpointSuffix=core.windows.net"
dvc push

# Create pipeline
# (Create dvc.yaml, params.yaml, and script files)
git add dvc.yaml params.yaml scripts/
git commit -m "Create DVC pipeline"

# Run pipeline
dvc repro
git add metrics.json
git commit -m "Run pipeline and save metrics"

# Run experiments
dvc exp run --name "higher_beta" --set-param model.beta=0.3
dvc exp run --name "lower_gamma" --set-param model.gamma=0.03

# Compare experiments
dvc exp diff

# Apply best experiment
dvc exp apply higher_beta
git add params.yaml metrics.json
git commit -m "Apply higher_beta experiment"

# Push changes
git push
dvc push
```

This workflow demonstrates how to use DVC to version data, create reproducible pipelines, track experiments, and compare results with the SEIR model.
