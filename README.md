# Data Science Ops Tools Comparison

This repository contains demonstrations of how to use different data science
tools (Metaflow, MLflow, and DVC) to experiment with parameter settings for a
SEIR (Susceptible-Exposed-Infectious-Recovered) compartmental model.

## Overview

The SEIR model is a compartmental model used in epidemiology to model the
spread of infectious diseases. This demo compares and contrasts how a modeler
might use different tools (Metaflow, MLflow, and DVC) to experiment with model
settings, track experiments, manage data, and deploy models.

## Files

- `seir_model.py`: The core SEIR model implementation
- `seir_demo.ipynb`: Basic demonstration of the SEIR model
- `seir_mlflow_demo.ipynb`: Demonstration of using MLflow for experiment
  tracking
- `seir_metaflow_demo.py`: Metaflow workflow for SEIR model experimentation
- `seir_metaflow_demo.ipynb`: Jupyter notebook demonstrating how to use the
  Metaflow workflow
- `seir_dvc_demo.md`: Hypothetical workflow using DVC for data versioning and
  pipeline tracking via config files and shell commands

## Tools Comparison

This repository demonstrates three popular tools for data science workflows and
model experimentation:

### Metaflow

Metaflow is a workflow orchestration framework developed by Netflix that
focuses on (most documentation and community examples in Python, but there is
and R version too):

- Defining data science workflows as code (DAGs)
- Running steps in parallel
- **Managing dependencies between steps**
- **Scaling from local development to production**
- Versioning data and results

Similar and competitive technologies to metaflow include:

- flyte
- zenml

### MLflow

MLflow is a platform for the machine learning lifecycle developed by Databricks
that focuses on:

- Tracking experiments (parameters, metrics, artifacts)
- Packaging code into reproducible runs
- Managing and deploying models
- Providing a central model registry
- **Offering a UI for visualizing and comparing experiments**
- *Bindings for popular ML packages*

Similar and competitive technologies to mlflow include:

- kubeflow

### DVC (Data Version Control)

DVC is a version control system for machine learning projects that focuses on:

- Versioning data and models (*built on top of Git*)
- Creating reproducible pipelines
- Tracking experiments
- Managing remote storage for large files
- Providing metrics comparison and visualization
- **Language agnosticism** (DVC is a python tool, but you don't need to be a
  python dev to use it)

## When to Use Each Tool

- **Metaflow**: Best for complex workflows with multiple steps that need to be
executed in a specific order, potentially in parallel, and when you need to
scale from local development to production.

- **MLflow**: Best for tracking experiments, comparing model performance, and
managing model deployment when you have multiple data scientists working on the
same project.

- **DVC**: Best for version controlling large data files and models, creating
reproducible pipelines, and when you want to leverage Git for versioning while
handling large files efficiently.

## Complementary Use

These tools can be used together in a complementary way:

- **Metaflow + MLflow**: Use Metaflow for workflow orchestration and MLflow for
  experiment tracking and model registry.

- **DVC + MLflow**: Use DVC for data and pipeline versioning and MLflow for
  experiment tracking and model registry.

- **All three**: Use Metaflow for workflow orchestration, DVC for data
  versioning, and MLflow for experiment tracking and model registry.

The choice of tools depends on your specific needs, team size, and existing
infrastructure. This repository aims to demonstrate the strengths and use cases
of each tool to help you make an informed decision.
