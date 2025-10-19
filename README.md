# MLOps: DVC + Google Cloud Storage Pipeline

This project demonstrates an end-to-end Machine Learning workflow using **DVC (Data Version Control)** with **Google Cloud Storage (GCS)** as the remote storage backend.
It automates data management, model versioning, and CI/CD testing with GitHub Actions and CML.

# Project Overview

The pipeline showcases how to manage ML experiments efficiently while maintaining reproducibility, traceability, and scalability across environments.

# Key Components

=> **Data Versioning:** DVC tracks data and model changes across commits
=> **Model Training:** train.py trains a Decision Tree Classifier on the Iris dataset
=> **Metrics Tracking:** Saves accuracy, precision, recall, and F1-score to artifacts/metrics.csv
=> **Remote Storage:** Model artifacts and datasets are stored on Google Cloud Storage (GCS)
=> **Automation:** Two shell scripts simplify setup, data sync, and pipeline execution

# Setup Instructions

**1. Clone the Repository**
git clone https://github.com/SwarnaliRoy1/MLOps_week4.git
cd MLOps_week4
**2. Create and Activate Virtual Environment**
python3 -m venv .env
source .env/bin/activate
**3. Install Dependencies**
pip install -r req.txt
**4. Configure DVC Remote (Google Cloud Storage)**
dvc remote add -d gcs_remote gs://<your-bucket-name>
dvc remote modify gcs_remote credentialpath ~/.config/gcloud/application_default_credentials.json
**5. Push or Pull Data and Artifacts**
dvc push -r gcs_remote       # Upload data/artifacts to GCS
dvc pull -r gcs_remote       # Fetch data/artifacts from GCS

# Continuous Integration (CI) & Sanity Tests

This repository implements automated Continuous Integration (CI) pipelines for both the main and dev branches using GitHub Actions.

Each branch has its own CI workflow that runs automatically:

**On every push**

**On every pull request (PR) merge**

**Sanity Tests**

A sanity test is included for both main and dev branches to ensure core functionality and stability before running the full pipeline.
These tests:

Validate existence and format of critical files (data.csv, artifacts/metrics.csv)

Check if model training and evaluation complete successfully

Confirm that key metrics (accuracy, precision, recall, F1) are generated and stored correctly

**Automated Reporting with CML**

Each CI workflow also runs a sanity test report using CML (Continuous Machine Learning).

The results of the sanity test are automatically posted as a comment in the corresponding GitHub pull request or commit.

This provides immediate feedback on whether the pipeline passed or failed, directly within the GitHub interface.

**Summary**

* Separate CI workflows for main and dev branches
* Triggered on every push or PR merge
* Sanity tests ensure data, model, and metrics consistency
* CML comments provide instant visual feedback on test outcomes
* GCS Remote provides Cloud-based DVC remote for secure artifact storage
