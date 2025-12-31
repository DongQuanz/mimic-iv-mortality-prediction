# MIMIC-IV Mortality Prediction Pipeline

## Project Overview
This project implements an end-to-end Extract, Transform, Load (ETL) and Machine Learning pipeline to predict in-hospital mortality using the MIMIC-IV dataset. The pipeline integrates multimodal data, including:
- **Demographics:** Patient admissions and background.
- **Clinical Labs:** Top 20 most frequent laboratory events.
- **Diagnoses:** ICD-9 and ICD-10 codes mapped to Clinical Classifications Software (CCS) categories.
- **ICU Chart Events:** High-frequency vital signs processed via frequency analysis and semantic clustering using BERT embeddings.

The system utilizes chunking mechanisms to process large-scale CSV files (approx. 300GB+) within limited memory constraints and trains benchmark models using LightGBM and XGBoost.

## Architecture
The pipeline consists of three main stages:
1. **Hospital Data Processing:** Merges patient demographics, aggregates lab results, and encodes diagnostic codes.
2. **ICU Data Processing:** Implements statistical frequency filtering and semantic clustering (SentenceBERT) to reduce feature dimensionality from raw chart events.
3. **Modeling:** Merges all processed data streams and trains gradient boosting models (LightGBM, XGBoost) to classify mortality risk.

## Prerequisites
- Python 3.8+
- Access to MIMIC-IV dataset (Credentialed access required via PhysioNet).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DongQuanZ/mimic-iv-mortality-prediction.git
   cd mimic-iv-mortality-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   Place the raw MIMIC-IV CSV files in the `./data/raw` directory following this structure:
   ```text
   data/raw/
   ├── hosp/
   │   ├── admissions.csv
   │   ├── patients.csv
   │   ├── labevents.csv
   │   └── diagnoses_icd.csv
   └── icu/
       ├── chartevents.csv
       ├── d_items.csv
       └── icustays.csv
   ```

## Usage

Execute the main pipeline script:

```bash
python main.py
```

The script will automatically:
1. Check for existing processed files to avoid redundant computation.
2. Process data in chunks to manage memory usage.
3. Train models and output AUC scores and feature importance.

## Results

Current benchmark performance on the test set:

- **LightGBM AUC:** [Insert your score, e.g., 0.9889]
- **XGBoost AUC:** [Insert your score, e.g., 0.9888]

## Disclaimer

This project utilizes the MIMIC-IV dataset. The data is not included in this repository due to the PhysioNet Data Use Agreement. Users must acquire their own credentials to access the raw data.