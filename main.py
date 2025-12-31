"""
MIMIC-IV Mortality Prediction Pipeline
Author: Dong-Quan Ngo Nguyen
Description: End-to-end pipeline for MIMIC-IV data processing (Hosp & ICU), 
             automated feature engineering using BERT, and mortality prediction model training.
"""

import pandas as pd
import numpy as np
import os
import gc
import warnings
from tqdm import tqdm
from icdmappings import Mapper
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
# Centralized configuration dictionary for easy adjustments
CFG = {
    'RAW_DIR': './data/raw',
    'OUT_DIR': './data/processed',
    'FILES': {
        'ADMISSIONS': 'hosp/admissions.csv',
        'PATIENTS': 'hosp/patients.csv',
        'LABEVENTS': 'hosp/labevents.csv',
        'DIAGNOSES': 'hosp/diagnoses_icd.csv',
        'CHARTEVENTS': 'icu/chartevents.csv',
        'D_ITEMS': 'icu/d_items.csv',
        'ICUSTAYS': 'icu/icustays.csv'
    },
    'CHUNK_SIZE': 5_000_000,
    'ICU_FREQ_THRESHOLD': 1.0,
    'BERT_MODEL': 'all-MiniLM-L6-v2',
    'SEED': 42
}

def get_file_path(key):
    return os.path.join(CFG['RAW_DIR'], CFG['FILES'][key])

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ======================================================
# 1. HOSPITAL DATA PIPELINE
# ======================================================

def process_hospital_data():
    """
    Handles the ETL process for hospital-level data: Admissions, Patients, Labs, and Diagnoses.
    """
    print("[INFO] Starting Hospital Data Processing Pipeline...")
    ensure_directory_exists(CFG['OUT_DIR'])
    
    # --- 1.1 Merge Admissions & Patients ---
    output_merge_path = os.path.join(CFG['OUT_DIR'], 'merged_admissions_patients.csv')
    
    if os.path.exists(output_merge_path):
        print(f"[INFO] File {output_merge_path} already exists. Skipping merge step.")
    else:
        print("[INFO] Merging Admissions and Patients data...")
        df_patients = pd.read_csv(get_file_path('PATIENTS')).drop_duplicates('subject_id')
        
        chunk_iterator = pd.read_csv(get_file_path('ADMISSIONS'), chunksize=50000)
        is_first_chunk = True
        
        for chunk in tqdm(chunk_iterator, desc="Merging Admissions"):
            chunk = chunk.dropna(subset=['hadm_id', 'subject_id'])
            merged_chunk = chunk.merge(df_patients, on='subject_id', how='left')
            
            write_mode = 'w' if is_first_chunk else 'a'
            merged_chunk.to_csv(output_merge_path, mode=write_mode, header=is_first_chunk, index=False)
            is_first_chunk = False
        
        print("[INFO] Merge completed successfully.")

    # --- 1.2 Process Lab Events ---
    output_labs_path = os.path.join(CFG['OUT_DIR'], 'processed_labs.parquet')
    
    if os.path.exists(output_labs_path):
        print(f"[INFO] File {output_labs_path} already exists. Skipping lab processing.")
    else:
        print("[INFO] Processing Lab Events...")
        # Dictionary of top 20 clinical lab item IDs
        TOP_LAB_ITEMS = {
            50971: 'Potassium', 50983: 'Sodium', 50902: 'Chloride', 50912: 'Creatinine',
            50931: 'Glucose', 51006: 'BUN', 51221: 'Hematocrit', 51222: 'Hemoglobin',
            51301: 'WBC', 51265: 'Platelet', 50868: 'Anion Gap', 50882: 'Bicarbonate',
            50893: 'Calcium', 50960: 'Magnesium', 50970: 'Phosphate', 51279: 'RBC',
            51250: 'MCV', 51248: 'MCH', 51249: 'MCHC', 51277: 'RDW'
        }
        
        temp_labs_csv = 'temp_labs_extract.csv'
        is_first_chunk = True
        
        reader = pd.read_csv(get_file_path('LABEVENTS'), chunksize=CFG['CHUNK_SIZE'], 
                             usecols=['hadm_id', 'itemid', 'valuenum'])
        
        for chunk in tqdm(reader, desc="Filtering Lab Events"):
            chunk = chunk.dropna()
            chunk = chunk[chunk['itemid'].isin(TOP_LAB_ITEMS.keys())]
            
            if not chunk.empty:
                chunk['test_name'] = chunk['itemid'].map(TOP_LAB_ITEMS)
                chunk.to_csv(temp_labs_csv, mode='w' if is_first_chunk else 'a', header=is_first_chunk, index=False)
                is_first_chunk = False
        
        # Aggregation Step
        print("[INFO] Aggregating Lab Data (Mean/Std)...")
        df_labs_raw = pd.read_csv(temp_labs_csv)
        df_agg = df_labs_raw.groupby(['hadm_id', 'test_name'])['valuenum'].agg(['mean', 'std']).unstack()
        
        # Flatten MultiIndex columns
        df_agg.columns = [f'{col[1]}_{col[0]}' for col in df_agg.columns]
        df_agg = df_agg.reset_index().fillna(0)
        
        df_agg.to_parquet(output_labs_path, index=False)
        
        # Clean up temporary file
        if os.path.exists(temp_labs_csv): os.remove(temp_labs_csv)
        
        print("[INFO] Lab processing completed.")
        del df_labs_raw, df_agg; gc.collect()

    # --- 1.3 Process Diagnoses ---
    output_diag_path = os.path.join(CFG['OUT_DIR'], 'processed_diagnoses.parquet')
    
    if os.path.exists(output_diag_path):
        print(f"[INFO] File {output_diag_path} already exists. Skipping diagnoses mapping.")
    else:
        print("[INFO] Mapping ICD Codes to CCS Categories...")
        df_diag = pd.read_csv(get_file_path('DIAGNOSES'), dtype={'icd_code': str, 'icd_version': str})
        mapper = Mapper()
        
        tqdm.pandas(desc="Mapping ICD codes")
        # Standardize ICD10 to ICD9 then to CCS for consistency
        df_diag['ccs'] = df_diag.progress_apply(
            lambda x: mapper.map(x['icd_code'], source='icd9', target='ccs') 
            if x['icd_version'] == '9' 
            else mapper.map(mapper.map(x['icd_code'], source='icd10', target='icd9'), source='icd9', target='ccs'), 
            axis=1
        )
        
        df_diag = df_diag.dropna(subset=['ccs'])
        df_diag['ccs'] = 'CCS_' + df_diag['ccs'].astype(str)
        
        pivot_diag = pd.crosstab(df_diag['hadm_id'], df_diag['ccs']).clip(upper=1).astype('int8').reset_index()
        pivot_diag.to_parquet(output_diag_path, index=False)
        print("[INFO] Diagnoses mapping completed.")

# ======================================================
# 2. ICU DATA PIPELINE
# ======================================================

def process_icu_data():
    """
    Handles the ETL process for ICU data: Frequency analysis, BERT clustering, and Aggregation.
    """
    print(f"\n[INFO] Starting ICU Data Processing Pipeline...")
    
    # --- 2.1 Feature Selection ---
    mapping_file_path = os.path.join(CFG['OUT_DIR'], 'icu_features_map.csv')
    
    if os.path.exists(mapping_file_path):
        print(f"[INFO] Feature map found at {mapping_file_path}. Skipping generation.")
        selected_features = pd.read_csv(mapping_file_path)
    else:
        print("[INFO] Generating feature map using frequency analysis and semantic clustering...")
        
        # Calculate Item Frequency
        total_stays_df = pd.read_csv(get_file_path('ICUSTAYS'), usecols=['stay_id'])
        total_stays_count = total_stays_df['stay_id'].nunique()
        item_counts = {}
        
        reader = pd.read_csv(get_file_path('CHARTEVENTS'), chunksize=CFG['CHUNK_SIZE'], usecols=['itemid', 'stay_id'])
        for chunk in tqdm(reader, desc="Counting Item Frequencies"):
            chunk_counts = chunk.groupby('itemid')['stay_id'].nunique()
            for k, v in chunk_counts.items():
                item_counts[k] = item_counts.get(k, 0) + v
                
        # Filter items based on frequency threshold
        valid_items = [k for k, v in item_counts.items() if (v / total_stays_count) * 100 >= CFG['ICU_FREQ_THRESHOLD']]
        
        d_items = pd.read_csv(get_file_path('D_ITEMS'))
        d_items = d_items[d_items['itemid'].isin(valid_items)].copy()
        # Combine label and category for better context
        d_items['text'] = d_items['label'].fillna('') + " " + d_items['category'].fillna('')
        
        # Semantic Embedding and Clustering using BERT
        print("[INFO] Generating embeddings and clustering features...")
        model = SentenceTransformer(CFG['BERT_MODEL'])
        embeddings = model.encode(d_items['text'].tolist(), show_progress_bar=True)
        
        cluster_algo = AgglomerativeClustering(n_clusters=None, distance_threshold=0.25, metric='cosine', linkage='average')
        d_items['cluster'] = cluster_algo.fit_predict(embeddings)
        
        d_items['feature_name'] = d_items.groupby('cluster')['label'].transform(lambda x: sorted(x, key=len)[0])
        d_items.to_csv(mapping_file_path, index=False)
        selected_features = d_items
        print(f"[INFO] Feature selection complete. Mapped {len(valid_items)} ItemIDs to {d_items['cluster'].nunique()} unique semantic features.")

    # --- 2.2 Data Extraction & Aggregation ---
    output_icu_path = os.path.join(CFG['OUT_DIR'], 'processed_icu.parquet')
    
    if not os.path.exists(output_icu_path):
        print("[INFO] Extracting and Aggregating ICU Data...")
        
        item_map = dict(zip(selected_features['itemid'], selected_features['feature_name']))
        valid_ids = set(selected_features['itemid'])
        
        temp_icu_csv = 'temp_icu_data.csv'
        is_first_chunk = True
        
        reader = pd.read_csv(get_file_path('CHARTEVENTS'), chunksize=CFG['CHUNK_SIZE'], 
                             usecols=['stay_id', 'itemid', 'valuenum'])
        
        for chunk in tqdm(reader, desc="Extracting ICU Data"):
            chunk = chunk.dropna(subset=['valuenum'])
            chunk = chunk[chunk['itemid'].isin(valid_ids)]
            if not chunk.empty:
                chunk['feature'] = chunk['itemid'].map(item_map)
                chunk.to_csv(temp_icu_csv, mode='w' if is_first_chunk else 'a', header=is_first_chunk, index=False)
                is_first_chunk = False
        
        print("[INFO] Pivoting data (Aggregating by Mean/Min/Max)...")
        
        df_icu_raw = pd.read_csv(temp_icu_csv)
        df_agg = df_icu_raw.groupby(['stay_id', 'feature'])['valuenum'].agg(['mean', 'min', 'max']).unstack()
        df_agg.columns = [f'{y}_{x}' for x, y in df_agg.columns]
        df_agg = df_agg.reset_index()
        
        # Map stay_id to hadm_id for final merging
        stays = pd.read_csv(get_file_path('ICUSTAYS'), usecols=['stay_id', 'hadm_id'])
        df_final = df_agg.merge(stays, on='stay_id', how='left').drop(columns=['stay_id'])
        # Handle cases where one admission has multiple ICU stays (average them)
        df_final = df_final.groupby('hadm_id').mean().reset_index() 
        
        df_final.to_parquet(output_icu_path, index=False)
        
        if os.path.exists(temp_icu_csv): os.remove(temp_icu_csv)
        print("[INFO] ICU Data processing completed.")

# ======================================================
# 3. MODEL TRAINING
# ======================================================

def train_models():
    """
    Loads processed data, merges them, and trains Benchmark models (LightGBM, XGBoost).
    """
    print(f"\n[INFO] Starting Model Training Pipeline...")
    
    # --- 3.1 Merge All Data Sources ---
    print("[INFO] Merging Hospital, Diagnoses, and ICU data...")
    df_hosp = pd.read_csv(os.path.join(CFG['OUT_DIR'], 'merged_admissions_patients.csv'))
    
    # Verify Target Variable
    if 'hospital_expire_flag' not in df_hosp.columns:
        raise ValueError("[ERROR] Target variable 'hospital_expire_flag' not found in admissions data.")
        
    df_labs = pd.read_parquet(os.path.join(CFG['OUT_DIR'], 'processed_labs.parquet'))
    df_diag = pd.read_parquet(os.path.join(CFG['OUT_DIR'], 'processed_diagnoses.parquet'))
    df_icu = pd.read_parquet(os.path.join(CFG['OUT_DIR'], 'processed_icu.parquet'))
    
    # Left Merge to ensure all admissions are retained, even if some have missing labs/icu data
    df = df_hosp[['hadm_id', 'hospital_expire_flag']].merge(df_labs, on='hadm_id', how='left')
    df = df.merge(df_diag, on='hadm_id', how='left')
    df = df.merge(df_icu, on='hadm_id', how='left')
    
    # Handle Missing Values (-1 is a robust placeholder for Tree-based models)
    df = df.fillna(-1)
    
    # Define Features (X) and Target (y)
    X = df.drop(columns=['hadm_id', 'hospital_expire_flag'])
    y = df['hospital_expire_flag']
    
    print(f"[INFO] Final Dataset Shape: {X.shape}. Mortality Rate: {y.mean():.2%}")
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=CFG['SEED'], stratify=y)
    
    # --- 3.2 LightGBM Training ---
    print("\n[INFO] Training LightGBM Model...")
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=CFG['SEED'])
    lgb_clf.fit(X_train, y_train)
    auc_lgb = roc_auc_score(y_test, lgb_clf.predict_proba(X_test)[:, 1])
    print(f"[RESULT] LightGBM AUC: {auc_lgb:.4f}")
    
    # --- 3.3 XGBoost Training ---
    print("\n[INFO] Training XGBoost Model...")
    xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, n_jobs=-1, random_state=CFG['SEED'], eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    auc_xgb = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])
    print(f"[RESULT] XGBoost AUC: {auc_xgb:.4f}")

    # --- 3.4 Feature Importance Analysis ---
    feature_importance = pd.DataFrame({'feature': X.columns, 'score': lgb_clf.feature_importances_})
    print("\n[INFO] Top 5 Most Important Features:")
    print(feature_importance.sort_values('score', ascending=False).head(5))

# ======================================================
# MAIN EXECUTION ENTRY POINT
# ======================================================
if __name__ == "__main__":
    if not os.path.exists(CFG['RAW_DIR']):
        print(f"[ERROR] Data directory {CFG['RAW_DIR']} not found. Please verify the path.")
    else:
        try:
            process_hospital_data()
            process_icu_data()
            train_models()
            print("\n[SUCCESS] Pipeline executed successfully.")
        except KeyboardInterrupt:
            print("\n[STOP] Process interrupted by user.")
        except Exception as e:
            print(f"\n[ERROR] Runtime exception: {e}")