import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve

# Load predictions
pred_df = pd.read_csv('prediction.csv')

# Load the dedicated evaluation ground truth file
true_df = pd.read_csv('data/fakeavceleb_eval.csv') 

# Extract basenames
pred_df['basename'] = pred_df['video_name'].apply(os.path.basename)
true_df['basename'] = true_df['video_name'].apply(os.path.basename)

# FIX: Drop duplicate predictions from multiple eval.py runs (keep the newest)
pred_df = pred_df.drop_duplicates(subset=['basename'], keep='last')

# Merge them together
merged = pd.merge(pred_df, true_df, on='basename')
print(f"Total Videos Merged (Deduplicated): {len(merged)}")

# Dynamically find the ground truth column 
if 'target' in merged.columns:
    gt_col = 'target'
elif 'label' in merged.columns:
    gt_col = 'label'
else:
    raise ValueError("Could not find 'target' or 'label' column in ground truth CSV!")

# Map strings to integers safely
def map_labels(val):
    if isinstance(val, str):
        val = val.strip().upper()
        if val in ['1', 'FAKE', 'SPOOF']: return 1
        if val in ['0', 'REAL', 'BONAFIDE']: return 0
    return int(val)

merged['y_true_clean'] = merged[gt_col].apply(map_labels)

print("\n--- Balanced Label Counts ---")
print(merged['y_true_clean'].value_counts(dropna=False))

unique_classes = merged['y_true_clean'].nunique()

if unique_classes > 1:
    y_true = merged['y_true_clean'].values
    y_scores = merged['y_pred'].values
    
    auc = roc_auc_score(y_true, y_scores)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    
    print("\n" + "="*40)
    print(f" Final AUC Score: {auc:.4f}")
    print(f" Final EER Score: {eer:.4f}")
    print("="*40)
else:
    print("\nCRITICAL: Still only 1 class found. Open the CSV and verify its contents manually.")