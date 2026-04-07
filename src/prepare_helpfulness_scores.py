import numpy as np
import pandas as pd
from tqdm import tqdm 
from pathlib import Path
from dataset import load_dataset
from collections import defaultdict

def calculate_helpfulness_scores(DATA_PATH):

    train_data, valid_data, test_data = load_dataset(DATA_PATH)
    note_ids = (set(train_data["noteId"].astype(str)) | 
                set(valid_data["noteId"].astype(str)) | 
                set(test_data["noteId"].astype(str)))

    # Folder containing all extracted TSV files
    # Ratings data from : https://x.com/i/communitynotes/download-data
    data_folder = Path("note_ratings")   
        
    helpfulness_counts = []

    tsv_files = list(data_folder.rglob("*.tsv"))
    
    for tsv_path in tqdm(tsv_files, desc="Processing TSV files"):
        try:
            
            for chunk in pd.read_csv(tsv_path, sep="\t", dtype=str, chunksize=100000):
                
                # Keep only note IDs in X-POSE 
                matched = chunk[chunk["noteId"].isin(note_ids)]
                
                if not matched.empty:
                    counts = matched.groupby(["noteId", "helpfulnessLevel"]).size().reset_index(name="count")
                    helpfulness_counts.append(counts)
                    
        except Exception as e:
            print(f"\nSkipping {tsv_path.name} due to error: {e}")

    if not helpfulness_counts:
        print("No matches found.")
        return

    final_df = pd.concat(helpfulness_counts)
    final = final_df.groupby(["noteId", "helpfulnessLevel"])["count"].sum().unstack(fill_value=0)
    
    final = final.drop(columns=["unknown"], errors="ignore")
    
    for col in ["HELPFUL", "NOT_HELPFUL"]:
        if col not in final.columns:
            final[col] = 0

    # Compute ratio
    final["helpfulness_ratio"] = final["HELPFUL"] / (final["HELPFUL"] + final["NOT_HELPFUL"])
    
    save_path = Path(DATA_PATH) / "helpfulness_scores.csv"
    final.to_csv(save_path)
    print(f"Results saved to {save_path}")