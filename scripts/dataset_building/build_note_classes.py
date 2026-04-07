import pandas as pd
import os
import re


def process_tsv(input_path):
    """
    Reads a TSV file, keeps only specific columns, prints its head,
    and saves it to CSV format.
    
    Parameters:
        input_path (str): Path to the .tsv file
    """

    try:
        # Step 1: Keep specific columns and read
        header = pd.read_csv(input_path, sep='\t', nrows=0)
        all_columns = header.columns.tolist()

        keep_indices = [0, 1, 2, 3, 4, 8, 9, 10, 11,12, 13,14,15,16, 17,18,19, 20, 21, 22]
        columns_to_use = [col for i, col in enumerate(all_columns) if i in keep_indices]

        df = pd.read_csv(input_path, sep='\t', dtype={"tweetId": str},usecols=columns_to_use, low_memory=False)

        # Step 2: Print preview
        print("Selected Columns:", df.columns.tolist())
        print(df.head())

        # Step 5: Save to file
        base, _ = os.path.splitext(input_path)            
        output_path = f"{base}_formatted.csv"
        
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

        return output_path

    except Exception as e:
        print(f"Error: {e}")



def searchImageKeywords(file):

    # Media keyword pattern
    image_keywords = re.compile(
        r"\b(?:photo|photos|image|images|photograph|photographs|screenshot|screenshot|pic|pics|photoshop|photoshopped|picture|pictures|snapshot|snapshots|visual|visuals|jpg|graphic|graphics|thumbnail|thumbnails|logo|logos|png|jpeg)\b",
        flags=re.IGNORECASE
    ) 


    df = pd.read_csv(file, dtype=str)
    df.columns = df.columns.str.strip()
    df["summary"] = df["summary"].fillna("")

    # Filter entries with media mentions in summary
    df_filtered = df[df["summary"].str.contains(image_keywords)].copy()

    df_filtered.loc[:, "tweetId"] = df_filtered["tweetId"].astype(str).str.replace(r"\.0$", "", regex=True)
    df_filtered.loc[:, "tweetUrl"] = "https://x.com/i/web/status/" + df_filtered["tweetId"]
    df_filtered.loc[:, "createdAtMillis"] = pd.to_datetime(df_filtered["createdAtMillis"], unit='ms', utc=True) 


    # Save filtered results
    image_path = "notes_with_images.csv"
    df_filtered.to_csv(image_path, index=False)

    return image_path 



def build_classes(file, output_dir="."):
    df = pd.read_csv(file)

    # Ensure tweetId is string and tweetUrl exists
    df["tweetId"] = df["tweetId"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "tweetUrl" not in df.columns or df["tweetUrl"].isna().all():
        df["tweetUrl"] = "https://x.com/i/web/status/" + df["tweetId"]

    # Factually Correct
    fc = (
        (df["notMisleadingFactuallyCorrect"].fillna(0).astype(int) == 1) &
        (df["notMisleadingClearlySatire"].fillna(0).astype(int) == 0) &
        (df["notMisleadingOutdatedButNotWhenWritten"].fillna(0).astype(int) == 0) &
        (df["trustworthySources"].fillna(0).astype(int) == 1)
    )

    # Miscaptioned
    miscaptioned_flags = (
        (df["misleadingFactualError"].fillna(0).astype(int) == 1) |
        (df["misleadingMissingImportantContext"].fillna(0).astype(int) == 1)
    )
    mc = (
        (df["misleadingManipulatedMedia"].fillna(0).astype(int) == 0) &
        miscaptioned_flags &
        (df["trustworthySources"].fillna(0).astype(int) == 1)
    )

    def save_class(mask, name):
        out = df[mask].drop_duplicates(subset="tweetId").reset_index(drop=True)
        out = out.rename_axis("id").reset_index()
        out_path = os.path.join(output_dir, f"{name}_class.csv")
        out.to_csv(out_path, index=False)
        
        print(f"Saved {name} -> {out_path}")
        return out_path

    factual_path = save_class(fc, "factually_correct")
    miscaptioned_path = save_class(mc, "miscaptioned")
    return factual_path, miscaptioned_path



"""
    Process the downloaded TSV file 
"""
# note_file = 'notes.tsv'
# notes_csv_path = process_tsv(note_file)

"""
    Keep Community Notes containing Image Keywords 
"""
# notes_image_path = searchImageKeywords(notes_csv_path)

"""
    Build Factually Correct and Miscaptioned Classes 
"""
# factual_path, miscaptioned_path = build_classes(notes_image_path)
