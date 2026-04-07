import json, time
from pathlib import Path
import pandas as pd
from .utils import scrapingdog_reverse_image_call, upload_local_to_imgbb, is_valid_image

"""Reverse Image Search Functions"""

def run_reverse_image_search_on_local_image(local_rel_path, out_base, entry_id, max_size= 512, validate=True):
    """
    Upload a local image to imgbb, optionally validate the hosted image,
    call Scrapingdog Reverse Image Search, and save results under:
      {out_base}/{entry_id}/lens/{entry_id}_lens.json

    Returns: Path to the saved JSON.
    """

    """
        1) Upload the local image to get a public URL
            Add a small expiration time (in seconds) so we don't clutter imgbb
    """
    # Upload the local image to get a public URL
    # local_rel_path is already the filesystem path we want to upload
    # upload_local_to_imgbb() expects (path, base_path) and internally joins them
    hosted_url = upload_local_to_imgbb(local_rel_path, base_path="", max_size=max_size, expiration=60)
    print(f"Uploaded {local_rel_path}")


    """
    2) Optional quick validation (saves credits)
       It downloads the hosted image and checks dimensions/keywords. That adds a small request but can save you credits by skipping malformed images. You can skip for speed.
    """
    if validate:
        ok, reason = is_valid_image(hosted_url, req_timeout=5)
        if not ok:
            print(f"Skipping Lens call for {local_rel_path} → {reason}")
            record = {
                "image_local_path": local_rel_path,
                "hosted_url": hosted_url,
                "skipped": True,
                "reason": reason,
            }
            out_dir = Path(out_base)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{entry_id}_lens.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            return out_path

    # 3) Call ScrapingDog Lens
    try:
        resp = scrapingdog_reverse_image_call(hosted_url)

        record = {"image_local_path": local_rel_path, "hosted_url": hosted_url, "response": resp}
    except Exception as e:
        record = {"image_local_path": local_rel_path, "hosted_url": hosted_url, "error": str(e)}

    # 4) Save result
    out_base = Path(out_base)
    out_path = out_base / f"{entry_id}_lens.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")

    # Optional small delay to be gentle with rate limits
    time.sleep(2.5)

    return out_path



def run_reverse_image_search_on_dataset(csv_path, out_base, local_images_path, subset_start=None,subset_end=None, max_size=512, validate=True):
    """
    Reads a dataset CSV and runs Reverse image search for each image

    Args:
        csv_path (str): Path to dataset CSV.
        out_base (str): Root output folder.
        local_images_path (str): Path to local images folder.
        max_size (int): Max size for upload resize.
        validate (bool): Whether to validate images before calling Lens.
    """

    df = pd.read_csv(csv_path)

    # Define subset for testing
    if subset_start is not None and subset_end is not None:
        df = df.iloc[subset_start:subset_end] 


    for _, row in df.iterrows():

        entry_id = str(row["uniqueID"])
        media_val = row.get("tweetMediaFiles", "")

        if pd.isna(media_val) or not str(media_val).strip():
            print(f"No media for entry {entry_id}, skipping.")
            continue
        
        # Use only the FIRST path in the list
        first_path = str(media_val).split(",")[0].strip()
        print(f"\nEntry {entry_id}: using first media file → {first_path}")
        
        try:
            run_reverse_image_search_on_local_image(
                local_rel_path=local_images_path + first_path,
                out_base=out_base,
                entry_id=entry_id,
                max_size=max_size,
                validate=validate,
            )
        except Exception as e:
            print(f"Failed on {entry_id} / {first_path}: {e}")
            continue


########################################################
########### ScrapingDog Reverse Image Search ##########
######################################################

# Example run


# # Subset
# START=0
# END=100

# BASE_IMAGE_FOLDER = "images/"

# csv_path = "dataset.csv"
# out_base = "reverse_image_search_results/"

# run_reverse_image_search_on_dataset(csv_path, out_base, BASE_IMAGE_FOLDER, START, END, max_size=512, validate=True)





