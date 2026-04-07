import pandas as pd
import time
import ast 
import json

from .utils import mm_inference_google, image_to_base64
from ..vlm_prompts import prompt_2



def extract_features_batch(path, output_path, claims_path, model_name, local_images_path, subset_start=None, subset_end=None):


    df = pd.read_csv(path)

    active_classes = ["factually_correct", "miscaptioned"]

    # Filter active classes
    df = df[df["class"].isin(active_classes)]
    print('For classes:',active_classes,' we have: ',len(df) ,' entries\n')


    # Set inference parameters manually
    temperature = 0.2
    max_tokens = 1024

    # Load pre-extracted claims
    claims_map = {}
    if claims_path:
        with open(claims_path, "r", encoding="utf-8") as f:
            claims_data = json.load(f)
            for entry in claims_data:
                claim_val = entry.get("extracted_claim", "")
                if isinstance(claim_val, dict):
                    claim_text = claim_val.get("claim", "")
                else:
                    claim_text = claim_val or ""
                claims_map[entry["id"]] = claim_text


    # Define a subset to run tests
    if subset_start is not None and subset_end is not None:
        df = df.iloc[subset_start:subset_end]

    results = []

    for idx, row in df.iterrows():
        unique_id = row["id"] if "id" in row else row["uniqueID"]
        post_text = row["post_text"] if "post_text" in row else row["tweetText"]
        media_raw = row["image_paths"] if "image_paths" in row else row["tweetMediaFiles"]
        label = row["label"] if "label" in row else row["class"]

        # Parse local image paths (assumes comma-separated if string)
        if isinstance(media_raw, str) and media_raw.startswith("[") and media_raw.endswith("]"):
            try:
                image_paths = ast.literal_eval(media_raw)
            except Exception:
                image_paths = []
        # Handle comma-separated string 
        elif isinstance(media_raw, str):
            image_paths = [p.strip() for p in media_raw.split(",") if p.strip()]
        # Already a list
        elif isinstance(media_raw, list):
            image_paths = media_raw
        else:
            image_paths = []

        # Get URLs from local images
        image_urls = []
        for path in image_paths:
            try:            
                # Convert images locally to base64 
                url = image_to_base64(local_images_path+path)
                image_urls.append(url)
            except Exception as e:
                print(f"❌ Failed to upload {path}: {e}")
                continue

        if not image_urls:
            continue


        # Get extracted claim for this entry
        claim = claims_map.get(unique_id, "")

        #  For feature extraction
        user_content = prompt_2(
                user_claim=claim,
                image_urls=image_urls
        )
        

        """
            Inference Google GenAI 
        """
        response = mm_inference_google(
            model=model_name,
            user_prompt= user_content["user_prompt"],
            image_urls= user_content["image_urls"],
            max_tokens=max_tokens,
            temperature=temperature
        )

        if response is None:
            print(f"❌ No response returned for {unique_id}, skipping.")
            features = {"extraction_failed": True, "raw_response": None}
        else:
            # Try to parse the model's output (assumes it returns JSON-like string)
            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[len("```json"):].strip()
                if cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:].strip()
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3].strip()

                features = json.loads(cleaned_response)
                print('\n\nFeatures:\n\n', features)
            except Exception as e:
                print(f"❌ Failed to parse features for {unique_id}: {e}")
                features = {"extraction_failed": True, "raw_response": response}

        print(f"\n[{idx+1}/{len(df)}] ID: {unique_id} → Extracted features.")
        



        # Build output row
        post_result = {
            "id": unique_id,
            "post_text": post_text,
            "image_paths": image_paths,
            "class": label,
            "features": features
        }

        results.append(post_result)

        # To bypass 20 requests per minute limit (1 req / 3 sec) 
        print('wait 2s for model')
        time.sleep(2.5)

    # Save results to DataFrame
    result_df = pd.DataFrame(results)

    # For feature/claim extraction
    result_df.to_json(output_path, orient="records", indent=2)
    print(f"✅ Saved extracted features to {output_path}!")

    return result_df




# Example run
# MODEL = 'gemma-3-27b-it' 

# """
#     For testing specific rows
# """
# START=0
# END=10

# LOCAL_IMAGES_PATH = "images/"

# dataset_path = "dataset.csv"
# output_path= f"dataset_gemma_features_{START}_{END}.json"


# claims_path = "dataset_gemma_claims.json"


# df = extract_features_batch(dataset_path, output_path, claims_path, MODEL ,LOCAL_IMAGES_PATH, START, END)



