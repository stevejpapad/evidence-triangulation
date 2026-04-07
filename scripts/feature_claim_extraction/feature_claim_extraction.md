## Feature & Claim Extraction

**Overview**  
This module runs multimodal **claim extraction** and **feature extraction** on social media posts (text + images) using Google AI Studio (Gemma) and saves the results to disk.

You typically run both scripts on the **same dataset** produced by the dataset building step:

- First extract **claims**.
- Then extract **features**, using the previously extracted claims.

### Setup

- Environment:
  - Setup a `.env` file with your Google AI Studio key
    - e.g. `GOOGLE_AI_STUDIO_API_KEY=<your_api_key>`
  - Local images under the folder pointed to by `LOCAL_IMAGES_PATH` in the scripts (e.g. `images/`).

---

### Claim extraction (`claim_extraction.py`)

1. **Load and filter dataset**
   - Reads `dataset_path` (the combined dataset CSV from the dataset building stage).
   - Keeps only rows with classes in `["factually_correct", "miscaptioned"]`.

2. **Subset selection**
   - Uses `START` and `END` to select a slice of the dataset for processing.

3. **Prepare inputs per row**
   - Extracts:
     - `id` / `uniqueID`
     - `post_text` / `tweetText`
     - `image_paths` / `tweetMediaFiles`
     - `class` / `label`

4. **Convert images to base64**
   - For each image path:
     - Join with `LOCAL_IMAGES_PATH`
     - Load and resize the image (keeping aspect ratio)
     - Encode it as base64 for API input
   - If no valid images are available for a row, that row is skipped

5. **Build prompts**
   - Calls `prompt_1(user_post, image_urls)` from `vlm_prompts.py` to construct the user prompt for claim reformulation

6. **Model inference**
   - Calls `mm_inference_google(...)` with:
     - `model_name` (default: `"gemma-3-27b-it"`)
     - `user_prompt` from `prompt_1`
     - `image_urls` (base64 images)
     - `max_tokens` and `temperature`
   - If the model returns **no response** (`None`), the script **uses the original post text as the claim** fallback.
   - Otherwise, the script uses the returned text (stripped) as the extracted claim.

7. **Rate limiting**
   - Waits ~2.5 seconds between requests to respect the API rate limit.

8. **Save outputs**
   - Collects per-row results into a DataFrame and saves them
   - Each entry contains:
     - `id`
     - `post_text`
     - `image_paths`
     - `class`
     - `extracted_claim`

9. **Disclaimer about fallback behavior**
   - For robustness, **if the claim extraction model returns `None` or fails**, the pipeline **falls back to using the original post text as the claim**

---

### Feature extraction (`feature_extraction.py`)

1. **Load and filter dataset**
   - Reads `dataset_path` (the same combined dataset CSV)
   - Also loads `claims_path` and builds a map from `id` → `extracted_claim`

2. **Subset selection**

3. **Prepare inputs per row & images**

4. **Convert images to base64**
   - For each image path:
     - Join with `LOCAL_IMAGES_PATH`.
     - Load, resize, and encode as base64.
   - Rows without valid images are skipped.

5. **Use pre-extracted claims**
   - Looks up the claim for each `id` from `claims_path`.
   - This claim is passed as `user_claim` into `prompt_2`.

6. **Build prompts**
   - Calls `prompt_2(user_claim, image_urls)` from `vlm_prompts.py` to construct the feature extraction prompt, which asks the model to return a JSON structure with multimodal features.

7. **Model inference**
   - Calls `mm_inference_google(...)`
   - If the model returns no response, the row is marked as `extraction_failed`.
   - Otherwise, the script tries to parse the response as JSON containing fields like:
     - `image_description`
     - `ocr_text`
     - `named_entities`
     - `five_Ws` (or similar fields, depending on the prompt).

8. **Rate limiting**

9. **Save outputs**
   - Collects per-row results into a DataFrame and saves to JSON (e.g. `dataset_gemma_features_*.json`).
   - Each entry contains:
     - `id`
     - `post_text`
     - `image_paths`
     - `class`
     - `features`

---

### Recommended run order

1. Run `claim_extraction.py` on your dataset to produce `dataset_gemma_claims.json`.

2. Then run `feature_extraction.py` using the **same** `dataset_path` and the generated `claims_path` to extract features conditioned on those claims.

This will give you two aligned JSON files: one with **claims** and one with **features** for the same set of posts.
