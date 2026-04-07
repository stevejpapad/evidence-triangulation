## Query Generation

This script generates search queries for fact-checking social media posts using a Google Gemma multimodal model.

- **Script**: `query_generation.py`
- **Model**: `gemma-3-27b-it` (via Google GenAI)

### Inputs

- **CSV dataset file** (same combined dataset used in feature/claim extraction)
- **Local images** referenced in the dataset
- **Features JSON** (`features_path`) from `feature_extraction.py`
- **Claims JSON** (`claims_path`) from `claim_extraction.py`

### What it does

- Loads pre-computed **features** and **claims** by `id`
- Converts local images to base64 and sends them, along with text, features and claim, to the Gemma model
- Uses `prompt3` to ask the model for:
  - 3 **text search queries**
  - 1 **image search query**
- Normalizes the model response with `prettify_response`

### Output

- Writes a JSON file containing an array of objects:
  - `id`
  - `post_text`
  - `image_paths`
  - `extracted_claim`
  - `class`
  - `queries` (dict with `text_queries` and `image_search_query`, after post-processing)
