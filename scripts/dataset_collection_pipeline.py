import asyncio, os, shutil
import pandas as pd

from scripts.dataset_building.build_note_classes import process_tsv, searchImageKeywords, build_classes
from scripts.dataset_building.tweet_extraction import get_tweets_by_ids
from scripts.dataset_building.combine_notes_tweets import process_and_build_dataset, fix_image_extensions
from scripts.feature_claim_extraction.claim_extraction import extract_claims_batch
from scripts.feature_claim_extraction.feature_extraction import extract_features_batch
from scripts.query_generation.query_generation import extract_queries
from scripts.evidence_extraction.text_image_search_scrapingdog import run_google_search_scrapingdog
from scripts.evidence_extraction.reverse_image_search_scrapingdog import run_reverse_image_search_on_dataset
from scripts.evidence_extraction.scrape_text_image_reverse_results import scrape_all_entries, process_lens_folder


"""
This script runs the entire dataset evidence pipeline:

1. Get Community Notes TSV checkpoint from X public files
2. Create the two classes, factual and miscaptioned
3. Extract tweets and images for each Community Note
4. Combine the notes, tweets and images to form the final X-POSE dataset

5. Extract claims for each entry and build the claim map
6. Extract features for each entry and build the feature map
7. Generate the text and image queries for each entry 

8. Run text and image search for each query 
9. Run reverse image search for the first image of each entry
10. Scrape the text and image search results for each query
11. Scrape the reverse image search results for each entry

"""

"""
    Helper Functions
"""
def flatten_image_folders(image_dir):
    """
    Move images from per‑class folders into the common LOCAL_IMAGES_PATH.
    Assumes images were saved under images/<base> for each class.
    """
    bases = [factual_path.replace(".csv", ""), miscaptioned_path.replace(".csv", "")]
    root = image_dir.rstrip("/")

    os.makedirs(root, exist_ok=True)

    for base in bases:
        src_dir = os.path.join(root, base)  # e.g. images/factually_correct_class
        if not os.path.isdir(src_dir):
            continue

        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src):
                continue

            dst = os.path.join(root, name)  # e.g. images/photo_...jpg
            if os.path.exists(dst):
                # skip or handle collisions as needed
                continue

            shutil.move(src, dst)

def normalize_dataset_image_paths(dataset_csv_path, images_root="images/"):
    """
    After flattening images into a single folder, update `tweetMediaFiles` in the dataset
    to contain just the filenames, so downstream code can open via images_root + filename.
    """
    

    if not os.path.isfile(dataset_csv_path):
        return

    df = pd.read_csv(dataset_csv_path)
    if "tweetMediaFiles" not in df.columns:
        return

    def norm_cell(val):
        if pd.isna(val) or not str(val).strip():
            return val
        parts = [p.strip() for p in str(val).split(",") if p.strip()]
        basenames = [os.path.basename(p) for p in parts]
        return ", ".join(basenames) if basenames else val

    df["tweetMediaFiles"] = df["tweetMediaFiles"].apply(norm_cell)
    df.to_csv(dataset_csv_path, index=False)



"""
    1) Download the Community Notes TSV checkpoint from X public files
"""
NOTE_CHECKPOINT_FILE = 'notes.tsv'
notes_csv_path = process_tsv(NOTE_CHECKPOINT_FILE)


"""
    2) - Keep Community Notes containing Image Keywords 
       - Build Factually Correct and Miscaptioned Classes
"""
notes_image_path = searchImageKeywords(notes_csv_path)
factual_path, miscaptioned_path = build_classes(notes_image_path)



"""
    3) Extract tweets and images for each Community Note
"""

# Constants for tweet extraction

# If you want to run for a subset set start_id and end_id

MODE = 'guest' # 'guest' or 'user'

# Run for both classes separately
for path in [factual_path, miscaptioned_path]:
    input_path = path
    output_path = input_path.replace('.csv', '_tweets.csv')

    image_dir = f"images/{path.replace('.csv', '')}"

    asyncio.run(
        get_tweets_by_ids(input_path, 
            output_path,
            image_dir, 
            MODE, 
            start_id=None, 
            end_id=None
        )
    )


"""
    4) Combine the notes, tweets and images to form the final X-POSE dataset
"""

files = [factual_path.replace(".csv", ""), miscaptioned_path.replace(".csv", "")]

DATASET_PATH = "x_pose_dataset.csv"

process_and_build_dataset(*files, output_path=DATASET_PATH)


# OPTIONAL: Helper to fix image extensions (pngs and jpgs)
# fix_image_bases = [factual_path.replace(".csv", ""), miscaptioned_path.replace(".csv", "")]
# for base in fix_image_bases:
#     fix_image_extensions(f"images/{base}")





"""
    5) Extract claims for each entry and build the claim map
"""

MODEL = 'gemma-3-27b-it' 


# If you want to run for a subset set subset_start and subset_end

# Important: Move images from both classes under the same folder
LOCAL_IMAGES_PATH = "images/"
flatten_image_folders(LOCAL_IMAGES_PATH)
normalize_dataset_image_paths(DATASET_PATH, images_root=LOCAL_IMAGES_PATH)


CLAIMS_PATH = "dataset_gemma_claims.json"


extract_claims_batch(
    path= DATASET_PATH, 
    output_path= CLAIMS_PATH, 
    model_name= MODEL,
    local_images_path= LOCAL_IMAGES_PATH,
    subset_start=None, 
    subset_end=None
)


"""
    6) Extract features for each entry and build the feature map
"""

# If you want to run for a subset set subset_start and subset_end

FEATURES_PATH = "dataset_gemma_features.json"

extract_features_batch(
    path= DATASET_PATH, 
    output_path= FEATURES_PATH, 
    claims_path= CLAIMS_PATH, 
    model_name= MODEL ,
    local_images_path= LOCAL_IMAGES_PATH,
    subset_start=None, 
    subset_end=None
)


"""
    Now we have: 
        - the dataset under DATASET_PATH
        - the claims under CLAIMS_PATH
        - the features under FEATURES_PATH
        - all images under LOCAL_IMAGES_PATH
    After this step, we will also extract:
        - the queries under QUERIES_PATH


    7) Generate the text and image queries for each entry
"""

# If you want to run for a subset set subset_start and subset_end

QUERIES_PATH = "dataset_gemma_queries.json"


extract_queries(
    path= DATASET_PATH, 
    output_path= QUERIES_PATH, 
    model_name= MODEL,
    claims_path= CLAIMS_PATH, 
    features_path= FEATURES_PATH,
    local_images_path= LOCAL_IMAGES_PATH,
    subset_start=None, 
    subset_end=None
)



"""
    8) Run text and image search for each query
"""

# If you want to run for a subset set subset_start and subset_end

# Text and image search results will be saved under this folder
SEARCH_RESULTS_PATH   = "search_results_scrapingdog"

# Number of results to keep for each query
NUM_RESULTS = 10

# Text queries to run
TEXT_QUERIES = 1

run_google_search_scrapingdog(
    queries_path=QUERIES_PATH,
    out_base=SEARCH_RESULTS_PATH,
    num_results=NUM_RESULTS,
    subset_start=None,
    subset_end=None,
    num_text_q=TEXT_QUERIES
)


"""
    9) Run reverse image search for the first image of each entry
"""

# If you want to run for a subset set subset_start and subset_end

# Reverse image search results will be saved under this folder
REVERSE_IMAGE_SEARCH_RESULTS_PATH = "reverse_image_search_results_scrapingdog"


run_reverse_image_search_on_dataset(
    csv_path=DATASET_PATH, 
    out_base=REVERSE_IMAGE_SEARCH_RESULTS_PATH, 
    local_images_path= LOCAL_IMAGES_PATH,
    subset_start=None, 
    subset_end=None, 
    max_size=512, 
    validate=True
)

"""
    Now we have: 
            - the dataset under DATASET_PATH
            - the claims under CLAIMS_PATH
            - the features under FEATURES_PATH
            - all images under LOCAL_IMAGES_PATH
            - the queries under QUERIES_PATH
            - the text and image search results under SEARCH_RESULTS_PATH
            - the reverse image search results under REVERSE_IMAGE_SEARCH_RESULTS_PATH

    10) Scrape the text, image and reverse image search results for each query
"""

scrape_all_entries(SEARCH_RESULTS_PATH)




"""
    11) Scrape the reverse image search results for each entry
"""

# If you want to run for a subset set subset_start and subset_end


process_lens_folder(
    lens_dir=REVERSE_IMAGE_SEARCH_RESULTS_PATH,
    start=None,
    end=None 
)