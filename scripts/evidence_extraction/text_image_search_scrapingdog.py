import os, json
import time
from .utils import scrapingdog_search_call

def run_google_search_scrapingdog(
    queries_path,
    out_base,
    num_results,
    subset_start, 
    subset_end,
    num_text_q=1
):
    """
    Reads the query file, runs Google search for up to 3 text queries + first image query per entry,
    and writes a consolidated JSON of all results (keyed by entry id)
    """

    REQUEST_DELAY_SECONDS = 2

    os.makedirs(out_base, exist_ok=True)

    with open(queries_path, "r", encoding="utf-8") as f:
        entries = json.load(f)


    """
        To run for a specific subset of entries
        add subset_start and subset_end
    """
    for entry in entries[subset_start:subset_end]:
        
        entry_id = entry["id"]
        entry_class = entry.get("class", None)

        queries_block = entry.get("queries")

        if not queries_block:
            print(f"Skipping entry {entry_id} (no queries)")
            continue

        # Keep the first num_text_q text queries 
        text_queries = queries_block.get("text_queries", [])[:num_text_q]

        # Keep the first image query
        img_block = queries_block.get("image_search_query")
        image_query = None

        if isinstance(img_block, list) and img_block:
            image_query = img_block[0]
        elif isinstance(img_block, str) and img_block.strip():
            image_query = img_block.strip()

        print(f"\nProcessing entry: {entry_id}")

        entry_result = {
            "id": entry_id,
            "class": entry_class,
            "text_queries": [],
            "image_query": None,
        }

        """
            Run the text queries
                Keep the first num_results results for each query
        """
        for i, q in enumerate(text_queries, start=1):
            query_id = f"{entry_id}_t{i}"
            print(f" TEXT [{i}/{len(text_queries)}]: {q}  (Query ID: {query_id})")
            try:
                resp = scrapingdog_search_call(
                    q,
                    image_search=False,
                    numResults=num_results,
                )
                entry_result["text_queries"].append({
                    "query_id": query_id,
                    "query": q,
                    "num_results": num_results,
                    "response": resp
                })
            except Exception as e:
                entry_result["text_queries"].append({
                    "query_id": query_id,
                    "query": q,
                    "num_results": num_results,
                    "error": str(e)
                })

            time.sleep(REQUEST_DELAY_SECONDS)

        """
            Run the image query
                Keep the first num_results results for each query
        """
        if image_query:
            query_id = f"{entry_id}_img1"
            print(f" IMAGE: {image_query}  (Query ID: {query_id})")
            try:
                resp = scrapingdog_search_call(
                    image_query,
                    image_search=True,
                    numResults=num_results,
                )
                entry_result["image_query"] = {
                    "query_id": query_id,
                    "query": image_query,
                    "num_results": num_results,
                    "response": resp
                }
            except Exception as e:
                entry_result["image_query"] = {
                    "query_id": query_id,
                    "query": image_query,
                    "num_results": num_results,
                    "error": str(e)
                }

            time.sleep(REQUEST_DELAY_SECONDS)


        """
            Save results per entry
        """
        entry_dir = os.path.join(out_base, str(entry_id))
        os.makedirs(entry_dir, exist_ok=True)
        out_path = os.path.join(entry_dir, f"{entry_id}_serp.json")

        with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entry_result, f, ensure_ascii=False, indent=2)

        print(f"Saved to {out_path}")

"""
    Run Google Searches with Scrapingdog on the generated queries
"""

# Example run


# QUERY_FILE= f"dataset_gemma_queries.json"

# OUT_BASE   = "search_results_scrapingdog"

# # Number of results to keep for each query
# NUM_RESULTS = 10

# # Text queries to run
# TEXT_QUERIES = 1

# START=0
# END=10 

# run_google_search_scrapingdog(
#     queries_path=QUERY_FILE,
#     out_base=OUT_BASE,
#     num_results=NUM_RESULTS,
#     subset_start=START,
#     subset_end=END,
#     num_text_q=TEXT_QUERIES
# )
















