## Evidence Extraction

This stage takes the generated queries and media from previous steps and runs web and reverse image search, then scrapes and structures the results for later LLM-based evidence reasoning.

### Files

- **`text_image_search_scrapingdog.py`**
  - Reads the JSON produced by `query_generation.py` (e.g. the `dataset_gemma_queries.json` output)
  - For each post `id`, runs:
    - Up to `num_text_q` **text web searches** via Scrapingdog Google Search
    - One **image search** via Scrapingdog Google Images
  - Stores, per entry, the raw SERP responses

- **`reverse_image_search_scrapingdog.py`**
  - Reads the main **dataset CSV**
  - For each entry:
    - Picks the first local image path from `tweetMediaFiles`
    - Uploads the local image to Imgbb and gets a temporary public URL
    - Optionally validates the hosted image (size, simple keyword filter)
    - Calls Scrapingdog **Google Lens (reverse image search)**
  - Saves raw reverse-image results as JSON

- **`scrape_text_image_reverse_results.py`**
  - Consumes the saved search outputs and turns them into **scraped pages and LLM-ready evidence JSON**:
    - For each `id` folder:
      - For each **text query**:
        - Scrapes every organic result URL with Playwright + Trafilatura libraries
        - Saves cleaned XML and extracted page image URLs
        - Writes a compact per-query `*_summary.json` with title/link/snippet/date
      - For the **image query**:
        - Writes `*_googleSearchImageURLs.json` with normalized image URL records (title, source, domain, image URL)
    - For each reverse image JSON in `reverse_image_search_results/`:
      - Reads `{id}_lens.json`
      - For the top organic results:
        - Scrapes each target page
        - Saves XML and extracted image URLs
        - Writes metadata and log status per URL

---

### Setup Environment

- Setup a `.env` file with
  - [ScrapingDog]("https://www.scrapingdog.com/) API key (you can use any provider) and
  - [IMGBB]("https://api.imgbb.com/") key for remporary image upload (it's free)

---

### Flow & Sequence

1. **Input**
   - Queries from `query_generation`
   - Main dataset CSV with tweet text and local image paths
   - Environment variables: `SCRAPING_DOG_KEY`, `IMGBB_API_KEY`

2. **Run text & image search (`text_image_search_scrapingdog.py`)**
   - For each post:
     - Run Scrapingdog **text search** on the top N text queries
     - Run Scrapingdog **image search** on the image search query
   - Save all raw results data

3. **Run reverse image search (`reverse_image_search_scrapingdog.py`)**
   - For each dataset row:
     - Upload the first local image to Imgbb
     - (Optionally) validate the hosted image
     - Call Scrapingdog **Google Lens**
   - Save raw Lens JSON

4. **Scrape and structure results (`scrape_text_image_reverse_results.py`)**
   - For each SERP JSON :
     - Scrape linked pages, extract main content as XML and collect image URLs
     - Save per-result XML + image URL JSON and per-query summaries
   - For each Lens JSON:
     - Scrape the top Lens organic links, save XML + image URL JSON, and write per-entry summaries/logs

5. **Outputs**
   - Cleaned web-page XML, normalized image URL lists, and compact metadata JSONs for:
     - Text search results
     - Image search results
     - Reverse image (Lens) results
