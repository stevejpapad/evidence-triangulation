import json, os, time
from natsort import natsorted
from pathlib import Path
from .utils import scrape_url

"""Scraping Helpers"""

def load_entry_json(entry_dir):
    """Expect exactly {folder}/{folder}_serp.json; skip if not found."""
    folder = os.path.basename(entry_dir)
    path = os.path.join(entry_dir, f"{folder}_serp.json")
    if not os.path.isfile(path):
        print(f"Missing file: {path} — skipping folder.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None


def save_text_query_results(organic_items, query_id, parent_dir):
    """
    For one text query:
      - Scrape each organic result URL with scrape_url()
      - Save XML and the page-extracted image URLs
      - Also write a compact summary file preserving title/link/snippet/date
    """
    if not organic_items:
        print(f"{query_id}: no organic items.")
        return

    save_dir = os.path.join(parent_dir, query_id)
    os.makedirs(save_dir, exist_ok=True)

    summary = []
    for i, item in enumerate(organic_items):
        title = item.get("title", "")
        link = item.get("link")
        snippet = item.get("snippet")
        date = item.get("date")  # may be absent
        if not link:
            continue

        print(f" → Scraping [{i}] {link}")


        xml_str, image_list = scrape_url(link)  
        

        if xml_str:
            xml_path = os.path.join(save_dir, f"{query_id}_result_{i}.xml")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_str)

        # Save page-extracted image URLs alongside source/snippet/date
        images_payload = []
        for img in (image_list or []):
            images_payload.append({
                "title": title,
                "source_url": link,
                "snippet": snippet,
                "date": date,
                "type": "image_url",
                "image_url": {
                    "url": img.get("url"),
                    "alt": img.get("alt", "")
                }
            })

        if images_payload:
            imgs_path = os.path.join(save_dir, f"{query_id}_result_{i}_scrapedImageURLs.json")
            with open(imgs_path, "w", encoding="utf-8") as f:
                json.dump(images_payload, f, ensure_ascii=False, indent=2)

        # Keep a compact per-result metadata row
        summary.append({
            "position": i + 1,
            "title": title,
            "link": link,
            "snippet": snippet,
            "date": date
        })

        time.sleep(REQUEST_DELAY_SECONDS)

    # Write one small summary per text query
    summary_path = os.path.join(save_dir, f"{query_id}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def save_image_query_results(image_items, query_id, parent_dir):
    """
    For the image query:
      - Save one JSON with entries like:
        { title, source, domain, type:"image_url", image_url:{url:<direct image URL>} }
      - This is LLM-ready and keeps source/domain metadata.
    """
    if not image_items:
        print(f"{query_id}: no images.")
        return

    save_dir = os.path.join(parent_dir, query_id)
    os.makedirs(save_dir, exist_ok=True)

    payload = []
    for it in image_items:
        image_url = it.get("imageUrl") or it.get("link") 
        
        if not image_url:
            continue
        payload.append({
            "title": it.get("title", ""),
            "source": it.get("source"),
            "domain": it.get("domain"),
            "type": "image_url",
            "image_url": {"url": image_url}
        })

    out_path = os.path.join(save_dir, f"{query_id}_googleSearchImageURLs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)




""" Scraping the Text and Image Search Results """

def process_entry_folder(entry_dir):
    """Scrape organic pages + save image URLs for a single entry folder."""

    entry_json = load_entry_json(entry_dir)
    if not entry_json:
        return

    entry_id = entry_json.get("id", os.path.basename(entry_dir))
    print(f"\n Processing {entry_id}")

    #  TEXT QUERIES 
    for tq in (entry_json.get("text_queries") or []):
        qid = tq.get("query_id")
        if not qid:
            continue
        resp = tq.get("response") or {}
        organic_results = resp.get("organic_results") or []
        print(f"\n\n Text query {qid}: {len(organic_results)} organic_results")
        save_text_query_results(organic_results, qid, entry_dir)


    #  IMAGE QUERY 
    img_block = entry_json.get("image_query") or {}
    img_qid = img_block.get("query_id")
    if img_qid:
        img_resp = img_block.get("response") or {}
        images_results = img_resp.get("images_results") or []
        print(f"\n\n Image query {img_qid}: {len(images_results)} images_results")
        save_image_query_results(images_results, img_qid, entry_dir)
    else:
        print(" No image_query block.")

def scrape_all_entries(out_base):
    """Walk OUT_BASE and process each entry folder."""
    if not os.path.isdir(out_base):
        print(f"OUT_BASE not found: {out_base}")
        return

    # Sort 1,2,... instead of 1,10,101,...
    entries = natsorted(
        d for d in os.listdir(out_base)
        if os.path.isdir(os.path.join(out_base, d))
    )

    print(f"Found {len(entries)} entry folders under {out_base}")

    for entry_id in entries:
        process_entry_folder(os.path.join(out_base, entry_id))

"""Scraping the Reverse Image Search Results"""

def save_lens_results(organic_items, entry_id, parent_dir):
    """
    Save reverse-image (Lens) results for one entry.
    Creates: {parent_dir}/{entry_id}_lens/...
      - {entry_id}_lens_result_{i}.xml  (if scrape_url returns xml)
      - {entry_id}_lens_result_{i}_scrapedImageURLs.json  (if scrape_url returned images)
      - {entry_id}_lens_summary.json  (compact metadata for all organic items)
    """
    if not organic_items:
        print(f"{entry_id}: no organic items.")
        return

    save_dir = Path(parent_dir) / f"{entry_id}_lens"
    save_dir.mkdir(parents=True, exist_ok=True)


    summary = []     # compact metadata (always written)
    scrape_log = []  # per-URL status (always written)

    for i, item in enumerate(organic_items):
        title = item.get("title", "")
        link = item.get("link")
        source = item.get("source")
        image_url = item.get("imageUrl")
        thumb_url = item.get("thumbnailUrl")
        snippet = item.get("snippet")  # lens may not provide this
        date = item.get("date")        # lens may not provide this

        if not link:
            scrape_log.append({
                "index": i, "link": None, "status": "skipped", "reason": "missing link"
            })
            continue

        # Add to summary up front (so you always keep what Lens returned)
        summary.append({
            "position": i + 1,
            "title": title,
            "link": link,
            "source": source,
            "imageUrl": image_url,
            "thumbnailUrl": thumb_url,
            "snippet": snippet,
            "date": date
        })

        # Scrape the target page
        try:
            print(f" -> Scraping [{i}] {link}")
            xml_str, image_list = scrape_url(link)

            # Save XML if present
            if xml_str:
                xml_path = save_dir / f"{entry_id}_lens_result_{i}.xml"
                xml_path.write_text(xml_str, encoding="utf-8")

            # Save page-extracted image URLs
            if image_list:
                imgs_payload = []
                for img in image_list:
                    imgs_payload.append({
                        "title": title,
                        "source_url": link,
                        "snippet": snippet,
                        "date": date,
                        "type": "image_url",
                        "image_url": {
                            "url": img.get("url"),
                            "alt": img.get("alt", "")
                        }
                    })
                imgs_path = save_dir / f"{entry_id}_lens_result_{i}_scrapedImageURLs.json"
                imgs_path.write_text(json.dumps(imgs_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            scrape_log.append({"index": i, "link": link, "status": "ok"})
        except Exception as e:
            scrape_log.append({"index": i, "link": link, "status": "error", "error": str(e)})

        time.sleep(2)

    # Always write a small summary and a scrape log
    (save_dir / f"{entry_id}_lens_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (save_dir / f"{entry_id}_lens_scrape_log.json").write_text(
        json.dumps(scrape_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_lens_folder(lens_dir,start=None,end=None):
    """
    Walk a folder with Lens JSON files named `{entry_id}_lens.json`,
    extract organic links & metadata, scrape each, and save results per entry.
    """
    lens_dir = Path(lens_dir)
    if not lens_dir.is_dir():
        print(f"Not a directory: {lens_dir}")
        return

    files = sorted(lens_dir.glob("*_lens.json"))
    print(f"Found {len(files)} Lens JSON files in {lens_dir}")

    # Run a subset 
    if start is not None and end is not None:
        files = files[start:end]

    for jpath in files:
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to read {jpath.name}: {e}")
            continue

        

        # derive entry_id from filename 
        entry_id = jpath.stem.replace("_lens", "").rstrip(")")
        organic = (data.get("response") or {}).get("organic") or []

        print(f"\ -> Processing {entry_id}: {len(organic)} organic items -- keeping first 10")

        # Keep only the first 10 organic results to limit scraping
        organic = organic[:10]

        save_lens_results(organic_items=organic, entry_id=entry_id, parent_dir=str(lens_dir))

    print("\n Done processing all Lens JSON files.")










"""
    Scraping the Text and Image Search Results
"""

OUT_BASE = "search_results_scrapingdog/"

REQUEST_DELAY_SECONDS = 2

# For test run
# scrape_all_entries(OUT_BASE)








""" 
    Scraping the Reverse Image Search Results
"""
START=0
END=10

out_base = "reverse_image_search_results/"

# For test run
# process_lens_folder(out_base,START,END)

