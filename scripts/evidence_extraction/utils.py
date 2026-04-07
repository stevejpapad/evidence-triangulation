import requests
import os
from PIL import Image
from io import BytesIO
import base64
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import signal
import trafilatura
from playwright.sync_api import sync_playwright, Error as PWError
from dotenv import load_dotenv

load_dotenv()

# Setup Scrapingdog API key
SCRAPINGDOG_API_KEY = os.getenv("SCRAPING_DOG_KEY")

# Setup Imgbb API key
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

def build_excluded_query(base_query):
    """
        Takes as input an array of blocked domains of Google Search 
        and formats the query for the API 
    """
    BLOCKED_DOMAINS = ["facebook.com", "instagram.com", "pinterest.com", "youtube.com", "x.com", "flickr.com","reddit.com", "titok.com"]

    exclusions = " ".join([f"-site:{domain}" for domain in BLOCKED_DOMAINS])
    return f"{base_query} {exclusions}".strip()

def scrapingdog_search_call(base_query, image_search=False, numResults=10):
    """
    Scrapingdog Google Search API wrapper.
    
    - base_query: query string
    - image_search
        True => Google Images endpoint
        False => Google Web Search
    - numResults: number of results to request (1–100)
    """
    if not SCRAPINGDOG_API_KEY:
        raise RuntimeError("Missing SCRAPINGDOG_API_KEY")

    endpoint = "https://api.scrapingdog.com/google_images" if image_search else "https://api.scrapingdog.com/google"

    # Build the excluded query
    query = build_excluded_query(base_query)

    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "query": query,
        "results": numResults,
        "country": "us",
        "advance_search": "false",
        "domain": "google.com"
    }

    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def upload_local_to_imgbb(path, base_path, max_size=None,expiration=3600):
    """ 
        Converts a local Image into a URL by uploading to imgbb 
        Optionally resizes the image while keeping the aspect ratio.
    """

    image_path = os.path.join(base_path, path)


    # Some images are png but mistakenly are saved as jpg in dataset column
    if not os.path.exists(image_path) and path.endswith(".jpg"):
        path = path.replace(".jpg", ".png")
        image_path = os.path.join(base_path, path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize if max_size is provided
    if max_size:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "key": IMGBB_API_KEY,
            "image": img_b64,
            "expiration": expiration # Auto-delete after 1 hour
        }

        res = requests.post("https://api.imgbb.com/1/upload", data=payload)
    else:
        with open(image_path, "rb") as f:
            res = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": IMGBB_API_KEY, "expiration": 3600}, # Auto-delete after 1 hour
                files={"image": f}
            )

    res.raise_for_status()
    data = res.json()
    return data["data"]["url"]

def is_valid_image(url, min_width=100, min_height=100, req_timeout=10):
    """
    Checks if an image URL points to a valid image for analysis:
        - Not too small (based on dimensions)
        - Doesn't contain undesirable keywords
    Returns:
        (is_valid, reason)
    """

    # Try to load the image and check dimensions
    try:
        response = requests.get(url, timeout=req_timeout)
        img = Image.open(BytesIO(response.content))
        width, height = img.size
        if width < min_width or height < min_height:
            return False, "Filtered by size"
    except Exception as e:
        return False, f"Failed to load image ({str(e)})"
    
    # Then also apply keyword filtering 
    bad_keywords = ["favicon", "logo", "icon", "banner"]

    if any(kw in url.lower() for kw in bad_keywords):
        return False, "Filtered by keyword"
    
    # If the image is valid, return True and None
    return True, None
    
def scrapingdog_reverse_image_call(image_url):
    """
    Scrapingdog Google Lens (reverse image search) API wrapper.

    - image_url: public URL of the image to search
    """
    if not SCRAPINGDOG_API_KEY:
        raise RuntimeError("Missing SCRAPINGDOG_API_KEY")

    endpoint = "https://api.scrapingdog.com/google_lens"

    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "url": image_url
    }

    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

"""
    Scraping Helpers
"""
 
def looks_non_html(url):
    """Heuristic: skip sitemaps/feeds/binaries that tend to hang Playwright."""

    NON_HTML_EXTS = {".xml", ".rss", ".atom"}

    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext in NON_HTML_EXTS:
        return True
    # sitemap-ish endpoints even without extension
    low = url.lower()
    if "sitemap" in low or "feed" in low or "/rss" in low:
        return True
    return False

def _timeout_handler(signum, frame):
    raise TimeoutException()

class TimeoutException(Exception):
    pass

def extract_and_fix_image_urls(xml_response, base_url):
    """
    Extracts all image URLs from a Trafilatura XML response and resolves relative URLs.
    Returns a list of absolute image URLs.
    """

    if not isinstance(xml_response, str):
        return []
    
    try:
        root = ET.fromstring(xml_response)
    except ET.ParseError:
        return []
    
    image_data = []
    for graphic in root.iter("graphic"):
        src = graphic.attrib.get("src")
        alt = graphic.attrib.get("alt", "")

        if not src:
            continue

        full_url = src if src.startswith("http") else urljoin(base_url, src)

        image_data.append({
            "url": full_url,
            "alt": alt
        })

    return image_data


def fetch_dynamic_html(url, hard_timeout=30):
    """
    Fetch HTML with Playwright, aborting after hard_timeout seconds 
    """
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(hard_timeout)   # start watchdog

    browser = None
    context = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True) # --> Set false for debugging
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9"
                },
                viewport={"width": 1280, "height": 800}
            )
            page = context.new_page()

            timeout = 10000  # 10 seconds per Playwright action

            page.set_default_navigation_timeout(timeout) 
            page.set_default_timeout(timeout)

            #################################
            ## TO DISMISS COOKIES BANNERS ##
            ###############################
            cookie_texts = [
                "Accept", "Continue", "Agree", "OK", "I agree", "AGREE", "Consent",
                "Allow all", "Allow all cookies", "Accept all", "Accept all cookies",
                "Accept All", "Accept all" "Agree & continue", "Accept cookies"
                ]   
                
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            except PWError as e:
                print(f" goto failed for {url}: {e}")
                return None  # skip this URL cleanly


            for phrase in cookie_texts:
                try:
                    el = page.get_by_text(phrase, exact=False)
                    if el.count():
                        el.first.click(timeout=1500)
                        print("Clicked a cookie consent button!")
                        break
                except Exception:
                    pass
            
                
            try:
                page.wait_for_timeout(500)  # Wait 0.5s for JS to load
                html = page.content()
            except PWError as e:
                print(f"page.content() failed for {url}: {e} — falling back to outerHTML")
                try:
                    html = page.evaluate("document.documentElement.outerHTML")
                except Exception as ee:
                    print(f"outerHTML also failed for {url}: {ee}")
                    html = None
            return html
        
    except TimeoutException:
        print(f"Hard Timeout after {hard_timeout}s fetching {url}")
        return None
    finally:
        signal.alarm(0)  # cancel watchdog
        if context:
            try: context.close()
            except: pass
        if browser:
            try: browser.close()
            except: pass

def scrape_url(url):

    # Fetch using playwright
    try:
        if looks_non_html(url):
            print(f"Skipping non-HTML URL: {url}")
            return None, []

        html = fetch_dynamic_html(url)
        if not html:
            return None, []


        result = trafilatura.extract(html, output_format="xml", with_metadata=True, include_tables=True, include_images=True)

        if result is None:
            print(f"Failed to extract from {url}")
            return None, []

        image_data = extract_and_fix_image_urls(result, url)

        return result, image_data
    
    except Exception as e:
        print(f"[scrape_url] Failed on {url}: {e}")
        return None, []