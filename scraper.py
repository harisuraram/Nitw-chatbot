# ...existing code...
import os
import time
import hashlib
import requests
import re
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# optional progress bar
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False
    print("tqdm not installed — progress bars disabled. Install with: pip install tqdm")


def fetch_page_source(driver, url, wait_seconds=10, extra_wait=1):
    try:
        print(f"[fetch_page_source] loading: {url}")
        if not driver.service.is_connectable():
            print("[fetch_page_source] Chrome driver appears disconnected, restarting.")
            driver.quit()
            service = Service(ChromeDriverManager().install())
            new_driver = webdriver.Chrome(service=service, options=chrome_options)
            driver = new_driver  # update reference
        driver.get(url)
        WebDriverWait(driver, wait_seconds).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        if extra_wait:
            time.sleep(extra_wait)  # allow JS to render dynamic content
        page = driver.page_source
        print(f"[fetch_page_source] loaded: {url} (size={len(page)} chars)")
        return page, driver
    except Exception as e:
        print(f"[fetch_page_source] Failed to load {url}: {e}")
        return "", driver


def is_same_domain(link, base_url):
    return urlparse(link).netloc == urlparse(base_url).netloc


def normalize_link(base, href):
    if not href:
        return None
    href = href.strip()
    if href.lower().startswith("javascript:") or href.startswith("#"):
        return None
    link = urljoin(base, href)
    if base in link and link.count(base) > 1:
        return None  # skip recursive "stuti/www.nitw.ac.in/stuti" type
    return link


def looks_like_pdf(link):
    if not link:
        return False
    parsed = urlparse(link)
    path = parsed.path.lower()
    return path.endswith(".pdf")


class CrawlStats:
    def __init__(self):
        self.all_discovered_links = set()
        self.visited_links = set()

    def add_discovered(self, link):
        self.all_discovered_links.add(link)

    def add_visited(self, link):
        self.visited_links.add(link)

    def print_stats(self):
        unvisited = self.all_discovered_links - self.visited_links
        print("\n=== Crawl Statistics ===")
        print(f"Total links discovered: {len(self.all_discovered_links)}")
        print(f"Links visited: {len(self.visited_links)}")
        print(f"Links not visited: {len(unvisited)}")
        if unvisited:
            print("\nUnvisited links:")
            for link in sorted(unvisited):
                print(f"- {link}")


def get_pdf_links(driver, url, base_url, visited, depth, max_depth, stats):
    if depth > max_depth or url in visited:
        return set(), driver
    print(f"[crawl] Visiting: {url} (depth={depth}) visited={len(visited)}")
    visited.add(url)
    stats.add_visited(url)
    html, driver = fetch_page_source(driver, url)
    if not html:
        return set(), driver

    soup = BeautifulSoup(html, "html.parser")
    found = set()

    # collect pdf links on this page
    page_pdf_count = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        link = normalize_link(url, href)
        if not link:
            continue
        if looks_like_pdf(link):
            if link not in found:
                page_pdf_count += 1
            found.add(link)
    if page_pdf_count:
        print(f"[crawl] Found {page_pdf_count} PDF(s) on {url}")

    # recurse into other pages within same domain
    if depth < max_depth:
        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            link = normalize_link(url, href)
            if not link:
                continue
            if is_same_domain(link, base_url):
                stats.add_discovered(link)
                if link not in visited and not looks_like_pdf(link):
                    candidates.append(link)

        if candidates:
            print(f"[crawl] Recursing {len(candidates)} link(s) from {url}")
        for link in candidates:
            try:
                pdf_links, driver = get_pdf_links(
                    driver, link, base_url, visited, depth + 1, max_depth, stats
                )
                found.update(pdf_links)
            except KeyboardInterrupt:
                print("[crawl] KeyboardInterrupt received, stopping recursion.")
                return found, driver

    return found, driver


def safe_filename_from_url(link, download_dir):
    name = os.path.basename(urlparse(link).path)
    if not name:
        # fallback to hash when URL doesn't end with a filename
        name = hashlib.sha1(link.encode("utf-8")).hexdigest() + ".pdf"
    # sanitize filename minimally
    name = "".join(c for c in name if c.isalnum() or c in "._-")
    return os.path.join(download_dir, name)


# Add regex patterns (edit these to suit). Patterns are applied to the file name only.
SKIP_FILE_PATTERNS = [
    r"(?i)tender",  # example: skip filenames containing "tender" (case-insensitive)
    r"(?i)niet",  # skip filenames containing "niet"
    r"(?i)form",  # skip filenames containing "form"
    r"(?i)niq",
    r"(?i)bid",
    r"(?i)meeting",
    r"(?i)minutes",
    r"(?i)NIT_2025",
    # add more patterns as needed, these are regular expressions
]
SKIP_COMPILED = [re.compile(p) for p in SKIP_FILE_PATTERNS]


def download_files(links, download_dir, timeout=20):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible)"})
    os.makedirs(download_dir, exist_ok=True)
    total = len(links)
    if TQDM_AVAILABLE:
        iterator = tqdm(links, desc="Overall downloads", unit="file")
    else:
        iterator = links

    for link in iterator:
        # derive the raw filename from the URL path (percent-decoded)
        raw_name = unquote(os.path.basename(urlparse(link).path))
        # if URL has no filename, fallback to the safe filename we would generate
        if not raw_name:
            raw_name = os.path.basename(safe_filename_from_url(link, download_dir))

        # check only the filename against skip patterns
        skip = False
        for cre in SKIP_COMPILED:
            if cre.search(raw_name):
                print(f"[download] Skipping (pattern match) {raw_name} from {link}")
                skip = True
                break
        if skip:
            continue

        filename = safe_filename_from_url(link, download_dir)
        if os.path.exists(filename):
            print(f"[download] Already exists, skipping: {filename}")
            continue
        print(f"[download] Downloading {link} -> {filename}")
        try:
            r = session.get(
                link, timeout=timeout, stream=True, verify=False, allow_redirects=False
            )
            if 300 <= r.status_code < 400:
                print(
                    f"[download] Skipping {link} (redirects to dead host {r.headers.get('Location')})"
                )
                continue
            r.raise_for_status()
            total_bytes = int(r.headers.get("content-length", 0))
            if TQDM_AVAILABLE:
                with open(filename, "wb") as f, tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(filename),
                    leave=False,
                ) as pbar:
                    for chunk in r.iter_content(1024 * 32):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # fallback: write without a progress bar, but show simple byte counts
                written = 0
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(1024 * 32):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
                print(f"[download] Completed {filename} ({written} bytes)")
        except Exception as e:
            print(f"[download] Failed to download {link}: {e}")


# ...existing code...

if __name__ == "__main__":
    # Configuration
    start_urls = [
        "https://www.nitw.ac.in",
        "https://www.nitw.ac.in/ap",
        "https://nitw.ac.in/keydocuments",
        # Add more starting URLs as needed
    ]
    max_depth = 5  # pages deep to follow links
    download_dir = "pdfs"

    # Setup headless Chrome (uses webdriver-manager to install driver)
    global chrome_options
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-dev-shm-usage")

    print("[main] Installing/starting Chrome driver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    print("[main] Chrome driver started.")

    try:
        visited = set()  # This will now track visited pages across all start URLs
        stats = CrawlStats()
        all_pdf_links = set()  # To store unique PDFs across all crawls

        for start_url in start_urls:
            print(f"\n[main] Starting crawl from {start_url} up to depth {max_depth}")
            pdf_links, driver = get_pdf_links(
                driver, start_url, start_url, visited, 0, max_depth, stats
            )
            all_pdf_links.update(pdf_links)  # Add new PDFs to the set

        all_pdf_links = sorted(all_pdf_links)  # Sort for consistent order
        print(
            f"\n[main] Found {len(all_pdf_links)} unique PDF(s) across all starting points."
        )
        stats.print_stats()
        if all_pdf_links:
            download_files(all_pdf_links, download_dir)
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — exiting early.")
    finally:
        driver.quit()
        print("[main] Chrome driver quit.")