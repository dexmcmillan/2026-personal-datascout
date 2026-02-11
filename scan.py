"""
Data Scout — Morning data journalism briefing generator.
Fetches RSS feeds and CKAN APIs, filters with Gemini, builds a static HTML briefing.
"""

import json
import os
import time
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader

# --- Configuration ---

RSS_FEEDS = {
    "The Pudding": "https://pudding.cool/feed/index.xml",
    "ProPublica": "https://www.propublica.org/feeds/propublica/main",
    "The Markup": "https://themarkup.org/feeds/rss.xml",
    "FiveThirtyEight": "https://abcnews.go.com/abcnews/538",
    "Reuters Graphics": "https://www.reuters.com/graphics/RSS",
    "Source OpenNews": "https://source.opennews.org/feed/",
    "GIJN": "https://gijn.org/feed/",
    "Datajournalism.com": "https://datajournalism.com/read/rss",
}

CANADIAN_RSS_FEEDS = {
    "Statistics Canada": "https://www150.statcan.gc.ca/n1/dai-quo/rssFeed-eng.xml",
}

CKAN_APIS = {
    "Toronto Open Data": {
        "search_url": "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_search",
        "base_url": "https://open.toronto.ca/dataset",
    },
    "Canada Open Data": {
        "search_url": "https://open.canada.ca/data/api/3/action/package_search",
        "base_url": "https://open.canada.ca/data/en/dataset",
    },
}

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"
TEMPLATE_DIR = BASE_DIR / "templates"
STATE_FILE = DATA_DIR / "seen_items.json"

HOURS_LOOKBACK = 48  # Look back 48 hours for items


def setup_gemini():
    """Setup Gemini API client."""
    api_key = None
    key_file = BASE_DIR / "gemini_api_key.txt"
    if key_file.exists():
        api_key = key_file.read_text().strip()
        if api_key == "PASTE_YOUR_GEMINI_API_KEY_HERE":
            api_key = None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Add your API key to gemini_api_key.txt or set GEMINI_API_KEY env var"
        )
    return genai.Client(api_key=api_key)


def load_state():
    """Load previously seen item IDs."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(seen):
    """Save seen item IDs. Prune entries older than 14 days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
    pruned = {k: v for k, v in seen.items() if v > cutoff}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(pruned, indent=2))


def item_id(title, link):
    """Generate a stable ID for an item."""
    raw = f"{title}|{link}"
    return hashlib.md5(raw.encode()).hexdigest()


def parse_feed_date(entry):
    """Extract a datetime from a feed entry, or None."""
    for attr in ("published_parsed", "updated_parsed"):
        tp = getattr(entry, attr, None)
        if tp:
            try:
                from calendar import timegm
                return datetime.fromtimestamp(timegm(tp), tz=timezone.utc)
            except Exception:
                pass
    return None


def fetch_rss_feeds(cutoff_dt, seen):
    """Fetch items from all RSS feeds published after cutoff_dt."""
    items = []
    for source_name, url in {**RSS_FEEDS, **CANADIAN_RSS_FEEDS}.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue
                iid = item_id(title, link)
                if iid in seen:
                    continue
                pub_date = parse_feed_date(entry)
                if pub_date and pub_date < cutoff_dt:
                    continue
                # Extract summary text
                summary_html = entry.get("summary", "") or entry.get("description", "")
                summary_text = BeautifulSoup(summary_html, "html.parser").get_text(
                    separator=" ", strip=True
                )[:500]
                is_canadian = source_name in CANADIAN_RSS_FEEDS
                items.append(
                    {
                        "id": iid,
                        "title": title,
                        "link": link,
                        "source": source_name,
                        "summary_raw": summary_text,
                        "is_canadian_data": is_canadian,
                    }
                )
            print(f"  Fetched {source_name}: {len(feed.entries)} entries")
        except Exception as e:
            print(f"  Error fetching {source_name}: {e}")
    return items


def fetch_ckan_updates(cutoff_dt, seen):
    """Fetch recently changed datasets from CKAN APIs."""
    items = []
    cutoff_str = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S")
    for source_name, config in CKAN_APIS.items():
        try:
            resp = requests.get(
                config["search_url"],
                params={
                    "sort": "metadata_modified desc",
                    "rows": 20,
                    "fq": f"metadata_modified:[{cutoff_str}Z TO NOW]",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            packages = data.get("result", {}).get("results", [])
            for pkg in packages:
                name = pkg.get("name", pkg.get("id", ""))
                # Handle bilingual titles (Canada Open Data uses dict)
                raw_title = pkg.get("title", name)
                if isinstance(raw_title, dict):
                    title = raw_title.get("en", raw_title.get("fr", name))
                else:
                    title = raw_title
                link = f"{config['base_url']}/{name}"
                excerpt = pkg.get("excerpt", pkg.get("notes", "")) or ""
                if isinstance(excerpt, dict):
                    excerpt = excerpt.get("en", "")
                iid = item_id(title, link)
                if iid in seen:
                    continue
                items.append(
                    {
                        "id": iid,
                        "title": title,
                        "link": link,
                        "source": source_name,
                        "summary_raw": excerpt[:500] if excerpt else f"Dataset updated on {source_name}",
                        "is_canadian_data": True,
                    }
                )
            print(f"  Fetched {source_name}: {len(packages)} datasets")
        except Exception as e:
            print(f"  Error fetching {source_name}: {e}")
    return items


def filter_with_gemini(client, items):
    """Use Gemini Flash to score and summarize items."""
    if not items:
        return []

    # Build the item list for the prompt
    items_text = ""
    for i, item in enumerate(items):
        items_text += (
            f"\n[{i}] Title: {item['title']}\n"
            f"    Source: {item['source']}\n"
            f"    Link: {item['link']}\n"
            f"    Excerpt: {item['summary_raw'][:300]}\n"
        )

    prompt = f"""You are a data editor at a Canadian newspaper. From these items, select the most newsworthy for a Canadian audience.

For each item worth including, provide:
- index: the item number from the list
- summary: a one-line summary (max 25 words)
- why: why it matters for a Canadian audience (max 20 words)
- score: relevance score 1-5 (5 = most relevant)

Prioritize: Canadian data, economy, housing, healthcare, climate, data journalism techniques.
Skip items that are clearly irrelevant, promotional, or not data-related.

Return ONLY valid JSON — an array of objects with keys: index, summary, why, score.
Example: [{{"index": 0, "summary": "...", "why": "...", "score": 4}}]

Items to evaluate:
{items_text}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
            ),
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            text = text.strip()
        scored = json.loads(text)
    except Exception as e:
        print(f"  Gemini error: {e}")
        # Fallback: include all items with default score
        scored = [
            {"index": i, "summary": item["summary_raw"][:100], "why": "", "score": 3}
            for i, item in enumerate(items)
        ]

    # Merge scores back into items
    results = []
    for s in scored:
        idx = s.get("index")
        if idx is not None and 0 <= idx < len(items):
            item = items[idx].copy()
            item["summary"] = s.get("summary", item["summary_raw"][:100])
            item["why"] = s.get("why", "")
            item["score"] = s.get("score", 3)
            results.append(item)
    return results


def get_archive_dates():
    """List existing archive dates, newest first."""
    if not ARCHIVE_DIR.exists():
        return []
    dates = sorted(
        [f.stem for f in ARCHIVE_DIR.glob("*.html")],
        reverse=True,
    )
    return dates[:30]  # Keep last 30 days visible


def build_html_page(scored_items, today_str):
    """Generate the briefing HTML and write to docs/."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("briefing.html")

    top_stories = [i for i in scored_items if i["score"] >= 4 and not i["is_canadian_data"]]
    canadian_data = [i for i in scored_items if i["is_canadian_data"] and i["score"] >= 3]
    worth_a_look = [
        i for i in scored_items
        if i["score"] == 3 and not i["is_canadian_data"] and i not in canadian_data
    ]

    # Sort each section by score descending
    top_stories.sort(key=lambda x: x["score"], reverse=True)
    canadian_data.sort(key=lambda x: x["score"], reverse=True)
    worth_a_look.sort(key=lambda x: x["score"], reverse=True)

    now = datetime.now(timezone.utc).astimezone()
    date_formatted = now.strftime("%A, %B %-d, %Y")
    scan_time = now.strftime("%-I:%M %p")
    source_count = len(RSS_FEEDS) + len(CANADIAN_RSS_FEEDS) + len(CKAN_APIS)
    archive_dates = [d for d in get_archive_dates() if d != today_str]

    html = template.render(
        date_formatted=date_formatted,
        top_stories=top_stories,
        canadian_data=canadian_data,
        worth_a_look=worth_a_look,
        source_count=source_count,
        scan_time=scan_time,
        archive_dates=archive_dates,
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write main page and archive copy
    (DOCS_DIR / "index.html").write_text(html)
    (ARCHIVE_DIR / f"{today_str}.html").write_text(html)
    print(f"  Wrote docs/index.html and docs/archive/{today_str}.html")


def main():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=HOURS_LOOKBACK)

    print(f"Data Scout — {today_str}")
    print(f"Looking back {HOURS_LOOKBACK} hours from {cutoff_dt.isoformat()}")

    # Load state
    seen = load_state()
    print(f"Previously seen items: {len(seen)}")

    # Fetch
    print("\nFetching RSS feeds...")
    rss_items = fetch_rss_feeds(cutoff_dt, seen)
    print(f"  Total new RSS items: {len(rss_items)}")

    print("\nFetching CKAN updates...")
    ckan_items = fetch_ckan_updates(cutoff_dt, seen)
    print(f"  Total new CKAN items: {len(ckan_items)}")

    all_items = rss_items + ckan_items
    print(f"\nTotal new items to evaluate: {len(all_items)}")

    if not all_items:
        print("No new items found. Building empty briefing.")
        build_html_page([], today_str)
    else:
        # Filter with Gemini
        print("\nFiltering with Gemini...")
        client = setup_gemini()
        scored_items = filter_with_gemini(client, all_items)
        print(f"  Gemini returned {len(scored_items)} scored items")

        # Build page
        print("\nBuilding HTML page...")
        build_html_page(scored_items, today_str)

    # Mark all fetched items as seen
    now_iso = datetime.now(timezone.utc).isoformat()
    for item in all_items:
        seen[item["id"]] = now_iso
    save_state(seen)
    print(f"\nState saved. Total seen items: {len(seen)}")
    print("Done.")


if __name__ == "__main__":
    main()
