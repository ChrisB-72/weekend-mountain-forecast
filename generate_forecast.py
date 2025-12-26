import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TIMEOUT_SECS = 20

# Priority for choosing "direction wording" and the best extractable values:
# MWIS + Met Office first. Windy/Mountain-Forecast best-effort only.
SOURCE_PRIORITY = ["mwis", "metoffice", "windy", "mountainforecast"]

# Keep inputs small to avoid token/rate issues
MAX_CHARS = {
    "mwis": 9000,
    "metoffice": 9000,
    "windy": 2000,
    "mountainforecast": 2000,
}

# Best-effort URLs (Windy & Mountain-Forecast are generic; often not scrape-friendly).
URLS = {
    "Peaks": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/peak-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/peak-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Snowdon": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/snowdonia-national-park",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/snowdonia",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Brecon": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/brecon-beacons",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/brecon-beacons",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Lakes": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/lake-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/lake-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
}

REGIONS = ["Peaks", "Snowdon", "Brecon", "Lakes"]


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def fetch_text(url: str) -> Tuple[bool, str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MountainForecastBot/1.0; +https://github.com/)"}
        r = requests.get(url, headers=headers, timeout=TIMEOUT_SECS)
        if r.status_code >= 400:
            return False, f"[HTTP {r.status_code}] {url}"
        return True, strip_html(r.text)
    except Exception as e:
        return False, f"[FETCH ERROR: {e}] {url}"


def next_three_days() -> List[date]:
    # "next three days" = today + next 2 days (UTC runner time)
    today = date.today()
    return [today, today + timedelta(days=1), today + timedelta(days=2)]


def day_title(d: date) -> str:
    # e.g. "Wednesday 27 December"
    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    return f"{d.strftime('%A')} {ordinal(d.day)} {d.strftime('%B')}"


def build_region_sources_blob(region: str) -> str:
    parts: List[str] = [f"=== REGION: {region} ==="]
    srcs = URLS[region]
    for src in SOURCE_PRIORITY:
        url = srcs.get(src)
        if not url:
            continue
        ok, text = fetch_text(url)
        cap = MAX_CHARS.get(src, 2000)
        parts.append(f"--- SOURCE: {src.upper()} URL: {url} OK: {ok} ---")
        parts.append(text[:cap])
    return "\n".join(parts)


def ask_region_for_day(
    client: OpenAI,
    region: str,
    target_day: date,
    target_day_title: str,
    sources_blob: str,
) -> Optional[str]:
    # One region + one day per request: keeps token usage low and avoids TPM errors.
    prompt = f"""
You are producing ONE region block for ONE day.

DAY: {target_day_title} (date: {target_day.isoformat()})

SOURCES (best-effort extracted text; may be incomplete):
{sources_blob}

You MUST follow these formatting rules:

1) If a min/max are the same, show only the single value (no range).
   Example: "Rain: 20%"

2) If min/max differ, show ONLY the range, and do NOT include any average/current value.
   Example: "Rain: 10 to 30%"

3) Use "to" for ranges (never hyphens).

4) Include wind direction on both wind lines.
   Example: "Valley wind: W to SW 10 to 20 mph"
   If only one direction is available: "Valley wind: W 10 mph"

5) Region title must be bold (HTML bold tags).
   Example: "<b>Peaks</b>"

6) You may append very brief helpful text ONLY if it adds operational value.
   - keep under ~6 words
   - lowercase
   - no full stops
   Examples: "morning then dry", "gusty", "hill fog", "improving", "poor vis on tops"

7) If a value cannot be extracted, use "n/a" (no range).
   Example: "Freezing level: n/a"

IMPORTANT:
- Best-effort parsing: use numbers only if clearly present in sources.
- Prioritise MWIS and Met Office when choosing wording/interpretation.
- Windy and Mountain-Forecast are optional; include only if clearly extractable.
- Do not add extra headings or summaries.

Return EXACTLY this 7-line block (no extra blank lines at start or end):

<b>{region}</b>
Rain: ...
Valley wind: ...
Hill wind: ...
Valley temp: ...
Hill temp: ...
Cloud base: ...
Freezing level: ...
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt if txt else None
    except Exception as e:
        print(f"Generation failed for {region} {target_day_title}: {e}")
        return None


def compute_confidence(have_counts: List[int], total_expected: int) -> str:
    """
    Very simple heuristic:
    - High: most regions produced blocks and few 'n/a'
    - Moderate: default
    - Low: many missing values
    """
    # have_counts is number of blocks produced per day (should be len(REGIONS))
    if all(c == total_expected for c in have_counts):
        return "Moderate"  # keep conservative; you can tune to "High" if you like
    return "Low"


def main() -> int:
    # Failure-safe: do not overwrite index.html on failures
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing. Leaving existing index.html unchanged.")
        return 0

    client = OpenAI(api_key=api_key)

    days = next_three_days()
    titles = [day_title(d) for d in days]
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Pre-fetch per region once (then reuse for 3 days)
    region_sources: Dict[str, str] = {}
    for region in REGIONS:
        print(f"Fetching sources for region: {region}")
        region_sources[region] = build_region_sources_blob(region)

    day_blocks: List[str] = []
    produced_counts: List[int] = []

    for d, t in zip(days, titles):
        blocks: List[str] = []
        for region in REGIONS:
            blk = ask_region_for_day(client, region, d, t, region_sources[region])
            if not blk:
                print(f"Missing output for {region} on {t}; leaving existing index.html unchanged.")
                return 0
            blocks.append(blk)
        produced_counts.append(len(blocks))
        day_blocks.append("\n\n".join(blocks).strip())

    confidence = compute_confidence(produced_counts, len(REGIONS))

    # Header (exact placement for Confidence + Last updated)
    header = "\n".join([
        "Mountain Forecast",
        f"Next 3 days ({titles[0]} to {titles[-1]})",
        f"Confidence: {confidence}",
        f"Last updated: {updated_utc}",
    ])

    # Build HTML from template placeholders
    with open("page_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    html = (template
            .replace("{{HEADER}}", header)
            .replace("{{DAY1_TITLE}}", titles[0])
            .replace("{{DAY2_TITLE}}", titles[1])
            .replace("{{DAY3_TITLE}}", titles[2])
            .replace("{{DAY1_BODY}}", day_blocks[0])
            .replace("{{DAY2_BODY}}", day_blocks[1])
            .replace("{{DAY3_BODY}}", day_blocks[2]))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Forecast generated and index.html updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
