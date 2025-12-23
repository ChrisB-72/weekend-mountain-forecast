import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TIMEOUT_SECS = 20

# Prefer MWIS + Met Office for "current/best" values
SOURCE_PRIORITY = ["mwis", "metoffice", "windy", "mountainforecast"]

# Keep inputs small (tune if needed)
MAX_CHARS = {
    "mwis": 9000,
    "metoffice": 9000,
    "windy": 2500,
    "mountainforecast": 2500,
}

URLS = {
    "Peak District": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/peak-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/peak-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Snowdonia": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/snowdonia-national-park",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/snowdonia",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Brecon Beacons": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/brecon-beacons",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/brecon-beacons",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Lake District": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/lake-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/lake-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
}


def upcoming_weekend_dates() -> str:
    today = date.today()
    days_until_sat = (5 - today.weekday()) % 7
    saturday = today + timedelta(days=days_until_sat)
    sunday = saturday + timedelta(days=1)

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    return f"{ordinal(saturday.day)} / {ordinal(sunday.day)} {saturday.strftime('%B')}"


def strip_html(html: str) -> str:
    # Remove scripts/styles
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    # Remove tags
    text = re.sub(r"(?s)<.*?>", " ", html)
    # Unescape common entities lightly (keep it simple)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_text(url: str) -> Tuple[bool, str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WeekendMountainForecastBot/1.0; +https://github.com/)"
        }
        r = requests.get(url, headers=headers, timeout=TIMEOUT_SECS)
        if r.status_code >= 400:
            return False, f"[HTTP {r.status_code}] {url}"
        return True, strip_html(r.text)
    except Exception as e:
        return False, f"[FETCH ERROR: {e}] {url}"


def build_region_sources_blob(region: str) -> str:
    parts: List[str] = [f"=== REGION: {region} ==="]
    sources = URLS[region]
    for src in SOURCE_PRIORITY:
        url = sources.get(src)
        if not url:
            continue
        ok, text = fetch_text(url)
        cap = MAX_CHARS.get(src, 2000)
        parts.append(f"--- SOURCE: {src.upper()} URL: {url} OK: {ok} ---")
        parts.append(text[:cap])
    return "\n".join(parts)


def ask_region(client: OpenAI, region: str, dates: str, sources_blob: str) -> Optional[str]:
    # Single-region prompt to keep token usage low
    prompt = f"""
You are generating ONE REGION section for a weekend mountain forecast.

DATE:
{dates}

SOURCES (best-effort extracted text; may be incomplete):
{sources_blob}

TASK:
Return ONLY the {region} section in EXACTLY this 6-line format:

{region}
Rain: <current%> (<min%–max%>) chance <brief>
Wind: <current mph> (<min–max mph>) on tops <brief>
Temp: Valleys <current°C> (<min–max°C>); summits <current°C> (<min–max°C>) (feels like <current°C> (<min–max°C>))
Cloud: Cloud base ~<current m> (~<min–max m>), <brief>
Freezing level: ~<current m> (~<min–max m>), <brief>

RULES:
- Use the sources above ONLY. If a value isn't present, treat it as missing.
- Min/max are the LOWEST and HIGHEST values extracted from any INDIVIDUAL SOURCE that provides that metric.
- Choose the "current/best" value by prioritising MWIS then Met Office; if they disagree, choose the more conservative (worse) for safety.
- Windy/Mountain-Forecast: use only if clearly extractable; never invent.
- If only one source provides a value, use (value–value).
- If no sources provide a value, write: n/a (n/a–n/a)
- Keep concise, professional mountain forecast language.
- Units: mph, °C, m.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt if txt else None
    except Exception as e:
        print(f"Region generation failed for {region}: {e}")
        return None


def insert_last_updated(full_text: str, updated_utc: str, dates: str) -> str:
    lines = full_text.splitlines()
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("Confidence:"):
            out.append(f"Last updated: {updated_utc}")
            inserted = True
    if not inserted:
        # add after date line if missing
        if len(out) >= 2:
            out.insert(2, "Confidence: Moderate")
            out.insert(3, f"Last updated: {updated_utc}")
    return "\n".join(out)


def main() -> int:
    # failure-safe: don't overwrite index.html on failure
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing. Leaving existing index.html unchanged.")
        return 0

    client = OpenAI(api_key=api_key)
    dates = upcoming_weekend_dates()
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    regions = ["Peak District", "Snowdonia", "Brecon Beacons", "Lake District"]
    region_sections: List[str] = []

    for region in regions:
        print(f"Fetching + generating region: {region}")
        sources_blob = build_region_sources_blob(region)
        sec = ask_region(client, region, dates, sources_blob)
        if not sec:
            print(f"No output for {region}; leaving existing index.html unchanged.")
            return 0
        region_sections.append(sec)

    # Confidence is still a single line (kept simple + stable)
    # If you want, we can compute it from coverage later.
    full_text = "\n".join([
        "Weekend Mountain Forecast",
        dates,
        "Confidence: Moderate",
        "",  # blank line before regions
        "\n\n".join(region_sections),
    ]).strip()

    full_text = insert_last_updated(full_text, updated_utc, dates)

    with open("page_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace("{{CONTENT}}", full_text)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Forecast generated and index.html updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
