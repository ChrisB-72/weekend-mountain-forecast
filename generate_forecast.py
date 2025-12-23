import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import Dict, List, Tuple

import requests
from openai import OpenAI


# ----------------------------
# Configuration
# ----------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TIMEOUT_SECS = 20

# Prioritise MWIS and Met Office when selecting the "current/best" value.
SOURCE_PRIORITY = ["mwis", "metoffice", "windy", "mountainforecast"]


# URLs to fetch (best-effort). You can tweak these if you prefer different sub-areas.
URLS = {
    "Peak District": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/peak-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/peak-district",
        # Windy and mountain-forecast are attempted; they may block scraping or vary by page structure.
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


# ----------------------------
# Helpers
# ----------------------------
def upcoming_weekend_dates() -> str:
    """Returns: '27th / 28th December' for the next Sat/Sun relative to today (UTC)."""
    today = date.today()
    days_until_sat = (5 - today.weekday()) % 7  # Sat = 5
    saturday = today + timedelta(days=days_until_sat)
    sunday = saturday + timedelta(days=1)

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    return f"{ordinal(saturday.day)} / {ordinal(sunday.day)} {saturday.strftime('%B')}"


def fetch_url(url: str) -> Tuple[bool, str]:
    """Best-effort fetch: returns (ok, text)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WeekendMountainForecastBot/1.0; +https://github.com/)"
        }
        r = requests.get(url, headers=headers, timeout=TIMEOUT_SECS)
        if r.status_code >= 400:
            return False, f"[HTTP {r.status_code}] {url}"
        # Keep text compact-ish: strip excessive whitespace to reduce token usage
        text = re.sub(r"\s+", " ", r.text)
        return True, text
    except Exception as e:
        return False, f"[FETCH ERROR: {e}] {url}"


def build_sources_blob() -> str:
    """
    Fetch all sources for all regions and return a single text blob for the model.
    We include short metadata so the model can compute per-source min/max.
    """
    parts: List[str] = []
    for region, sources in URLS.items():
        parts.append(f"=== REGION: {region} ===")
        for src_name in SOURCE_PRIORITY:
            url = sources.get(src_name)
            if not url:
                continue
            ok, text = fetch_url(url)
            parts.append(f"--- SOURCE: {src_name.upper()} URL: {url} OK: {ok} ---")
            # Hard cap per source so we don't explode tokens; MWIS/MetOffice usually contain the key info early.
            parts.append(text[:12000])
    return "\n".join(parts)


def insert_last_updated(forecast_text: str, updated_utc: str, dates: str) -> str:
    """Insert Last updated directly under Confidence line; if missing, add Confidence + Last updated after dates."""
    lines = forecast_text.splitlines()
    out_lines = []
    inserted = False

    for line in lines:
        out_lines.append(line)
        if not inserted and line.startswith("Confidence:"):
            out_lines.append(f"Last updated: {updated_utc}")
            inserted = True

    if not inserted:
        # Add after the date line (2nd line).
        if len(out_lines) >= 2:
            out_lines.insert(2, "Confidence: Moderate")
            out_lines.insert(3, f"Last updated: {updated_utc}")
        else:
            out_lines = [
                "Weekend Mountain Forecast",
                dates,
                "Confidence: Moderate",
                f"Last updated: {updated_utc}",
                "",
            ] + out_lines

    return "\n".join(out_lines)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    # FAILURE-SAFE:
    # If anything fails, we do NOT overwrite index.html.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing. Leaving existing index.html unchanged.")
        return 0

    dates = upcoming_weekend_dates()
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("Fetching source pages (best-effort)...")
    sources_blob = build_sources_blob()

    prompt = f"""
You are generating a weekend mountain forecast in a STRICT fixed format.

DATE:
{dates}

SOURCES (best-effort fetched HTML/text; some may be incomplete or blocked):
{sources_blob}

REQUIREMENTS:
- Use the sources above ONLY. If a value isn't present in the sources, treat it as missing.
- Compute min/max ranges using the LOWEST and HIGHEST value you can extract from any INDIVIDUAL SOURCE.
- Prioritise MWIS and Met Office when choosing the "current/best" value. If MWIS+MetOffice disagree, pick the more conservative (worse) for safety.
- Windy and Mountain-Forecast are optional: use their values only if clearly extractable; never invent.
- If only one source provides a value, show (value–value) for the range.
- If no sources provide a value, write "n/a (n/a–n/a)" for that metric.

OUTPUT FORMAT MUST BE EXACTLY (no extra headings, bullets, or summaries):

Weekend Mountain Forecast
{dates}
Confidence: High / Moderate / Low

Peak District
Rain: <current%> (<min%–max%>) chance <brief>
Wind: <current mph> (<min–max mph>) on tops <brief>
Temp: Valleys <current°C> (<min–max°C>); summits <current°C> (<min–max°C>) (feels like <current°C> (<min–max°C>))
Cloud: Cloud base ~<current m> (~<min–max m>), <brief>
Freezing level: ~<current m> (~<min–max m>), <brief>

Snowdonia
Rain:
Wind:
Temp:
Cloud:
Freezing level:

Brecon Beacons
Rain:
Wind:
Temp:
Cloud:
Freezing level:

Lake District
Rain:
Wind:
Temp:
Cloud:
Freezing level:

STYLE RULES:
- Keep concise, professional mountain forecast language.
- Keep units consistent: mph, °C, m.
- Rain line: must start with a % and include (min–max).
- Cloud line: must start with "Cloud base ~" and include (~min–max).
- Freezing level: must start with "~" and include (~min–max).
"""

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        forecast = (resp.choices[0].message.content or "").strip()
        if not forecast:
            print("Empty forecast returned. Leaving existing index.html unchanged.")
            return 0
    except Exception as e:
        print(f"Forecast generation failed: {e}. Leaving existing index.html unchanged.")
        return 0

    # Insert Last updated directly under Confidence
    final_text = insert_last_updated(forecast, updated_utc, dates)

    # Build index.html from template
    with open("page_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    html = template.replace("{{CONTENT}}", final_text)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Forecast generated and index.html updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
