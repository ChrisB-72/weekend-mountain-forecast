import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import List, Optional, Tuple

import requests
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TIMEOUT_SECS = 20

SOURCE_PRIORITY = ["mwis", "metoffice", "windy", "mountainforecast"]

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

def upcoming_weekend_dates() -> Tuple[str, str, str]:
    """Returns (display_dates, saturday_label, sunday_label)"""
    today = date.today()
    days_until_sat = (5 - today.weekday()) % 7
    sat = today + timedelta(days=days_until_sat)
    sun = sat + timedelta(days=1)

    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    display = f"{ordinal(sat.day)} / {ordinal(sun.day)} {sat.strftime('%B')}"
    sat_label = f"{sat.strftime('%A')} {ordinal(sat.day)} {sat.strftime('%B')}"
    sun_label = f"{sun.strftime('%A')} {ordinal(sun.day)} {sun.strftime('%B')}"
    return display, sat_label, sun_label

def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()

def fetch_text(url: str) -> Tuple[bool, str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WeekendMountainForecastBot/1.0; +https://github.com/)"}
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

def ask_region_day(client: OpenAI, region: str, day_name: str, dates_display: str, sources_blob: str) -> Optional[str]:
    prompt = f"""
You are generating ONE REGION section for ONE DAY of a weekend mountain forecast.

WEEKEND DATE RANGE (for context): {dates_display}
DAY TO REPORT: {day_name}

SOURCES (best-effort extracted text; may be incomplete):
{sources_blob}

TASK:
Return ONLY the {region} section for {day_name} in EXACTLY this 6-line format:

{region}
Rain: <current%> (<min%–max%>) chance <brief>
Wind: <current mph> (<min–max mph>) on tops <brief>
Temp: Valleys <current°C> (<min–max°C>); summits <current°C> (<min–max°C>) (feels like <current°C> (<min–max°C>))
Cloud: Cloud base ~<current m> (~<min–max m>), <brief>
Freezing level: ~<current m> (~<min–max m>), <brief>

RULES:
- Use the sources above ONLY. If a value isn't present for that day, treat it as missing.
- Min/max are the LOWEST and HIGHEST values extracted from any INDIVIDUAL SOURCE that provides that metric for that day.
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
        print(f"Region/day generation failed for {region} {day_name}: {e}")
        return None

def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing. Leaving existing index.html unchanged.")
        return 0

    client = OpenAI(api_key=api_key)

    dates_display, sat_label, sun_label = upcoming_weekend_dates()
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    regions = ["Peak District", "Snowdonia", "Brecon Beacons", "Lake District"]

    sat_sections: List[str] = []
    sun_sections: List[str] = []

    for region in regions:
        print(f"Fetching + generating: {region}")
        sources_blob = build_region_sources_blob(region)

        sat = ask_region_day(client, region, "Saturday", dates_display, sources_blob)
        sun = ask_region_day(client, region, "Sunday", dates_display, sources_blob)

        if not sat or not sun:
            print("Missing output for Sat/Sun; leaving existing index.html unchanged.")
            return 0

        sat_sections.append(sat)
        sun_sections.append(sun)

    header_text = "\n".join([
        "Weekend Mountain Forecast",
        dates_display,
        "Confidence: Moderate",
        f"Last updated: {updated_utc}",
    ])

    saturday_block = "\n\n".join(sat_sections).strip()
    sunday_block = "\n\n".join(sun_sections).strip()

    with open("page_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    html = (template
            .replace("{{HEADER}}", header_text)
            .replace("{{SATURDAY}}", saturday_block)
            .replace("{{SUNDAY}}", sunday_block))

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Forecast generated and index.html updated.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
