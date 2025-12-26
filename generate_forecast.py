# generate_forecast.py
import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import Dict, List, Tuple, Optional

import requests
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TIMEOUT_SECS = 20

SOURCE_PRIORITY = ["mwis", "metoffice", "windy", "mountainforecast"]

MAX_CHARS = {
    "mwis": 9000,
    "metoffice": 9000,
    "windy": 2000,
    "mountainforecast": 2000,
}

# Areas + URLs (best-effort)
URLS = {
    "The Peak District": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/peak-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/peak-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Eryri": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/snowdonia-national-park",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/snowdonia",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "Bannau Brycheiniog": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/brecon-beacons",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/brecon-beacons",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
    "The Lake District": {
        "mwis": "https://www.mwis.org.uk/forecasts/english-and-welsh/lake-district",
        "metoffice": "https://weather.metoffice.gov.uk/specialist-forecasts/mountain/lake-district",
        "windy": "https://www.windy.com/",
        "mountainforecast": "https://www.mountain-forecast.com/",
    },
}

AREAS = list(URLS.keys())


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
    today = date.today()
    return [today, today + timedelta(days=1), today + timedelta(days=2)]


def day_title(d: date) -> str:
    def ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"
    return f"{d.strftime('%A')} {ordinal(d.day)} {d.strftime('%B')}"


def build_area_sources_blob(area: str) -> str:
    parts: List[str] = [f"=== AREA: {area} ==="]
    srcs = URLS[area]
    for src in SOURCE_PRIORITY:
        url = srcs.get(src)
        if not url:
            continue
        ok, text = fetch_text(url)
        cap = MAX_CHARS.get(src, 2000)
        parts.append(f"--- SOURCE: {src.upper()} URL: {url} OK: {ok} ---")
        parts.append(text[:cap])
    return "\n".join(parts)


def lines_to_cell_html(lines: List[str]) -> str:
    # Wrap each line in a no-wrap div; truncate with ellipsis if needed (CSS).
    safe = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # minimal HTML escaping
        ln = (ln.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
        safe.append(f'<span class="line">{ln}</span>')
    return '<div class="lines">' + "".join(safe) + "</div>"


def ask_area_for_day(client: OpenAI, area: str, target_day_label: str, sources_blob: str) -> Optional[List[str]]:
    """
    Returns exactly 7 lines (labels only, no area title):
    Rain:
    Valley wind:
    Hill wind:
    Valley temp:
    Hill temp:
    Cloud base:
    Freezing level:
    """
    prompt = f"""
You are producing ONE forecast cell for ONE area and ONE day.

AREA: {area}
DAY: {target_day_label}

SOURCES (best-effort extracted text; may be incomplete):
{sources_blob}

OUTPUT:
Return EXACTLY 7 lines, in this exact order (no extra lines):

Rain: ...
Valley wind: ...
Hill wind: ...
Valley temp: ...
Hill temp: ...
Cloud base: ...
Freezing level: ...

RULES:
- Use values you can clearly extract from sources; otherwise use "n/a".
- Min/max formatting:
  * If min = max, show only the single value (no range).
  * If min ≠ max, show ONLY the range and do NOT include any average/current value.
  * Always use "to" for ranges (never hyphens).
- Include wind direction on both wind lines (e.g. "W to SW 10 to 20 mph" or "W 15 mph").
- You may append very brief helpful text ONLY if it adds value:
  * must fit on ONE LINE (no commas/semicolons)
  * keep under ~25 characters
  * lowercase
  * no full stops
  Examples: "morning then dry", "gusty", "hill fog", "improving", "poor vis on tops"
- Prioritise MWIS and Met Office for interpretation; Windy/Mountain-Forecast only if clearly extractable.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        if not txt:
            return None
        lines = [l.rstrip() for l in txt.splitlines() if l.strip() != ""]
        if len(lines) != 7:
            # best-effort correction: truncate or pad with n/a
            lines = lines[:7]
            while len(lines) < 7:
                lines.append("n/a")
        return lines
    except Exception as e:
        print(f"Generation failed for {area} {target_day_label}: {e}")
        return None


def main() -> int:
    # Failure-safe: do not overwrite index.html on failures
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY missing. Leaving existing index.html unchanged.")
        return 0

    client = OpenAI(api_key=api_key)

    days = next_three_days()
    day_titles = [day_title(d) for d in days]
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Prefetch per area once
    area_sources: Dict[str, str] = {}
    for area in AREAS:
        print(f"Fetching sources for: {area}")
        area_sources[area] = build_area_sources_blob(area)

    # Build cells: 3 days + outlook
    cells: Dict[Tuple[str, str], str] = {}  # (area, slot) -> HTML
    # slots: day1, day2, day3, outlook
    for area in AREAS:
        src_blob = area_sources[area]

        # Day 1..3
        for idx, label in enumerate(day_titles, start=1):
            lines = ask_area_for_day(client, area, label, src_blob)
            if not lines:
                print("Missing output; leaving existing index.html unchanged.")
                return 0
            cells[(area, f"day{idx}")] = lines_to_cell_html(lines)

        # Outlook (broad trend) – still 7 lines, best-effort; may be n/a-heavy.
        outlook_lines = ask_area_for_day(client, area, "Outlook (days 4 to 7)", src_blob)
        if not outlook_lines:
            print("Missing outlook output; leaving existing index.html unchanged.")
            return 0
        cells[(area, "outlook")] = lines_to_cell_html(outlook_lines)

    # Header
    header = "\n".join([
        "Mountain Forecast",
        f"Next 3 days ({day_titles[0]} to {day_titles[-1]})",
        "Confidence: Moderate",
        f"Last updated: {updated_utc}",
    ])

    with open("page_template.html", "r", encoding="utf-8") as f:
        template = f.read()

    # Map areas to placeholders
    def get(area: str, slot: str) -> str:
        return cells[(area, slot)]

    html = (template
            .replace("{{HEADER}}", header)
            .replace("{{DAY1_TITLE}}", day_titles[0])
            .replace("{{DAY2_TITLE}}", day_titles[1])
            .replace("{{DAY3_TITLE}}", day_titles[2])

            .replace("{{PEAKS_DAY1}}", get("The Peak District", "day1"))
            .replace("{{PEAKS_DAY2}}", get("The Peak District", "day2"))
            .replace("{{PEAKS_DAY3}}", get("The Peak District", "day3"))
            .replace("{{PEAKS_OUTLOOK}}", get("The Peak District", "outlook"))

            .replace("{{ERYRI_DAY1}}", get("Eryri", "day1"))
            .replace("{{ERYRI_DAY2}}", get("Eryri", "day2"))
            .replace("{{ERYRI_DAY3}}", get("Eryri", "day3"))
            .replace("{{ERYRI_OUTLOOK}}", get("Eryri", "outlook"))

            .replace("{{BANNAU_DAY1}}", get("Bannau Brycheiniog", "day1"))
            .replace("{{BANNAU_DAY2}}", get("Bannau Brycheiniog", "day2"))
            .replace("{{BANNAU_DAY3}}", get("Bannau Brycheiniog", "day3"))
            .replace("{{BANNAU_OUTLOOK}}", get("Bannau Brycheiniog", "outlook"))

            .replace("{{LAKES_DAY1}}", get("The Lake District", "day1"))
            .replace("{{LAKES_DAY2}}", get("The Lake District", "day2"))
            .replace("{{LAKES_DAY3}}", get("The Lake District", "day3"))
            .replace("{{LAKES_OUTLOOK}}", get("The Lake District", "outlook"))
            )

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("Forecast generated and index.html updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
