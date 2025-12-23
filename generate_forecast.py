import os
from datetime import date, timedelta, datetime, timezone
from openai import OpenAI

# FAILURE-SAFE BEHAVIOUR:
# - If the API call fails for any reason, the script will NOT overwrite index.html.
# - That means your site keeps the last successful forecast.

def upcoming_weekend_dates() -> str:
    """
    Returns: '27th / 28th December'
    Uses the next Sat/Sun relative to the runner's current date (UTC).
    """
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

def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing. Leaving existing index.html unchanged.")
        return 0  # do not fail the workflow; just keep the last good page

    client = OpenAI(api_key=api_key)

    dates = upcoming_weekend_dates()
    updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    prompt = f"""
Generate a Weekend Mountain Forecast for {dates}.

Use combined interpretation of:
- Met Office Mountain Forecast
- MWIS
- Windy
- Mountain-Forecast.com

OUTPUT FORMAT MUST BE EXACTLY:

Weekend Mountain Forecast
{dates}
Confidence: High / Moderate / Low

Peak District
Rain:
Wind:
Temp:
Cloud:
Freezing level:

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

RULES:
- Do not add any extra headings, bullets, or summaries.
- Exactly 5 lines per region, in the order shown.
- Rain line MUST start with a % chance (e.g. '30% chance ...').
- Wind line: summit wind speeds in mph; mention gusts if relevant.
- Temp line: include valley temps; summit temps MUST include 'feels like' (wind chill). Do NOT add feels-like for valleys.
- Cloud line MUST start with cloud base in meters (e.g. 'Cloud base ~700 m, ...').
- Freezing level line MUST start with meters (e.g. '~900 m, ...').
- Keep concise, professional mountain forecast language.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
        )
        forecast = (resp.choices[0].message.content or "").strip()
        if not forecast:
            print("Empty forecast returned. Leaving existing index.html unchanged.")
            return 0
    except Exception as e:
        print(f"Forecast generation failed: {e}. Leaving existing index.html unchanged.")
        return 0

    # Insert "Last updated" directly under the Confidence line (and nowhere else).
    lines = forecast.splitlines()
    out_lines = []
    inserted = False

    for line in lines:
        out_lines.append(line)
        if not inserted and line.startswith("Confidence:"):
            out_lines.append(f"Last updated: {updated_utc}")
            inserted = True

    # If no Confidence line appeared, add one after date line.
    if not inserted:
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

    final_text = "\n".join(out_lines)

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
