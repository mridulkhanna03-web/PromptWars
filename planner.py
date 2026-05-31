"""Pure planning logic for the Travel Planner Engine.

This module deliberately has NO Streamlit, network, or API dependencies, so every
function here is fully unit-testable in isolation. ``app.py`` imports these helpers
and wires them to the Gemini client and the Streamlit UI.
"""

import json
import unicodedata
from urllib.parse import quote

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# JSON schema enforced natively by Gemini (no manual code-fence stripping needed).
ITINERARY_SCHEMA = {
    "type": "object",
    "properties": {
        "trip_summary": {"type": "string"},
        "estimated_total_cost": {"type": "number"},
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "theme": {"type": "string"},
                    "activities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string"},
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "location": {"type": "string"},
                                "estimated_cost": {"type": "number"},
                                "duration_minutes": {"type": "integer"},
                                "lat": {"type": "number"},
                                "lng": {"type": "number"},
                            },
                            "required": [
                                "time", "title", "description",
                                "location", "estimated_cost", "duration_minutes",
                                "lat", "lng",
                            ],
                        },
                    },
                },
                "required": ["day", "theme", "activities"],
            },
        },
    },
    "required": ["trip_summary", "estimated_total_cost", "days"],
}

# Distinct, colour-blind-friendly RGB colours cycled per day on the map.
DAY_COLORS = [
    [228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
    [255, 127, 0], [166, 86, 40], [247, 129, 191], [153, 153, 153],
]

# Expected activities-per-day range for each pace.
PACE_RANGES = {"Relaxed": (2, 3), "Balanced": (4, 5), "Packed": (6, 99)}

# A realistic upper bound on planned activity time per day (~14 waking hours).
MAX_DAILY_MINUTES = 840


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def fold(text: str) -> str:
    """Lowercase and strip accents so "Park Guell" matches "Park Güell"."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )


def _split_terms(raw: str) -> list[str]:
    """Split a comma/newline separated free-text field into clean terms."""
    return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_initial_prompt(prefs: dict, realtime_context: str = "") -> str:
    """Build the initial trip-planning prompt, optionally grounded in real-time data."""
    interests_str = ", ".join(prefs["interests"]) if prefs["interests"] else "general sightseeing"
    dietary_str = ", ".join(prefs["dietary_needs"]) if prefs["dietary_needs"] else "no restrictions"
    must_see_str = prefs["must_see"] if prefs.get("must_see") else "none specified"
    avoid_str = prefs["avoid"] if prefs.get("avoid") else "nothing specified"

    context_block = ""
    if realtime_context.strip():
        context_block = f"""
Use the following REAL-TIME information (from live web search) to make the plan accurate
about weather, current prices, opening hours, and any closures or events:
\"\"\"
{realtime_context.strip()}
\"\"\"
"""

    return f"""You are an expert travel planner. Create a detailed, day-by-day itinerary based on:

Destination: {prefs['destination']}
Trip Duration: {prefs['num_days']} days
Total Budget: {prefs['currency']} {prefs['budget']}
Interests: {interests_str}
Travel Pace: {prefs['travel_pace']}
Dietary Needs: {dietary_str}
Must-See Places: {must_see_str}
Things to Avoid: {avoid_str}
{context_block}
Create a realistic, enjoyable itinerary that:
1. Stays within the total budget (the sum of all estimated_cost values must be <= the budget)
2. Respects the travel pace (Relaxed = 2-3 activities/day, Balanced = 4-5, Packed = 6+)
3. Matches the stated interests
4. Includes realistic, location-appropriate cost estimates for each activity
5. Includes every must-see place and excludes everything in the avoid list
6. Orders activities sensibly within each day (group nearby places, logical time flow)
7. Reflects the real-time information above when provided (weather-appropriate activities, current prices)

For every activity, set "lat" and "lng" to the real-world latitude/longitude of that location
(your best precise estimate for the named place; never 0).
Set estimated_total_cost to the exact sum of every activity's estimated_cost."""


def build_budget_revision_prompt(itinerary: dict, budget: float, currency: str) -> str:
    """Build the prompt that asks Gemini to revise an over-budget itinerary."""
    current_cost = itinerary.get("estimated_total_cost", 0)
    return f"""The following travel itinerary exceeds the budget and must be revised.

Current Cost: {currency} {current_cost:.2f}
Budget Limit: {currency} {budget:.2f}
Excess: {currency} {current_cost - budget:.2f}

Current itinerary:
{json.dumps(itinerary, indent=2)}

Revise this itinerary so the sum of all estimated_cost values is <= {currency} {budget:.2f}. You may:
1. Remove lower-priority activities
2. Swap activities for cheaper alternatives or shorten durations
3. Combine activities

Keep the same trip duration, main interests, and overall structure.
Set estimated_total_cost to the exact new sum."""


def build_disruption_prompt(itinerary: dict, disruption: str, prefs: dict, realtime_context: str = "") -> str:
    """Build the prompt that re-plans an itinerary around a disruption."""
    context_block = ""
    if realtime_context.strip():
        context_block = f"""
Real-time information relevant to this disruption (from live web search):
\"\"\"
{realtime_context.strip()}
\"\"\"
"""

    interests = ", ".join(prefs["interests"]) if prefs["interests"] else "general"
    return f"""You are re-planning a travel itinerary due to a real-time disruption.

Current itinerary:
{json.dumps(itinerary, indent=2)}

Disruption: {disruption}
{context_block}
Trip context:
- Destination: {prefs['destination']}
- Interests: {interests}
- Travel Pace: {prefs['travel_pace']}
- Budget: {prefs['currency']} {prefs['budget']}

Re-plan only the affected days/activities while keeping all unaffected days intact.
Use the real-time information above when deciding replacements (e.g. weather-appropriate alternatives).
Maintain the same trip duration and respect the original budget.
For every activity, set "lat" and "lng" to the real-world coordinates of that location (never 0).
Set estimated_total_cost to the exact new sum of all activity costs."""


def build_destination_research_query(prefs: dict) -> str:
    """Build the grounded-search query used to gather real-time destination intel."""
    return (
        f"Provide concise, current, practical travel intel for a trip to {prefs['destination']}. "
        f"Cover: typical weather this time of year, any major attraction closures or notable events "
        f"happening soon, and approximate current prices (in {prefs['currency']}) for top attractions, "
        f"local meals, and transport. Keep it factual and brief."
    )


def build_disruption_research_query(disruption: str, prefs: dict) -> str:
    """Build the grounded-search query used to gather real-time facts for a disruption."""
    return (
        f"For a trip in {prefs['destination']}, regarding this situation: \"{disruption}\". "
        f"Provide the current real-world facts needed to re-plan: actual current/forecast weather if "
        f"relevant, whether any named place is open today, and realistic nearby alternatives. "
        f"Keep it factual and brief."
    )


# ---------------------------------------------------------------------------
# Itinerary analysis
# ---------------------------------------------------------------------------

def needs_budget_correction(itinerary: dict, budget: float) -> bool:
    """True if the itinerary's estimated total cost exceeds the budget."""
    return itinerary.get("estimated_total_cost", 0) > budget


def build_map_rows(itinerary: dict) -> list[dict]:
    """Flatten an itinerary into mappable rows (one per activity with valid coords)."""
    rows = []
    for day in itinerary.get("days", []):
        day_num = day.get("day", 0)
        color = DAY_COLORS[(day_num - 1) % len(DAY_COLORS)]
        for activity in day.get("activities", []):
            lat, lng = activity.get("lat"), activity.get("lng")
            if isinstance(lat, (int, float)) and isinstance(lng, (int, float)) and (lat or lng):
                rows.append({
                    "lat": float(lat), "lng": float(lng),
                    "day": day_num, "title": activity.get("title", ""),
                    "location": activity.get("location", ""),
                    "time": activity.get("time", ""),
                    "color": color,
                })
    return rows


def validate_constraints(itinerary: dict, prefs: dict) -> list[dict]:
    """Programmatically check the itinerary against the user's constraints.

    Returns a list of ``{"label", "ok", "detail"}`` results (``ok`` True = satisfied).
    """
    results = []
    text = fold(json.dumps(itinerary, ensure_ascii=False))

    # Budget
    total = itinerary.get("estimated_total_cost", 0)
    results.append({
        "label": "Within budget",
        "ok": total <= prefs["budget"],
        "detail": f"{prefs['currency']} {total:.0f} of {prefs['currency']} {prefs['budget']:.0f}",
    })

    # Trip length
    n_days = len(itinerary.get("days", []))
    results.append({
        "label": "Correct trip length",
        "ok": n_days == prefs["num_days"],
        "detail": f"{n_days} of {prefs['num_days']} days planned",
    })

    # Must-see inclusion
    must = _split_terms(prefs.get("must_see", ""))
    if must:
        missing = [m for m in must if fold(m) not in text]
        results.append({
            "label": "Must-see places included",
            "ok": not missing,
            "detail": "All included" if not missing else f"Missing: {', '.join(missing)}",
        })

    # Avoid exclusion
    avoid = _split_terms(prefs.get("avoid", ""))
    if avoid:
        present = [a for a in avoid if fold(a) in text]
        results.append({
            "label": "Avoided items excluded",
            "ok": not present,
            "detail": "None present" if not present else f"Found: {', '.join(present)}",
        })

    # Per-day time feasibility
    over = []
    for day in itinerary.get("days", []):
        mins = sum(a.get("duration_minutes", 0) for a in day.get("activities", []))
        if mins > MAX_DAILY_MINUTES:
            over.append(f"Day {day.get('day')} ({mins // 60}h)")
    results.append({
        "label": "Days fit a realistic schedule",
        "ok": not over,
        "detail": "All days reasonable" if not over else f"Overpacked: {', '.join(over)}",
    })

    # Pace adherence
    lo, hi = PACE_RANGES.get(prefs["travel_pace"], (1, 99))
    counts = [len(d.get("activities", [])) for d in itinerary.get("days", [])]
    avg = (sum(counts) / len(counts)) if counts else 0
    results.append({
        "label": f"Matches '{prefs['travel_pace']}' pace",
        "ok": lo <= avg <= hi,
        "detail": f"{avg:.1f} activities/day (target {lo}–{hi if hi < 99 else '6+'})",
    })

    return results


def diff_itineraries(old: dict, new: dict) -> list[dict]:
    """Compare two itineraries day-by-day; return per-day added/removed activity titles."""
    def titles_by_day(itin):
        out = {}
        for day in itin.get("days", []):
            out[day.get("day")] = [a.get("title", "") for a in day.get("activities", [])]
        return out

    old_by, new_by = titles_by_day(old), titles_by_day(new)
    changes = []
    for day in sorted(set(old_by) | set(new_by)):
        before = old_by.get(day, [])
        after = new_by.get(day, [])
        added = [t for t in after if t not in before]
        removed = [t for t in before if t not in after]
        if added or removed:
            changes.append({"day": day, "added": added, "removed": removed})
    return changes


def cost_breakdown_by_day(itinerary: dict) -> list[dict]:
    """Return per-day total cost as ``[{"day", "cost"}]`` for charting/summaries."""
    return [
        {
            "day": day.get("day", 0),
            "cost": sum(a.get("estimated_cost", 0) for a in day.get("activities", [])),
        }
        for day in itinerary.get("days", [])
    ]


# ---------------------------------------------------------------------------
# Wikipedia image helpers (pure URL building + response parsing).
# The actual network fetch lives in app.py so this stays testable offline.
# ---------------------------------------------------------------------------

def wiki_search_url(query: str, lang: str = "en") -> str:
    """Build the Wikipedia search-API URL for a free-text query."""
    return (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&list=search"
        f"&srsearch={quote(query)}&format=json&srlimit=1"
    )


def wiki_summary_url(title: str, lang: str = "en") -> str:
    """Build the Wikipedia REST summary URL for a page title."""
    return f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"


def parse_wiki_search_title(search_json: dict) -> str | None:
    """Extract the best-matching page title from a Wikipedia search response."""
    try:
        hits = search_json["query"]["search"]
        return hits[0]["title"] if hits else None
    except (KeyError, IndexError, TypeError):
        return None


def parse_wiki_summary_image(summary_json: dict) -> tuple[str | None, str | None]:
    """Extract ``(image_url, alt_text)`` from a Wikipedia summary response.

    Returns ``(None, None)`` when no usable thumbnail is present.
    """
    if not isinstance(summary_json, dict):
        return None, None
    thumb = summary_json.get("thumbnail") or {}
    url = thumb.get("source")
    if not url:
        return None, None
    title = summary_json.get("title") or "location"
    return url, f"Photo of {title}"


# ---------------------------------------------------------------------------
# Openverse image helpers (keyless API that returns relevant photos for ANY
# query — landmarks, dishes, neighbourhoods). Network fetch lives in app.py.
# ---------------------------------------------------------------------------

def openverse_search_url(query: str, page_size: int = 1) -> str:
    """Build the Openverse image-search API URL for a free-text query."""
    return f"https://api.openverse.org/v1/images/?q={quote(query)}&page_size={page_size}&mature=false"


def parse_openverse_image(data: dict) -> tuple[str | None, str | None]:
    """Extract ``(thumbnail_url, alt_text)`` from an Openverse search response."""
    if not isinstance(data, dict):
        return None, None
    results = data.get("results") or []
    if not results:
        return None, None
    first = results[0]
    url = first.get("thumbnail") or first.get("url")
    if not url:
        return None, None
    title = first.get("title") or "location"
    return url, f"Photo: {title}"


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def gmaps_link(lat, lng) -> str:
    """Build a Google Maps link that opens the exact coordinates."""
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
