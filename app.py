import os
import json
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def get_api_key() -> str | None:
    """Read the Gemini API key from env (.env locally) or Streamlit secrets (cloud)."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


API_KEY = get_api_key()

if not API_KEY:
    st.error("GEMINI_API_KEY not found. Set it in your .env file or Streamlit secrets, then refresh.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

st.set_page_config(page_title="Travel Planner Engine", layout="wide")

# Session state
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "prefs" not in st.session_state:
    st.session_state.prefs = None
if "correction_applied" not in st.session_state:
    st.session_state.correction_applied = False
if "disruption_applied" not in st.session_state:
    st.session_state.disruption_applied = False


# JSON schema enforced natively by Gemini (no manual parsing / code-fence stripping needed)
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
                            },
                            "required": [
                                "time", "title", "description",
                                "location", "estimated_cost", "duration_minutes",
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


def call_gemini_json(prompt: str, temperature: float = 0.8) -> dict:
    """Call Gemini with native structured output and return a parsed itinerary dict."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ITINERARY_SCHEMA,
                temperature=temperature,
                max_output_tokens=8192,
            ),
        )
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

    if not response.text:
        raise ValueError("Gemini returned an empty response. Please try again.")

    # response_schema guarantees valid JSON, but guard just in case.
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse Gemini response as JSON:\n\n{response.text}")


def build_initial_prompt(prefs: dict) -> str:
    """Build the initial trip planning prompt."""
    interests_str = ", ".join(prefs["interests"]) if prefs["interests"] else "general sightseeing"
    dietary_str = ", ".join(prefs["dietary_needs"]) if prefs["dietary_needs"] else "no restrictions"
    must_see_str = prefs["must_see"] if prefs["must_see"] else "none specified"
    avoid_str = prefs["avoid"] if prefs["avoid"] else "nothing specified"

    return f"""You are an expert travel planner. Create a detailed, day-by-day itinerary based on:

Destination: {prefs['destination']}
Trip Duration: {prefs['num_days']} days
Total Budget: {prefs['currency']} {prefs['budget']}
Interests: {interests_str}
Travel Pace: {prefs['travel_pace']}
Dietary Needs: {dietary_str}
Must-See Places: {must_see_str}
Things to Avoid: {avoid_str}

Create a realistic, enjoyable itinerary that:
1. Stays within the total budget (the sum of all estimated_cost values must be <= the budget)
2. Respects the travel pace (Relaxed = 2-3 activities/day, Balanced = 4-5, Packed = 6+)
3. Matches the stated interests
4. Includes realistic, location-appropriate cost estimates for each activity
5. Includes every must-see place and excludes everything in the avoid list
6. Orders activities sensibly within each day (group nearby places, logical time flow)

Set estimated_total_cost to the exact sum of every activity's estimated_cost."""


def revise_for_budget(itinerary: dict, budget: float, currency: str) -> dict:
    """Revise itinerary to fit within budget."""
    current_cost = itinerary.get("estimated_total_cost", 0)

    prompt = f"""The following travel itinerary exceeds the budget and must be revised.

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

    return call_gemini_json(prompt, temperature=0.7)


def apply_disruption(itinerary: dict, disruption: str, prefs: dict) -> dict:
    """Apply a disruption and re-plan affected parts."""
    prompt = f"""You are re-planning a travel itinerary due to a real-time disruption.

Current itinerary:
{json.dumps(itinerary, indent=2)}

Disruption: {disruption}

Trip context:
- Destination: {prefs['destination']}
- Interests: {', '.join(prefs['interests']) if prefs['interests'] else 'general'}
- Travel Pace: {prefs['travel_pace']}
- Budget: {prefs['currency']} {prefs['budget']}

Re-plan only the affected days/activities while keeping all unaffected days intact.
Maintain the same trip duration and respect the original budget.
Set estimated_total_cost to the exact new sum of all activity costs."""

    return call_gemini_json(prompt, temperature=0.7)


def generate_itinerary(prefs: dict) -> tuple[dict, bool]:
    """Generate itinerary with budget validation."""
    itinerary = call_gemini_json(build_initial_prompt(prefs))

    correction_applied = False
    if itinerary.get("estimated_total_cost", 0) > prefs["budget"]:
        itinerary = revise_for_budget(itinerary, prefs["budget"], prefs["currency"])
        correction_applied = True

    return itinerary, correction_applied


def render_itinerary(itinerary: dict, prefs: dict):
    """Render the itinerary in the UI."""
    total_cost = itinerary.get("estimated_total_cost", 0)
    budget = prefs["budget"]

    if total_cost <= budget:
        st.success(f"✅ Total Cost: {prefs['currency']} {total_cost:.2f} (Budget: {prefs['currency']} {budget:.2f})")
    else:
        st.error(f"❌ Total Cost: {prefs['currency']} {total_cost:.2f} (Budget: {prefs['currency']} {budget:.2f})")

    st.info(itinerary.get("trip_summary", ""))

    for day in itinerary.get("days", []):
        day_num = day.get("day", "?")
        theme = day.get("theme", "")
        with st.expander(f"📅 Day {day_num} — {theme}", expanded=False):
            activities = day.get("activities", [])
            if not activities:
                st.write("No activities scheduled.")
            else:
                for activity in activities:
                    cols = st.columns([1, 2, 1.5, 0.8, 0.8])
                    with cols[0]:
                        st.caption(f"🕐 {activity.get('time', '?')}")
                    with cols[1]:
                        st.write(f"**{activity.get('title', '?')}**")
                        st.caption(activity.get('description', ''))
                    with cols[2]:
                        st.caption(f"📍 {activity.get('location', '?')}")
                    with cols[3]:
                        st.caption(f"💰 {prefs['currency']} {activity.get('estimated_cost', 0):.0f}")
                    with cols[4]:
                        st.caption(f"⏱️ {activity.get('duration_minutes', 0)}m")

    if st.session_state.correction_applied:
        st.caption("ℹ️ Budget correction pass applied — costs were adjusted to fit budget")
    if st.session_state.disruption_applied:
        st.caption("⚡ Disruption re-plan applied — itinerary was updated for the disruption")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌍 Travel Planner Engine")
st.write("Let Gemini AI plan your perfect trip. Adjust constraints and handle real-time disruptions on the fly.")
st.caption(f"Powered by Google {MODEL}")

with st.sidebar:
    st.header("✈️ Trip Preferences")

    destination = st.text_input("Destination", placeholder="e.g., Paris, Tokyo, Bali")
    num_days = st.number_input("Number of Days", min_value=1, max_value=30, value=3)

    col1, col2 = st.columns([2, 1])
    with col1:
        budget = st.number_input("Total Budget", min_value=100, value=1000)
    with col2:
        currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR"])

    interests = st.multiselect(
        "Interests",
        ["Food", "History", "Nature", "Nightlife", "Shopping", "Art", "Adventure"],
        default=["History", "Food"],
    )

    travel_pace = st.radio("Travel Pace", ["Relaxed", "Balanced", "Packed"], index=1)

    dietary_needs = st.multiselect(
        "Dietary Needs",
        ["None", "Vegetarian", "Vegan", "Halal", "Gluten-Free"],
        default=["None"],
    )

    must_see = st.text_area("Must-See Places (optional)", placeholder="e.g., Eiffel Tower, Louvre Museum")
    avoid = st.text_area("Things to Avoid (optional)", placeholder="e.g., crowded areas, expensive restaurants")

    generate_btn = st.button("🚀 Generate Itinerary", key="gen_btn", use_container_width=True)

if generate_btn:
    if not destination:
        st.error("Please enter a destination")
    elif not interests:
        st.error("Please select at least one interest")
    else:
        prefs = {
            "destination": destination,
            "num_days": int(num_days),
            "budget": float(budget),
            "currency": currency,
            "interests": interests,
            "travel_pace": travel_pace,
            "dietary_needs": dietary_needs,
            "must_see": must_see,
            "avoid": avoid,
        }
        try:
            with st.spinner("🤔 Planning your trip..."):
                itinerary, corrected = generate_itinerary(prefs)
                # Snapshot the prefs that produced this itinerary so the rendered
                # cost summary and disruptions stay consistent if the sidebar changes.
                st.session_state.itinerary = itinerary
                st.session_state.prefs = prefs
                st.session_state.correction_applied = corrected
                st.session_state.disruption_applied = False
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating itinerary: {str(e)}")

# Render itinerary from the snapshot taken at generation time
if st.session_state.itinerary and st.session_state.prefs:
    prefs = st.session_state.prefs
    render_itinerary(st.session_state.itinerary, prefs)

    st.divider()
    st.subheader("⚡ Inject a Real-Time Disruption")
    st.write("Something unexpected happened? Re-plan affected parts instantly.")

    col1, col2 = st.columns([1, 1])
    with col1:
        preset_disruptions = [
            "None",
            "Rain forecast 2–5pm today",
            "Top attraction closed today",
            "Flight delayed to afternoon",
            "Budget cut by 20%",
        ]
        preset = st.selectbox("Simulated Event", preset_disruptions, key="preset")
    with col2:
        custom = st.text_input("Or type a custom disruption", key="custom")

    disruption = custom if custom.strip() else (preset if preset != "None" else "")

    if st.button("Apply Disruption", key="disrupt_btn", use_container_width=True):
        if disruption:
            try:
                with st.spinner("⚡ Re-planning affected parts..."):
                    updated = apply_disruption(st.session_state.itinerary, disruption, prefs)
                    st.session_state.itinerary = updated
                    st.session_state.disruption_applied = True
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error applying disruption: {str(e)}")
        else:
            st.warning("Please select or enter a disruption")
