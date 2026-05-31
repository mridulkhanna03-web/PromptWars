import os
import json
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file. Please set it and refresh.")
    st.stop()

genai.configure(api_key=API_KEY)

st.set_page_config(page_title="Travel Planner Engine", layout="wide")

if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "correction_applied" not in st.session_state:
    st.session_state.correction_applied = False
if "disruption_applied" not in st.session_state:
    st.session_state.disruption_applied = False


def build_initial_prompt(prefs: dict) -> str:
    """Build the initial trip planning prompt."""
    interests_str = ", ".join(prefs["interests"]) if prefs["interests"] else "general sightseeing"
    dietary_str = ", ".join(prefs["dietary_needs"]) if prefs["dietary_needs"] else "no restrictions"
    must_see_str = prefs["must_see"] if prefs["must_see"] else "none specified"
    avoid_str = prefs["avoid"] if prefs["avoid"] else "nothing specified"

    prompt = f"""You are an expert travel planner. Create a detailed, day-by-day itinerary based on:

Destination: {prefs['destination']}
Trip Duration: {prefs['num_days']} days
Total Budget: {prefs['currency']} {prefs['budget']}
Interests: {interests_str}
Travel Pace: {prefs['travel_pace']}
Dietary Needs: {dietary_str}
Must-See Places: {must_see_str}
Things to Avoid: {avoid_str}

Create a realistic, enjoyable itinerary that:
1. Fits within the total budget
2. Respects the travel pace (Relaxed = 2-3 activities/day, Balanced = 4-5, Packed = 6+)
3. Matches the interests
4. Includes realistic cost estimates for each activity
5. Provides a good mix of the requested interests

Return STRICT JSON only. No markdown. No prose. No code fences. Use this exact schema:

{{
  "trip_summary": "Brief 2-3 sentence overview of the trip",
  "estimated_total_cost": 0,
  "days": [
    {{
      "day": 1,
      "theme": "Day theme/title",
      "activities": [
        {{
          "time": "HH:MM-HH:MM or time description",
          "title": "Activity name",
          "description": "Brief description",
          "location": "Location name",
          "estimated_cost": 0,
          "duration_minutes": 0
        }}
      ]
    }}
  ]
}}"""
    return prompt


def parse_itinerary_json(text: str) -> dict | None:
    """Parse JSON response, stripping code fences and handling errors."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def call_gemini(prompt: str, temperature: float = 0.8) -> str:
    """Call Gemini API with given prompt."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 8192
            }
        )
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")


def revise_for_budget(itinerary: dict, budget: float, currency: str) -> dict:
    """Revise itinerary to fit within budget."""
    current_cost = itinerary.get("estimated_total_cost", 0)

    prompt = f"""The following travel itinerary was created but exceeds the budget:

Current Cost: {currency} {current_cost:.2f}
Budget Limit: {currency} {budget:.2f}
Excess: {currency} {current_cost - budget:.2f}

Here is the itinerary:
{json.dumps(itinerary, indent=2)}

Please revise this itinerary to fit within the {currency} {budget:.2f} budget. You can:
1. Remove lower-priority activities
2. Reduce the cost of activities (cheaper alternatives, shorter duration)
3. Combine activities
4. Adjust the itinerary while keeping the main interests and trip structure

Keep the JSON schema identical. Return STRICT JSON only. No markdown. No prose. No code fences."""

    response = call_gemini(prompt, temperature=0.7)
    parsed = parse_itinerary_json(response)

    if not parsed:
        raise ValueError("Failed to parse revised itinerary")

    return parsed


def apply_disruption(itinerary: dict, disruption: str, prefs: dict) -> dict:
    """Apply a disruption and re-plan affected parts."""
    prompt = f"""You are replanning a travel itinerary due to a real-time disruption.

Current Itinerary:
{json.dumps(itinerary, indent=2)}

Disruption: {disruption}

Trip Context:
- Destination: {prefs['destination']}
- Interests: {', '.join(prefs['interests']) if prefs['interests'] else 'general'}
- Travel Pace: {prefs['travel_pace']}
- Budget: {prefs['currency']} {prefs['budget']}

Re-plan only the affected days/activities while keeping all other constraints and unaffected days intact.
Maintain the same trip duration and overall structure.

Return the full updated itinerary with the same JSON schema. Return STRICT JSON only. No markdown. No prose. No code fences."""

    response = call_gemini(prompt, temperature=0.7)
    parsed = parse_itinerary_json(response)

    if not parsed:
        raise ValueError("Failed to parse re-planned itinerary")

    return parsed


def generate_itinerary(prefs: dict) -> tuple[dict, bool]:
    """Generate itinerary with budget validation and retry logic."""
    prompt = build_initial_prompt(prefs)

    response = call_gemini(prompt)
    itinerary = parse_itinerary_json(response)

    if not itinerary:
        # Retry with fix prompt
        retry_prompt = f"""The following response was not valid JSON:

{response}

Please provide the same travel itinerary response as STRICT JSON only. No markdown, prose, or code fences. Follow this schema exactly:

{{
  "trip_summary": "string",
  "estimated_total_cost": number,
  "days": [
    {{
      "day": number,
      "theme": "string",
      "activities": [
        {{
          "time": "string",
          "title": "string",
          "description": "string",
          "location": "string",
          "estimated_cost": number,
          "duration_minutes": number
        }}
      ]
    }}
  ]
}}"""
        response = call_gemini(retry_prompt)
        itinerary = parse_itinerary_json(response)

    if not itinerary:
        raise ValueError(f"Failed to parse itinerary after retry. Raw response:\n\n{response}")

    # Budget check and correction
    correction_applied = False
    estimated_cost = itinerary.get("estimated_total_cost", 0)

    if estimated_cost > prefs["budget"]:
        itinerary = revise_for_budget(itinerary, prefs["budget"], prefs["currency"])
        correction_applied = True

    return itinerary, correction_applied


def render_itinerary(itinerary: dict, prefs: dict):
    """Render the itinerary in the UI."""
    # Cost summary banner
    total_cost = itinerary.get("estimated_total_cost", 0)
    budget = prefs["budget"]
    cost_ok = total_cost <= budget

    if cost_ok:
        st.success(f"✅ Total Cost: {prefs['currency']} {total_cost:.2f} (Budget: {prefs['currency']} {budget:.2f})")
    else:
        st.error(f"❌ Total Cost: {prefs['currency']} {total_cost:.2f} (Budget: {prefs['currency']} {budget:.2f})")

    # Trip summary
    st.info(itinerary.get("trip_summary", ""))

    # Per-day expanders
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
                        cost = activity.get('estimated_cost', 0)
                        st.caption(f"💰 {prefs['currency']} {cost:.0f}")
                    with cols[4]:
                        duration = activity.get('duration_minutes', 0)
                        st.caption(f"⏱️ {duration}m")

    # Correction and disruption flags
    if st.session_state.correction_applied:
        st.caption("ℹ️ Budget correction pass applied — costs were adjusted to fit budget")
    if st.session_state.disruption_applied:
        st.caption("⚡ Disruption re-plan applied — itinerary was updated for the disruption")


# Main App UI
st.title("🌍 Travel Planner Engine")
st.write("Let Gemini AI plan your perfect trip. Adjust constraints and handle real-time disruptions on the fly.")

# Sidebar inputs
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
        default=["History", "Food"]
    )

    travel_pace = st.radio("Travel Pace", ["Relaxed", "Balanced", "Packed"], index=1)

    dietary_needs = st.multiselect(
        "Dietary Needs",
        ["None", "Vegetarian", "Vegan", "Halal", "Gluten-Free"],
        default=["None"]
    )

    must_see = st.text_area("Must-See Places (optional)", placeholder="e.g., Eiffel Tower, Louvre Museum")
    avoid = st.text_area("Things to Avoid (optional)", placeholder="e.g., crowded areas, expensive restaurants")

    generate_btn = st.button("🚀 Generate Itinerary", key="gen_btn", use_container_width=True)

# Validate inputs
if generate_btn:
    if not destination:
        st.error("Please enter a destination")
    elif not interests:
        st.error("Please select at least one interest")
    else:
        # Prepare preferences dict
        prefs = {
            "destination": destination,
            "num_days": int(num_days),
            "budget": float(budget),
            "currency": currency,
            "interests": interests,
            "travel_pace": travel_pace,
            "dietary_needs": dietary_needs,
            "must_see": must_see,
            "avoid": avoid
        }

        # Generate itinerary
        try:
            with st.spinner("🤔 Planning your trip..."):
                itinerary, corrected = generate_itinerary(prefs)
                st.session_state.itinerary = itinerary
                st.session_state.correction_applied = corrected
                st.session_state.disruption_applied = False
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating itinerary: {str(e)}")

# Render itinerary if it exists
if st.session_state.itinerary:
    render_itinerary(st.session_state.itinerary, {
        "destination": destination,
        "num_days": int(num_days),
        "budget": float(budget),
        "currency": currency,
        "interests": interests,
        "travel_pace": travel_pace,
        "dietary_needs": dietary_needs,
        "must_see": must_see,
        "avoid": avoid
    })

    # Disruption handler
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
            "Budget cut by 20%"
        ]
        preset = st.selectbox("Simulated Event", preset_disruptions, key="preset")

    with col2:
        custom = st.text_input("Or type a custom disruption", key="custom")

    disruption = custom if custom.strip() else (preset if preset != "None" else "")

    if st.button("Apply Disruption", key="disrupt_btn", use_container_width=True):
        if disruption:
            try:
                with st.spinner("⚡ Re-planning affected parts..."):
                    prefs = {
                        "destination": destination,
                        "num_days": int(num_days),
                        "budget": float(budget),
                        "currency": currency,
                        "interests": interests,
                        "travel_pace": travel_pace,
                        "dietary_needs": dietary_needs,
                        "must_see": must_see,
                        "avoid": avoid
                    }
                    updated = apply_disruption(st.session_state.itinerary, disruption, prefs)
                    st.session_state.itinerary = updated
                    st.session_state.disruption_applied = True
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error applying disruption: {str(e)}")
        else:
            st.warning("Please select or enter a disruption")
