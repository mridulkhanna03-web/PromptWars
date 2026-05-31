import os
import json
import urllib.request

import streamlit as st
import pandas as pd
import pydeck as pdk
from google import genai
from google.genai import types
from dotenv import load_dotenv

import planner

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

st.set_page_config(
    page_title="Travel Planner Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not API_KEY:
    st.error("GEMINI_API_KEY not found. Set it in your .env file or Streamlit secrets, then refresh.")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Styling (also serves accessibility: high-contrast text, clear focus states)
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        /* Higher-contrast body text for readability (WCAG AA). */
        .stApp, .stMarkdown, p, li, span { color: #1a1a2e; }

        /* Animated gradient hero. */
        .hero {
            border-radius: 18px;
            padding: 2.6rem 2rem;
            margin-bottom: 1.2rem;
            background: linear-gradient(120deg, #1e3a8a, #6d28d9, #0e7490, #1e3a8a);
            background-size: 300% 300%;
            animation: heroShift 12s ease infinite;
            color: #ffffff;
            text-align: center;
        }
        .hero h1 { color: #ffffff; font-size: 2.4rem; margin: 0 0 .4rem 0; }
        .hero p { color: #f1f5ff; font-size: 1.1rem; margin: 0; }
        .hero .plane { font-size: 2.2rem; display: inline-block; animation: floaty 3s ease-in-out infinite; }
        @keyframes heroShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes floaty {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        /* Feature cards. */
        .feature-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
        .feature-card {
            flex: 1 1 220px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1.2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            animation: fadeUp .6s ease both;
        }
        .feature-card h3 { margin: .2rem 0 .4rem 0; font-size: 1.05rem; color: #1e293b; }
        .feature-card p { font-size: .92rem; color: #334155; margin: 0; }
        .feature-card .ico { font-size: 1.8rem; }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Activity image. */
        .act-img { width: 100%; height: 150px; object-fit: cover; border-radius: 10px; }
        .hero-img { width: 100%; max-height: 280px; object-fit: cover; border-radius: 14px; margin-bottom: .6rem; }

        /* Visible, strong focus outline for keyboard users. */
        button:focus, a:focus, input:focus, select:focus { outline: 3px solid #6d28d9 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "itinerary": None,
    "prefs": None,
    "correction_applied": False,
    "disruption_applied": False,
    "sources": [],
    "realtime_summary": "",
    "prev_itinerary": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Gemini API layer
# ---------------------------------------------------------------------------

def call_gemini_json(prompt: str, temperature: float = 0.8) -> dict:
    """Call Gemini with native structured output and return a parsed itinerary dict."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=planner.ITINERARY_SCHEMA,
                temperature=temperature,
                max_output_tokens=32768,
                # Disable "thinking": on Gemini 2.5 it consumes the output-token budget
                # and can truncate the JSON mid-stream. Off = complete JSON + faster.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as e:
        raise Exception(f"Gemini API error: {str(e)}")

    try:
        finish_reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError):
        finish_reason = None
    if finish_reason and "MAX_TOKENS" in str(finish_reason):
        raise ValueError("The itinerary was too long to return in full. Try fewer days or a less packed pace.")

    if not response.text:
        raise ValueError("Gemini returned an empty response. Please try again.")

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse Gemini response as JSON:\n\n{response.text}")


def grounded_research(query: str, temperature: float = 0.4) -> tuple[str, list[dict]]:
    """Run a Google Search-grounded query. Returns (summary_text, sources).

    Grounding can't be combined with response_schema on Gemini 2.5, so this is a
    separate 'research' step whose factual output feeds the structured planning step.
    Degrades gracefully (returns empty) if grounding is unavailable.
    """
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=temperature,
            ),
        )
    except Exception:
        return "", []

    text = response.text or ""
    sources: list[dict] = []
    try:
        gm = response.candidates[0].grounding_metadata
        if gm and gm.grounding_chunks:
            seen = set()
            for chunk in gm.grounding_chunks:
                web = getattr(chunk, "web", None)
                if web and web.uri and web.uri not in seen:
                    seen.add(web.uri)
                    sources.append({"title": web.title or web.uri, "uri": web.uri})
    except (AttributeError, IndexError):
        pass
    return text, sources


def revise_for_budget(itinerary: dict, budget: float, currency: str) -> dict:
    return call_gemini_json(planner.build_budget_revision_prompt(itinerary, budget, currency), temperature=0.7)


def apply_disruption(itinerary: dict, disruption: str, prefs: dict, realtime_context: str = "") -> dict:
    prompt = planner.build_disruption_prompt(itinerary, disruption, prefs, realtime_context)
    return call_gemini_json(prompt, temperature=0.7)


def research_destination(prefs: dict) -> tuple[str, list[dict]]:
    return grounded_research(planner.build_destination_research_query(prefs))


def research_disruption(disruption: str, prefs: dict) -> tuple[str, list[dict]]:
    return grounded_research(planner.build_disruption_research_query(disruption, prefs))


def generate_itinerary(prefs: dict, use_realtime: bool = True) -> tuple[dict, bool, str, list[dict]]:
    """Generate an itinerary with optional real-time grounding and budget validation.

    Returns (itinerary, correction_applied, realtime_summary, sources).
    """
    realtime_summary, sources = ("", [])
    if use_realtime:
        realtime_summary, sources = research_destination(prefs)

    itinerary = call_gemini_json(planner.build_initial_prompt(prefs, realtime_summary))

    correction_applied = False
    if planner.needs_budget_correction(itinerary, prefs["budget"]):
        itinerary = revise_for_budget(itinerary, prefs["budget"], prefs["currency"])
        correction_applied = True

    return itinerary, correction_applied, realtime_summary, sources


# ---------------------------------------------------------------------------
# Images (free Wikipedia API — no key; cached + graceful fallback)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_place_image(query: str, use_search: bool = False) -> tuple[str | None, str | None]:
    """Return (image_url, alt_text) for a place, or (None, None) if unavailable."""
    if not query or not query.strip():
        return None, None

    def _get_json(url):
        req = urllib.request.Request(url, headers={"User-Agent": "TravelPlannerEngine/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        title = query.strip()
        if use_search:
            search = _get_json(planner.wiki_search_url(query))
            title = planner.parse_wiki_search_title(search) or title
        summary = _get_json(planner.wiki_summary_url(title))
        return planner.parse_wiki_summary_image(summary)
    except Exception:
        return None, None


def get_day_image(day: dict) -> tuple[str | None, str | None]:
    """Find a representative photo for a day by trying its activities' landmarks."""
    for activity in day.get("activities", [])[:4]:
        for candidate in (activity.get("title", ""), activity.get("location", "")):
            url, alt = fetch_place_image(candidate)
            if url:
                return url, alt
    return None, None


def render_image(url: str, alt: str, css_class: str):
    """Render an external image with a guaranteed alt attribute (accessibility)."""
    safe_alt = (alt or "Travel photo").replace('"', "'")
    st.markdown(
        f'<img src="{url}" alt="{safe_alt}" class="{css_class}" loading="lazy" />',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_map(itinerary: dict):
    """Plot every activity on an interactive map, color-coded by day."""
    rows = planner.build_map_rows(itinerary)
    if not rows:
        st.caption("Map unavailable — no coordinates for this itinerary.")
        return

    df = pd.DataFrame(rows)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lng, lat]",
        get_fill_color="color",
        get_radius=120,
        radius_min_pixels=6,
        radius_max_pixels=18,
        pickable=True,
    )
    view = pdk.ViewState(latitude=df["lat"].mean(), longitude=df["lng"].mean(), zoom=11)
    tooltip = {"html": "<b>Day {day} · {time}</b><br/>{title}<br/>{location}"}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

    legend = "&nbsp;&nbsp;".join(
        f"<span style='color:rgb({c[0]},{c[1]},{c[2]});font-size:1.1em'>●</span> Day {d}"
        for d, c in sorted({r["day"]: r["color"] for r in rows}.items())
    )
    st.markdown(
        f"<div role='note' aria-label='Map legend' style='font-size:0.95em;color:#1a1a2e'>{legend}</div>",
        unsafe_allow_html=True,
    )


def render_constraint_panel(itinerary: dict, prefs: dict):
    results = planner.validate_constraints(itinerary, prefs)
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    header = f"✅ All {total} constraints satisfied" if passed == total else f"⚠️ {passed}/{total} constraints satisfied"
    with st.expander(header, expanded=(passed != total)):
        for r in results:
            icon = "✅" if r["ok"] else "⚠️"
            status = "OK" if r["ok"] else "Needs attention"
            st.markdown(f"{icon} **{r['label']}** ({status}) — {r['detail']}")


def render_cost_breakdown(itinerary: dict, prefs: dict):
    breakdown = planner.cost_breakdown_by_day(itinerary)
    if not breakdown:
        return
    df = pd.DataFrame(breakdown).set_index("day")
    df.index.name = "Day"
    df = df.rename(columns={"cost": f"Cost ({prefs['currency']})"})
    st.bar_chart(df, height=220)


def render_diff(old: dict, new: dict):
    changes = planner.diff_itineraries(old, new)
    old_cost = old.get("estimated_total_cost", 0)
    new_cost = new.get("estimated_total_cost", 0)
    delta = new_cost - old_cost

    with st.expander("🔀 What changed", expanded=True):
        sign = "+" if delta > 0 else ""
        st.markdown(f"**Total cost:** {old_cost:.0f} → {new_cost:.0f} ({sign}{delta:.0f})")
        if not changes:
            st.markdown("Activities unchanged (timing or details may have been adjusted).")
        for c in changes:
            st.markdown(f"**Day {c['day']}**")
            for t in c["removed"]:
                st.markdown(f"- :red[➖ Removed:] {t}")
            for t in c["added"]:
                st.markdown(f"- :green[➕ Added:] {t}")


def render_itinerary(itinerary: dict, prefs: dict):
    """Render the itinerary with high-contrast text and photos."""
    total_cost = itinerary.get("estimated_total_cost", 0)
    budget = prefs["budget"]
    cur = prefs["currency"]

    # Destination hero photo
    hero_url, hero_alt = fetch_place_image(prefs["destination"], use_search=True)
    if hero_url:
        render_image(hero_url, hero_alt or f"Photo of {prefs['destination']}", "hero-img")

    if total_cost <= budget:
        st.success(f"✅ Total Cost: {cur} {total_cost:.2f}  (Budget: {cur} {budget:.2f})")
    else:
        st.error(f"❌ Total Cost: {cur} {total_cost:.2f}  (Budget: {cur} {budget:.2f})")

    st.info(itinerary.get("trip_summary", ""))

    # Real-time grounding panel with citations
    if st.session_state.realtime_summary or st.session_state.sources:
        with st.expander("🌐 Real-time data used (Google Search grounding)", expanded=False):
            if st.session_state.realtime_summary:
                st.markdown(st.session_state.realtime_summary)
            if st.session_state.sources:
                st.markdown("**Sources:**")
                for s in st.session_state.sources:
                    st.markdown(f"- [{s['title']}]({s['uri']})")

    render_constraint_panel(itinerary, prefs)

    st.markdown("#### 💸 Cost by day")
    render_cost_breakdown(itinerary, prefs)

    st.markdown("#### 🗺️ Trip map")
    render_map(itinerary)

    st.markdown("#### 🗓️ Day-by-day plan")
    for day in itinerary.get("days", []):
        day_num = day.get("day", "?")
        theme = day.get("theme", "")
        with st.expander(f"Day {day_num} — {theme}", expanded=(day_num == 1)):
            img_col, list_col = st.columns([1, 2])
            with img_col:
                url, alt = get_day_image(day)
                if url:
                    render_image(url, alt or f"Photo from day {day_num}", "act-img")
                else:
                    st.markdown(
                        f"<div class='act-img' style='background:linear-gradient(135deg,#1e3a8a,#6d28d9);"
                        f"display:flex;align-items:center;justify-content:center;color:#fff;font-size:2rem' "
                        f"role='img' aria-label='Day {day_num}'>🏙️</div>",
                        unsafe_allow_html=True,
                    )
            with list_col:
                for activity in day.get("activities", []):
                    title = activity.get("title", "?")
                    time = activity.get("time", "?")
                    location = activity.get("location", "?")
                    cost = activity.get("estimated_cost", 0)
                    mins = activity.get("duration_minutes", 0)
                    st.markdown(f"**{title}**  ·  🕐 {time}")
                    st.markdown(activity.get("description", ""))
                    st.markdown(f"📍 {location}  ·  💰 {cur} {cost:.0f}  ·  ⏱️ {mins} min")
                    st.markdown("---")

    if st.session_state.correction_applied:
        st.info("ℹ️ Budget correction pass applied — costs were adjusted to fit your budget.")
    if st.session_state.disruption_applied:
        st.info("⚡ Disruption re-plan applied — the itinerary was updated for the disruption.")


def render_landing():
    """Animated, interactive welcome screen shown before any itinerary exists."""
    st.markdown(
        """
        <div class="hero" role="banner">
            <div class="plane">✈️</div>
            <h1>Travel Planner Engine</h1>
            <p>Your AI travel agent — grounded in real-time data, constraint-aware, and ready to re-plan on the fly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="feature-grid">
            <div class="feature-card"><div class="ico">🌐</div>
                <h3>Real-time grounding</h3>
                <p>Plans use live weather, prices, and closures from Google Search — with cited sources.</p></div>
            <div class="feature-card"><div class="ico">✅</div>
                <h3>Constraint-aware</h3>
                <p>Every plan is checked against your budget, must-sees, pace, and daily time limits.</p></div>
            <div class="feature-card"><div class="ico">⚡</div>
                <h3>Instant re-planning</h3>
                <p>Inject a disruption — rain, a closure, a budget cut — and watch it adapt with a clear diff.</p></div>
            <div class="feature-card"><div class="ico">🗺️</div>
                <h3>Map & photos</h3>
                <p>See your trip on an interactive map and preview destinations with real photos.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 🚀 Get started")
    st.markdown(
        "1. Fill in your trip preferences in the **sidebar** ➡️  \n"
        "2. Click **Generate Itinerary**  \n"
        "3. Or try a ready-made example below 👇"
    )


# ---------------------------------------------------------------------------
# Generation helper (shared by the sidebar button and the sample-trip button)
# ---------------------------------------------------------------------------

def run_generation(prefs: dict, use_realtime: bool):
    try:
        status_msg = "🌐 Researching real-time conditions, then planning..." if use_realtime else "🤔 Planning your trip..."
        with st.spinner(status_msg):
            itinerary, corrected, summary, sources = generate_itinerary(prefs, use_realtime)
            st.session_state.itinerary = itinerary
            st.session_state.prefs = prefs
            st.session_state.correction_applied = corrected
            st.session_state.disruption_applied = False
            st.session_state.prev_itinerary = None
            st.session_state.realtime_summary = summary
            st.session_state.sources = sources
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error generating itinerary: {str(e)}")


SAMPLE_PREFS = {
    "destination": "Paris",
    "num_days": 3,
    "budget": 1200.0,
    "currency": "USD",
    "interests": ["History", "Food", "Art"],
    "travel_pace": "Balanced",
    "dietary_needs": ["None"],
    "must_see": "Eiffel Tower, Louvre Museum",
    "avoid": "",
}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
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

    st.divider()
    use_realtime = st.toggle(
        "🌐 Use real-time data (Google Search)",
        value=True,
        help="Grounds the plan in live weather, current prices, opening hours, and closures — with cited sources.",
    )

    generate_btn = st.button("🚀 Generate Itinerary", key="gen_btn", use_container_width=True, type="primary")

# Sidebar generate action
if generate_btn:
    if not destination:
        st.error("Please enter a destination.")
    elif not interests:
        st.error("Please select at least one interest.")
    else:
        run_generation(
            {
                "destination": destination,
                "num_days": int(num_days),
                "budget": float(budget),
                "currency": currency,
                "interests": interests,
                "travel_pace": travel_pace,
                "dietary_needs": dietary_needs,
                "must_see": must_see,
                "avoid": avoid,
            },
            use_realtime,
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
if not st.session_state.itinerary:
    render_landing()
    if st.button("✨ Try a sample trip (Paris · 3 days)", use_container_width=True):
        run_generation(SAMPLE_PREFS, use_realtime)
else:
    prefs = st.session_state.prefs

    if st.session_state.disruption_applied and st.session_state.prev_itinerary:
        render_diff(st.session_state.prev_itinerary, st.session_state.itinerary)

    render_itinerary(st.session_state.itinerary, prefs)

    st.divider()
    st.subheader("⚡ Inject a real-time disruption")
    st.markdown("Something unexpected happened? Re-plan the affected parts instantly.")

    col1, col2 = st.columns([1, 1])
    with col1:
        preset_disruptions = [
            "None",
            "Rain forecast 2–5pm today",
            "Top attraction closed today",
            "Flight delayed to afternoon",
            "Budget cut by 20%",
        ]
        preset = st.selectbox("Simulated event", preset_disruptions, key="preset")
    with col2:
        custom = st.text_input("Or type a custom disruption", key="custom")

    disruption = custom if custom.strip() else (preset if preset != "None" else "")

    if st.button("Apply disruption", key="disrupt_btn", use_container_width=True):
        if disruption:
            try:
                spin = "🌐 Checking real-time conditions, then re-planning..." if use_realtime else "⚡ Re-planning affected parts..."
                with st.spinner(spin):
                    summary, sources = ("", [])
                    if use_realtime:
                        summary, sources = research_disruption(disruption, prefs)
                    st.session_state.prev_itinerary = st.session_state.itinerary
                    updated = apply_disruption(st.session_state.itinerary, disruption, prefs, summary)
                    st.session_state.itinerary = updated
                    st.session_state.disruption_applied = True
                    if use_realtime and (summary or sources):
                        st.session_state.realtime_summary = summary
                        st.session_state.sources = sources
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error applying disruption: {str(e)}")
        else:
            st.warning("Please select or enter a disruption.")

    if st.button("🔄 Start a new trip", use_container_width=True):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

st.caption(f"Powered by Google {MODEL} · Plans are AI estimates — verify hours and prices before you go.")
