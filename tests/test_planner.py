"""Unit tests for the pure planning logic in planner.py.

These tests are fully offline: no Gemini API key, no network, no Streamlit.
Run with: pytest
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs():
    return {
        "destination": "Paris",
        "num_days": 2,
        "budget": 500.0,
        "currency": "USD",
        "interests": ["History", "Food"],
        "travel_pace": "Balanced",
        "dietary_needs": ["Vegetarian"],
        "must_see": "Eiffel Tower, Louvre",
        "avoid": "nightclubs",
    }


@pytest.fixture
def itinerary():
    return {
        "trip_summary": "A short Paris trip.",
        "estimated_total_cost": 300.0,
        "days": [
            {
                "day": 1,
                "theme": "Icons",
                "activities": [
                    {"time": "09:00", "title": "Eiffel Tower", "description": "Views",
                     "location": "Champ de Mars", "estimated_cost": 30, "duration_minutes": 120,
                     "lat": 48.8584, "lng": 2.2945},
                    {"time": "13:00", "title": "Louvre", "description": "Art",
                     "location": "Louvre", "estimated_cost": 20, "duration_minutes": 180,
                     "lat": 48.8606, "lng": 2.3376},
                ],
            },
            {
                "day": 2,
                "theme": "Bohemian",
                "activities": [
                    {"time": "10:00", "title": "Montmartre Walk", "description": "Stroll",
                     "location": "Montmartre", "estimated_cost": 0, "duration_minutes": 150,
                     "lat": 48.8867, "lng": 2.3431},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def test_fold_lowercases_and_strips_accents():
    assert planner.fold("Park Güell") == "park guell"
    assert planner.fold("CAFÉ") == "cafe"
    assert planner.fold("Málaga") == planner.fold("malaga")


def test_split_terms_handles_commas_and_newlines():
    assert planner._split_terms("a, b\nc") == ["a", "b", "c"]
    assert planner._split_terms("  ,  ") == []
    assert planner._split_terms("") == []


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def test_initial_prompt_includes_core_fields(prefs):
    p = planner.build_initial_prompt(prefs)
    assert "Paris" in p
    assert "USD 500" in p
    assert "History" in p and "Food" in p
    assert "Eiffel Tower" in p
    assert "nightclubs" in p


def test_initial_prompt_includes_realtime_only_when_provided(prefs):
    assert "REAL-TIME" not in planner.build_initial_prompt(prefs)
    grounded = planner.build_initial_prompt(prefs, "It will rain on Tuesday.")
    assert "REAL-TIME" in grounded
    assert "rain on Tuesday" in grounded


def test_budget_revision_prompt_mentions_costs(itinerary):
    p = planner.build_budget_revision_prompt(itinerary, 250.0, "USD")
    assert "USD 250" in p
    assert "300" in p  # current cost


def test_disruption_prompt_includes_disruption_and_context(itinerary, prefs):
    p = planner.build_disruption_prompt(itinerary, "Rain all day", prefs, "Heavy rain confirmed")
    assert "Rain all day" in p
    assert "Heavy rain confirmed" in p
    assert "Paris" in p


def test_research_queries_reference_destination(prefs):
    assert "Paris" in planner.build_destination_research_query(prefs)
    dq = planner.build_disruption_research_query("Flight delayed", prefs)
    assert "Paris" in dq and "Flight delayed" in dq


# ---------------------------------------------------------------------------
# Budget / map
# ---------------------------------------------------------------------------

def test_needs_budget_correction():
    assert planner.needs_budget_correction({"estimated_total_cost": 600}, 500) is True
    assert planner.needs_budget_correction({"estimated_total_cost": 500}, 500) is False
    assert planner.needs_budget_correction({"estimated_total_cost": 100}, 500) is False
    assert planner.needs_budget_correction({}, 500) is False


def test_build_map_rows_keeps_valid_coords(itinerary):
    rows = planner.build_map_rows(itinerary)
    assert len(rows) == 3  # all three activities have valid coords
    assert {r["day"] for r in rows} == {1, 2}
    assert rows[0]["color"] == planner.DAY_COLORS[0]
    assert rows[2]["color"] == planner.DAY_COLORS[1]  # day 2 -> second colour


def test_build_map_rows_drops_invalid_coords():
    itin = {"days": [{"day": 1, "activities": [
        {"title": "no coords", "lat": None, "lng": None},
        {"title": "zero", "lat": 0, "lng": 0},
        {"title": "ok", "lat": 1.0, "lng": 2.0},
    ]}]}
    rows = planner.build_map_rows(itin)
    assert len(rows) == 1
    assert rows[0]["title"] == "ok"


def test_build_map_rows_empty():
    assert planner.build_map_rows({}) == []
    assert planner.build_map_rows({"days": []}) == []


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def _result(results, label_part):
    return next(r for r in results if label_part in r["label"])


def test_validate_budget_pass_and_fail(itinerary, prefs):
    ok = _result(planner.validate_constraints(itinerary, prefs), "Within budget")
    assert ok["ok"] is True

    prefs["budget"] = 100.0
    fail = _result(planner.validate_constraints(itinerary, prefs), "Within budget")
    assert fail["ok"] is False


def test_validate_trip_length(itinerary, prefs):
    assert _result(planner.validate_constraints(itinerary, prefs), "trip length")["ok"] is True
    prefs["num_days"] = 5
    assert _result(planner.validate_constraints(itinerary, prefs), "trip length")["ok"] is False


def test_validate_must_see_present_and_missing(itinerary, prefs):
    assert _result(planner.validate_constraints(itinerary, prefs), "Must-see")["ok"] is True
    prefs["must_see"] = "Eiffel Tower, Versailles"
    res = _result(planner.validate_constraints(itinerary, prefs), "Must-see")
    assert res["ok"] is False
    assert "Versailles" in res["detail"]


def test_validate_must_see_is_accent_insensitive(prefs):
    itin = {"estimated_total_cost": 10, "days": [
        {"day": 1, "activities": [
            {"title": "Park Güell", "description": "", "location": "Barcelona",
             "estimated_cost": 10, "duration_minutes": 60}]}]}
    prefs.update({"num_days": 1, "must_see": "Park Guell", "avoid": ""})
    assert _result(planner.validate_constraints(itin, prefs), "Must-see")["ok"] is True


def test_validate_avoid_excluded_and_present(itinerary, prefs):
    assert _result(planner.validate_constraints(itinerary, prefs), "Avoided")["ok"] is True
    bad = dict(itinerary)
    bad_text_itin = json.loads(json.dumps(itinerary))
    bad_text_itin["days"][0]["activities"][0]["description"] = "Wild nightclubs tour"
    res = _result(planner.validate_constraints(bad_text_itin, prefs), "Avoided")
    assert res["ok"] is False


def test_validate_skips_optional_checks_when_blank(itinerary, prefs):
    prefs["must_see"] = ""
    prefs["avoid"] = ""
    labels = [r["label"] for r in planner.validate_constraints(itinerary, prefs)]
    assert not any("Must-see" in l for l in labels)
    assert not any("Avoided" in l for l in labels)


def test_validate_time_feasibility_flags_overpacked(prefs):
    itin = {"estimated_total_cost": 0, "days": [
        {"day": 1, "activities": [{"title": "x", "duration_minutes": 900, "estimated_cost": 0}]}]}
    prefs.update({"num_days": 1})
    res = _result(planner.validate_constraints(itin, prefs), "realistic schedule")
    assert res["ok"] is False
    assert "Day 1" in res["detail"]


def test_validate_pace_balanced(itinerary, prefs):
    # Day1 has 2, Day2 has 1 -> avg 1.5, below Balanced target 4-5 -> not ok
    res = _result(planner.validate_constraints(itinerary, prefs), "pace")
    assert res["ok"] is False


def test_validate_pace_relaxed_matches(itinerary, prefs):
    prefs["travel_pace"] = "Relaxed"  # target 2-3, avg 1.5 -> still below
    res = _result(planner.validate_constraints(itinerary, prefs), "pace")
    assert "Relaxed" in res["label"]


# ---------------------------------------------------------------------------
# Diff & cost breakdown
# ---------------------------------------------------------------------------

def test_diff_detects_added_and_removed(itinerary):
    new = json.loads(json.dumps(itinerary))
    # Replace Louvre with Musée d'Orsay on day 1
    new["days"][0]["activities"][1]["title"] = "Musee d'Orsay"
    changes = planner.diff_itineraries(itinerary, new)
    day1 = next(c for c in changes if c["day"] == 1)
    assert "Louvre" in day1["removed"]
    assert "Musee d'Orsay" in day1["added"]


def test_diff_no_changes(itinerary):
    same = json.loads(json.dumps(itinerary))
    assert planner.diff_itineraries(itinerary, same) == []


def test_cost_breakdown_by_day(itinerary):
    breakdown = planner.cost_breakdown_by_day(itinerary)
    assert breakdown == [{"day": 1, "cost": 50}, {"day": 2, "cost": 0}]


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def test_wiki_urls_encode_query():
    assert "Eiffel%20Tower" in planner.wiki_search_url("Eiffel Tower")
    assert planner.wiki_summary_url("Louvre").endswith("/summary/Louvre")
    assert "M%C3%BCnchen" in planner.wiki_summary_url("München")


def test_parse_wiki_search_title():
    good = {"query": {"search": [{"title": "Eiffel Tower"}]}}
    assert planner.parse_wiki_search_title(good) == "Eiffel Tower"
    assert planner.parse_wiki_search_title({"query": {"search": []}}) is None
    assert planner.parse_wiki_search_title({}) is None
    assert planner.parse_wiki_search_title(None) is None


def test_parse_wiki_summary_image():
    with_thumb = {"title": "Louvre", "thumbnail": {"source": "http://img/x.jpg"}}
    url, alt = planner.parse_wiki_summary_image(with_thumb)
    assert url == "http://img/x.jpg"
    assert alt == "Photo of Louvre"

    assert planner.parse_wiki_summary_image({"title": "x"}) == (None, None)
    assert planner.parse_wiki_summary_image(None) == (None, None)
    assert planner.parse_wiki_summary_image({"thumbnail": {}}) == (None, None)


def test_openverse_search_url_encodes_query():
    u = planner.openverse_search_url("Eiffel Tower", page_size=3)
    assert "q=Eiffel%20Tower" in u
    assert "page_size=3" in u
    assert "mature=false" in u


def test_parse_openverse_image():
    data = {"results": [{"title": "Eiffel Tower", "thumbnail": "http://t/1", "url": "http://u/1"}]}
    assert planner.parse_openverse_image(data) == ("http://t/1", "Photo: Eiffel Tower")
    # Falls back to url when thumbnail missing
    assert planner.parse_openverse_image({"results": [{"url": "http://u/2"}]}) == ("http://u/2", "Photo: location")
    assert planner.parse_openverse_image({"results": []}) == (None, None)
    assert planner.parse_openverse_image({}) == (None, None)
    assert planner.parse_openverse_image(None) == (None, None)


def test_gmaps_link():
    assert planner.gmaps_link(48.8584, 2.2945) == (
        "https://www.google.com/maps/search/?api=1&query=48.8584,2.2945"
    )


def test_clean_activity_title():
    assert planner.clean_activity_title("Eiffel Tower Ascent (Second Floor by Lift)") == "Eiffel Tower Ascent"
    assert planner.clean_activity_title("Lunch: Classic Parisian Sandwich") == "Classic Parisian Sandwich"
    assert planner.clean_activity_title("Stroll - Seine Riverbank") == "Seine Riverbank"
    assert planner.clean_activity_title("Louvre Museum") == "Louvre Museum"


def test_build_image_query_prefers_clean_venue_name():
    # Location is a clean venue name -> use it, add destination
    act = {"title": "Eiffel Tower Ascent (Second Floor by Lift)",
           "location": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris"}
    assert planner.build_image_query(act, "Paris") == "Champ de Mars Paris"


def test_build_image_query_falls_back_when_location_is_address():
    # Location's first segment has digits (street address) -> use cleaned title
    act = {"title": "Musée d'Orsay", "location": "1 Rue de la Légion d'Honneur, 75007 Paris"}
    assert planner.build_image_query(act, "Paris") == "Musée d'Orsay Paris"


def test_build_image_query_handles_vague_location():
    act = {"title": "Lunch: Local Bistro", "location": "Various"}
    assert planner.build_image_query(act, "Rome") == "Local Bistro Rome"


def test_build_image_query_does_not_duplicate_destination():
    act = {"title": "Tokyo Tower", "location": "Tokyo Tower, Tokyo"}
    q = planner.build_image_query(act, "Tokyo")
    assert q == "Tokyo Tower"  # destination already present, not appended twice


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_schema_requires_core_fields():
    assert planner.ITINERARY_SCHEMA["required"] == ["trip_summary", "estimated_total_cost", "days"]
    activity = (planner.ITINERARY_SCHEMA["properties"]["days"]["items"]
                ["properties"]["activities"]["items"])
    for field in ("time", "title", "location", "estimated_cost", "lat", "lng"):
        assert field in activity["required"]
