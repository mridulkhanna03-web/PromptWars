# 🌍 Travel Planner Engine

An AI-powered, **agentic** travel planner built with **Streamlit** and the **Google Gemini API**. It turns your preferences and constraints into a structured, day-by-day itinerary — grounded in **real-time data** — and re-plans on the fly when disruptions hit.

**Live demo:** https://promptwars-sdw2bzh7egvq4vwr3rimat.streamlit.app/

## ✨ Features

- **🧠 Agentic research → plan pipeline** — A grounded research step (live weather, prices, opening hours, closures) feeds a structured planning step using `gemini-2.5-flash` with schema-enforced JSON.
- **🌐 Real-time grounding (Google Search)** — Itineraries and disruption re-plans are grounded in live web data, with **cited sources** shown in the app.
- **✅ Constraint validation** — Programmatic checks for budget, trip length, must-see inclusion, avoid exclusion, daily time feasibility, and pace.
- **🗺️ Interactive map** — Every activity plotted as a pin, color-coded by day, with hover tooltips and a legend.
- **🖼️ Photos** — A destination hero image and per-day photos via the free Wikipedia API (cached, with graceful fallback and alt text).
- **⚡ Instant re-planning + diff** — Inject a disruption (rain, closure, delay, budget cut) and see exactly what changed, with the cost delta.
- **💸 Cost breakdown** — Per-day cost chart.
- **🎨 Accessible, animated UI** — High-contrast text, alt text on images, semantic structure, keyboard focus styles, and an animated landing screen with a one-click sample trip.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Installation
```bash
git clone https://github.com/mridulkhanna03-web/PromptWars.git
cd PromptWars
pip install -r requirements.txt
cp .env.example .env          # then add your GEMINI_API_KEY
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## 🧪 Tests

The pure planning logic lives in [`planner.py`](planner.py) with **no Streamlit/API/network dependencies**, so it is fully unit-testable offline.

```bash
pip install -r requirements-dev.txt
pytest
```

27 tests cover prompt building, constraint validation (incl. accent-insensitive matching), the disruption diff, map-row extraction, cost breakdown, and Wikipedia helpers. They run automatically on every push via **GitHub Actions** ([.github/workflows/tests.yml](.github/workflows/tests.yml)) across Python 3.10–3.12.

## 🏗️ Architecture

```
PromptWars/
├── app.py                       # Streamlit UI + Gemini API + image fetching
├── planner.py                   # Pure logic: schema, prompts, validation, diff, map, image URLs
├── tests/test_planner.py        # Offline unit tests (pytest)
├── .github/workflows/tests.yml  # CI: run tests on push/PR
├── requirements.txt             # App dependencies
├── requirements-dev.txt         # App + pytest (CI)
├── pytest.ini                   # Test config
├── .env.example                 # API key template
└── README.md
```

- **Two-step agent** — Grounding can't combine with `response_schema` on Gemini 2.5, so a grounded *research* call gathers live facts, which then feed the structured *planning* call.
- **Reliability** — Thinking is disabled on structured calls (it otherwise consumes the output-token budget and truncates JSON); output limit is raised for long trips; a clear error is shown if a plan is still too long.

## 📖 Usage

1. Set your trip preferences in the sidebar (destination, days, budget, interests, pace, dietary needs, must-sees, things to avoid).
2. Toggle **🌐 real-time data** to ground the plan in live web results.
3. Click **Generate Itinerary** (or **Try a sample trip**).
4. Review the map, cost chart, constraint panel, and day-by-day plan with photos.
5. **Inject a disruption** to watch the agent re-plan, with a before/after diff.

## 🔧 Configuration

`.env` (or Streamlit Cloud **Secrets**):
```
GEMINI_API_KEY=your_api_key_here
```

## 📝 License

Open source under the MIT License.

---
Built for the PromptWars competition. Plans are AI estimates — verify hours and prices before you travel.
