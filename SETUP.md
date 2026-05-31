# Travel Planner Engine — Setup Guide

## ✅ Installation Complete

Your Travel Planner Engine is ready to run! Here's what was created:

- **app.py** — The complete Streamlit application
- **requirements.txt** — All Python dependencies (already installed)
- **.env** — Environment file (from template)
- **.env.example** — Template showing required variables

## 🚀 Getting Started

### Step 1: Add Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key and open `.env` in this directory
4. Replace `your_api_key_here` with your actual API key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

### Step 2: Run the App

Open a terminal in this directory and run:

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## 📋 Features

### Trip Planning Input (Sidebar)
- **Destination**: Where you want to go
- **Number of Days**: 1–30 days
- **Budget**: Total budget with currency (USD, EUR, GBP, INR)
- **Interests**: Select multiple (Food, History, Nature, Nightlife, Shopping, Art, Adventure)
- **Travel Pace**: Relaxed, Balanced, or Packed
- **Dietary Needs**: Special dietary requirements
- **Must-See Places**: Specific attractions (optional)
- **Things to Avoid**: Places/activities to skip (optional)

### Itinerary Generation
Click **"Generate Itinerary"** and Gemini will create a structured, day-by-day plan with:
- Daily themes and activities
- Estimated costs for each activity
- Duration estimates
- Locations

### Budget Validation
If the itinerary exceeds your budget, the app automatically revises it to fit within constraints.

### Real-Time Disruption Handling ⚡
When something changes (weather, closure, flight delay, budget cut), use the disruption handler to:
- Select a **preset disruption** (rain, attraction closed, flight delayed, budget cut)
- Or **type a custom disruption** (e.g., "Museum closed on Day 2")

Gemini will re-plan only the affected parts while keeping everything else intact.

## 🧪 Test Run

Try this to verify everything works:

1. **Destination**: Paris
2. **Days**: 3
3. **Budget**: $1000 USD
4. **Interests**: Food, History
5. **Travel Pace**: Balanced
6. Click **"Generate Itinerary"**
7. After getting results, try a disruption: "Budget cut by 20%"

## 📝 Notes

- The app uses **Gemini 1.5 Flash** (fast and cost-effective)
- All responses are returned as **strict JSON** for reliable parsing
- Budget corrections are automatic
- Disruption re-planning respects all original constraints
- The app is fully self-contained (no databases, no external APIs except Gemini)

## 🐛 Troubleshooting

**"GEMINI_API_KEY not found"**
- Make sure you've added your API key to the `.env` file
- Check that the .env file is in the same directory as `app.py`

**"Failed to parse itinerary"**
- This is rare, but if it happens, try again — the retry logic will engage
- If persistent, check your API quota

**Port 8501 already in use**
- Run: `streamlit run app.py --server.port 8502`

## 📚 Code Structure

Everything is in `app.py`:
- **Helper functions**: Prompt building, JSON parsing, Gemini calls, disruption handling
- **UI components**: Sidebar inputs, itinerary rendering, disruption handler
- **Main flow**: Input validation → Generation → Rendering → Disruption loop

No external modules — just Streamlit, Gemini SDK, and python-dotenv.
