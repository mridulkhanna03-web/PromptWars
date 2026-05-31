# 🌍 Travel Planner Engine

An AI-powered travel planning application built with **Streamlit** and **Google Gemini API**. Generate structured, day-by-day itineraries and handle real-time disruptions with intelligent re-planning.

## ✨ Features

- **AI Itinerary Generation** — Uses Gemini to create detailed, personalized travel plans
- **Structured JSON Output** — Day-by-day activities with times, costs, and locations
- **Budget Validation** — Automatically adjusts itineraries to fit your budget
- **Real-Time Disruption Handling** — Re-plan affected parts when weather, closures, or delays occur
- **Multi-Interest Support** — Food, History, Nature, Nightlife, Shopping, Art, Adventure
- **Dietary Customization** — Vegetarian, Vegan, Halal, Gluten-Free options
- **Flexible Travel Pace** — Relaxed, Balanced, or Packed schedules

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PromptWars.git
   cd PromptWars
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 📖 Usage

### Sidebar Inputs
- **Destination**: Where you want to travel
- **Trip Duration**: Number of days (1-30)
- **Budget**: Total budget with currency (USD, EUR, GBP, INR)
- **Interests**: Select multiple interests (food, history, nature, etc.)
- **Travel Pace**: Relaxed (2-3 activities/day), Balanced (4-5), or Packed (6+)
- **Dietary Needs**: Specify any dietary restrictions
- **Must-See Places**: Specific attractions you don't want to miss
- **Things to Avoid**: Places or activities to skip

### Generate Itinerary
Click **"Generate Itinerary"** to create a personalized plan. The app will:
1. Call Gemini to generate a structured itinerary
2. Validate it against your budget
3. Automatically revise if costs exceed budget
4. Display a clean, expandable day-by-day view

### Handle Disruptions
When something changes (weather, attraction closed, flight delayed, budget cut):
1. Go to **"Inject a Real-Time Disruption"**
2. Select a preset event or type a custom disruption
3. Click **"Apply Disruption"**
4. Gemini re-plans only affected parts while keeping other constraints

## 🏗️ Architecture

- **Single-file app** (`app.py`) — Easy to understand and modify
- **Streamlit UI** — Responsive, no frontend framework needed
- **Gemini 1.5 Flash** — Fast, cost-effective LLM for structured JSON tasks
- **Minimal dependencies** — Just Streamlit, google-generativeai, python-dotenv
- **No database** — Fully stateless session-based app

## 📁 Project Structure

```
PromptWars/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── README.md          # This file
├── SETUP.md           # Detailed setup guide
└── .gitignore         # Git ignore rules
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with:
```
GEMINI_API_KEY=your_api_key_here
```

### Customization
- **Model**: Change `gemini-1.5-flash` in `app.py` to use different Gemini models
- **Temperature**: Adjust `temperature` parameter in `call_gemini()` for more/less creative responses
- **Max Tokens**: Modify `max_output_tokens` for longer/shorter responses

## 🧪 Example Usage

**Trip to Paris**
1. Destination: `Paris`
2. Days: `3`
3. Budget: `$1000 USD`
4. Interests: `Food`, `History`
5. Pace: `Balanced`

**Expected Output**
- Day 1: Eiffel Tower & Trocadéro, French Cafe Lunch, Local Bistro Dinner
- Day 2: Louvre Museum, Seine River Cruise, Montmartre
- Day 3: Notre-Dame, Latin Quarter, Champs-Élysées

**Apply Disruption**
- "Budget cut by 20%" → Itinerary revises to fit $800
- "Museum closed" → Day 2 re-planned with alternatives

## 🐛 Troubleshooting

**ModuleNotFoundError: google.generativeai**
```bash
pip install google-generativeai
```

**GEMINI_API_KEY not found**
- Verify `.env` file exists in project root
- Check API key format: should be a long alphanumeric string
- Restart Streamlit: `streamlit run app.py`

**Port 8501 already in use**
```bash
streamlit run app.py --server.port 8502
```

**Failed to parse itinerary**
- Rare occurrence; app has automatic retry logic
- If persistent, check Gemini API quota
- Try with a simpler destination/fewer days

## 📊 Performance

- **API Calls**: 1-3 per itinerary (initial + retry if needed + budget revision)
- **Response Time**: 5-15 seconds for itinerary generation
- **Token Usage**: ~1500-2000 tokens per itinerary
- **Cost**: ~$0.01-0.03 per itinerary with Gemini Flash pricing

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🎓 Educational Context

This is an educational project demonstrating:
- Agentic AI workflows
- JSON schema enforcement
- Constraint-based re-planning
- Streamlit web application development
- Google Gemini API integration

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API](https://ai.google.dev/)
- [Python Documentation](https://docs.python.org/)

## 👤 Author

Created as part of PromptWars competition.

---

**Ready to plan your next adventure?** 🌍✈️ Get your Gemini API key and start using the Travel Planner Engine today!
