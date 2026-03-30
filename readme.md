Project Title- Travel Forge- An AI Travel Planner
Project Overview-✦ Walk Through History ✦
Travel.py
 
A desktop heritage travel planner for India — offline, single-file, zero cloud dependency.
PYTHON	GUI	PDF	CITIES	API CALLS	LICENSE
3.x	Tkinter	ReportLab	18	0	MIT
 
What it does
 
Travel.py is a standalone desktop application that builds personalised day-by-day itineraries for Indian heritage destinations. Pick a city, choose how many days you have, set a budget, tick a few interests, and the app assembles a structured plan — three time slots per day, each with a monument name, a paragraph of context, and topic tags. No internet. No sign-up. No waiting.
When you're happy with the result you can export it as a formatted A4 PDF or as a self-contained HTML page that opens straight in your browser. Both exports look exactly like the app itself — dark palette, gold accents, the lot.
Features
 
■ 18 Indian heritage cities — Agra, Jaipur, Varanasi, Hampi, Khajuraho, and more
■ Trip lengths of 2, 3, 5, 7, or 10 days
■ Three budget tiers: Budget, Mid, Luxury
■ Interest tags: History, Architecture, Religion, Nature, Food, Photography
■ Travel pace options: Relaxed, Packed, Focused, Flexible
■ Traveller type: Solo, Couple, Family, Group
■ Three time slots per day — Morning, Afternoon, Evening
■ PDF export via ReportLab Platypus (A4, dark theme, paginated)
■ HTML share page via Python f-string (inline CSS, no external deps)
■ Animated loading sequence across 8 steps while the itinerary builds
■ Graceful fallback for cities without bespoke itinerary data
■ Cross-platform: Windows, macOS, Linux
Installation
 
You need Python 3 and one pip package. That's it.
# 1.  Clone or download this repo git clone https://github.com/your-username/travel-py.git cd travel-py # 2.  Install the one dependency pip install reportlab # 3.  Run python Travel.py
Linux only — if tkinter isn't available:
sudo apt install python3-tk
Quick Start
 
1.	Run python Travel.py — the window opens immediately.
2.	Select a destination from the dropdown (18 cities available).
3.	Choose a duration: 2, 3, 5, 7, or 10 days.
4.	Pick your budget tier and tick any interest tags that apply.
5.	Set a travel pace and traveller type.
6.	Click Generate AI Itinerary — wait about 3 seconds for the animation to finish.
7.	Read through your day-by-day plan in the right panel.
8.	Click Download PDF to save a formatted A4 document, or Share Link to get an HTML page.
Project Structure
 
travel-py/
■■■ Travel.py          # Entire application — 965 lines, single file
■■■ README.md          # This document
■■■ requirements.txt   # reportlab only
Architecture at a Glance
 
The app is one file split into three logical layers that communicate only via plain Python dicts — no global mutable state, no shared references between layers.
LAYER	WHAT IT CONTAINS
Data	DESTINATION_DATA dict — all 18 cities, monuments, itinerary text
Logic	generate_pdf(), generate_share_html(), default_days() — no UI code
Presentation	TravelPlannerApp — all Tkinter widgets, event handlers, layout
Adding a New City
 
No UI code needs to change. Open Travel.py and add a new entry to DESTINATION_DATA following the pattern below. The dropdown will pick it up automatically on the next run.
DESTINATION_DATA["mysuru"] = {     "name":      "Mysuru, Karnataka",
    "emoji":     "Palace",
    "tagline":   "The City of Palaces",     "monuments": ["Mysore Palace", "Chamundi Hills", "Brindavan Gardens"],
    "days": {
        1: {
            "title":     "Royal Mysuru",             "morning":   {"name": "Mysore Palace", "desc": "...", "tags": ["History", "
Arch"]},             "afternoon": {"name": "Chamundi Hills", "desc": "...", "tags": ["Religion"]
},             "evening":   {"name": "Brindavan Gardens", "desc": "...", "tags": ["Nature"
]},
        },
    },
}
If you only define one or two days, the app will ask for more days than you've defined. For a quick entry without full data, simply omit the 'days' key and the app will use default_days() as a fallback — no error, no crash.
Dependencies
 
PACKAGE	VERSION	PURPOSE
reportlab	latest	A4 PDF generation — the only pip install required
tkinter	built-in	Desktop GUI — ships with every Python 3 installation
threading	built-in	Background loading animation without freezing the UI
tempfile	built-in	Cross-platform temp directory for HTML share files
webbrowser	built-in	Opens PDFs and HTML in the OS default handler
datetime	built-in	Calendar dates for itinerary days and export timestamps
Known Issues and Limitations
 
■ Only 5 of the 18 cities have hand-written, detailed itinerary data. The rest use a two-day generic template.
■ Maximum trip length is 10 days. Longer trips would need additional day entries in
DESTINATION_DATA.
■ The destination dropdown has no search or filter — with 18 cities this is fine; at 50+ it would need one.
■ The ttk.Combobox dark-mode fix (the 'clam' theme workaround) doesn't look identical on every OS.
■ There are no unit tests. generate_pdf() and generate_share_html() are the obvious candidates for a test suite.
■ The app has no memory — closing it loses the current itinerary. A future version could auto-save to
JSON.
Roadmap
 
■ Connect to Claude or OpenAI API for dynamically generated, unique itineraries
■ Migrate DESTINATION_DATA to SQLite (sqlite3 ships in stdlib — no new dependency)
■ Map view of monuments using folium or the Google Maps Embed API
■ User accounts with cloud-synced saved itineraries
■ Packaging as a standalone .exe (Windows) and .app (macOS) via PyInstaller
■ Support for international heritage destinations beyond India
■ Java + SQL backend version exposing a REST API consumed by a web frontend
Contributing
 
Pull requests are welcome. The most useful contributions right now are additional city data entries (see Adding a New City above) and unit tests for the export functions. Please open an issue first to discuss any larger changes so nothing gets duplicated.
License
 
MIT — do what you like with it, just keep the attribution.
 

