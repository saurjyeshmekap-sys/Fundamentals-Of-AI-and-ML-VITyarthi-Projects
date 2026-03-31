import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, time, os, webbrowser, tempfile
from datetime import datetime, timedelta

# ── ML / NLP IMPORTS ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import re

# ── reportlab imports ────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable, KeepTogether)

# ── COLOUR PALETTE ───────────────────────────────────────────────────────────
BG      = "#0d1117"
BG2     = "#13181f"
BG3     = "#1a2130"
CARD    = "#161c26"
GOLD    = "#d4a843"
GOLD2   = "#f0c96a"
TEXT    = "#e8e0d4"
TEXT2   = "#9a9080"
TEXT3   = "#6a6055"
BORDER  = "#2a2e38"
MORNING = "#fcc419"
EVENING = "#818cf8"

# PDF colours
PDF_GOLD  = colors.HexColor("#d4a843")
PDF_DARK  = colors.HexColor("#0d1117")
PDF_CARD  = colors.HexColor("#1a2130")
PDF_TEXT  = colors.HexColor("#e8e0d4")
PDF_TEXT2 = colors.HexColor("#9a9080")
PDF_MORN  = colors.HexColor("#fcc419")
PDF_EVE   = colors.HexColor("#818cf8")

# ════════════════════════════════════════════════════════════════════════════
#  ML ENGINE 1 — KNN RECOMMENDATION SYSTEM
#  Maps each city to a feature vector over 10 interest dimensions.
#  When the user clicks Generate, their interest selections are fed into
#  a trained KNN model (k=3) to find the closest matching cities.
#  The chosen city from the dropdown is RERANKED: if it appears in the
#  top-3 KNN results it is confirmed; otherwise the best KNN match is
#  suggested to the user via a non-blocking info panel.
# ════════════════════════════════════════════════════════════════════════════

# Feature order matches INTERESTS list exactly:
# History, Architecture, Religion, Nature, Food,
# Photography, Culture, Adventure, Shopping, Nightlife
CITY_INTEREST_MATRIX = {
    "agra":       [5, 5, 3, 2, 3, 5, 4, 1, 3, 1],
    "delhi":      [5, 4, 4, 2, 5, 3, 5, 2, 5, 4],
    "varanasi":   [4, 3, 5, 3, 4, 5, 5, 2, 3, 2],
    "lucknow":    [4, 4, 3, 2, 5, 3, 5, 1, 4, 3],
    "amritsar":   [4, 4, 5, 2, 4, 4, 5, 2, 3, 1],
    "jaipur":     [4, 5, 3, 2, 4, 4, 5, 2, 5, 3],
    "jodhpur":    [4, 5, 3, 3, 3, 5, 4, 3, 4, 2],
    "udaipur":    [3, 5, 3, 5, 4, 5, 4, 3, 4, 3],
    "jaisalmer":  [4, 5, 2, 4, 3, 5, 3, 5, 3, 1],
    "hampi":      [5, 5, 4, 4, 2, 5, 4, 3, 2, 1],
    "thanjavur":  [5, 5, 5, 2, 3, 4, 5, 1, 2, 1],
    "madurai":    [4, 4, 5, 2, 4, 4, 5, 1, 3, 2],
    "mysore":     [4, 5, 4, 3, 4, 4, 5, 2, 4, 2],
    "mumbai":     [3, 4, 3, 3, 5, 4, 5, 3, 5, 5],
    "ajanta":     [5, 5, 4, 3, 2, 5, 4, 2, 1, 1],
    "khajuraho":  [5, 5, 4, 3, 2, 5, 4, 1, 2, 1],
    "sanchi":     [5, 4, 5, 3, 1, 4, 4, 1, 1, 1],
    "bhubaneswar":[5, 5, 5, 3, 3, 4, 5, 2, 2, 1],
}

def build_knn_model():
    """Build and return (knn_model, scaler, city_keys) trained on CITY_INTEREST_MATRIX."""
    keys   = list(CITY_INTEREST_MATRIX.keys())
    X      = np.array([CITY_INTEREST_MATRIX[k] for k in keys], dtype=float)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=3, metric="cosine")
    knn.fit(X_scaled)
    return knn, scaler, keys

# Train once at startup
_KNN_MODEL, _KNN_SCALER, _KNN_KEYS = build_knn_model()

def knn_recommend(interest_scores: dict, top_n: int = 3):
    """
    Given a dict {interest_name: 0/1 selected}, return top_n city keys
    ranked by cosine similarity using KNN.
    interest_scores keys must match INTERESTS order.
    """
    INTERESTS_ORDER = ["History","Architecture","Religion","Nature","Food",
                       "Photography","Culture","Adventure","Shopping","Nightlife"]
    vec = np.array([[interest_scores.get(i, 0) for i in INTERESTS_ORDER]], dtype=float)
    vec_scaled = _KNN_SCALER.transform(vec)
    distances, indices = _KNN_MODEL.kneighbors(vec_scaled, n_neighbors=top_n)
    return [_KNN_KEYS[i] for i in indices[0]]


# ════════════════════════════════════════════════════════════════════════════
#  ML ENGINE 2 — NLP INTENT & ENTITY EXTRACTION
#  Parses a free-text query typed by the user (optional field added below
#  the dropdown — same form, no layout change) to:
#    1. Extract CITY mentions (entity extraction via keyword matching +
#       synonym dictionary).
#    2. Extract INTEREST keywords (intent classification via token overlap
#       with a domain lexicon).
#  The extracted interests are fed into the KNN model to auto-select the
#  best matching city and pre-tick the interest checkboxes.
# ════════════════════════════════════════════════════════════════════════════

# Synonym → canonical city key
NLP_CITY_SYNONYMS = {
    "agra": "agra", "taj mahal": "agra", "taj": "agra",
    "delhi": "delhi", "new delhi": "delhi", "old delhi": "delhi",
    "varanasi": "varanasi", "banaras": "varanasi", "benares": "varanasi", "kashi": "varanasi",
    "lucknow": "lucknow", "nawabs": "lucknow",
    "amritsar": "amritsar", "golden temple": "amritsar", "punjab": "amritsar",
    "jaipur": "jaipur", "pink city": "jaipur",
    "jodhpur": "jodhpur", "blue city": "jodhpur", "mehrangarh": "jodhpur",
    "udaipur": "udaipur", "lake city": "udaipur",
    "jaisalmer": "jaisalmer", "golden city": "jaisalmer", "desert": "jaisalmer",
    "hampi": "hampi", "vijayanagara": "hampi",
    "thanjavur": "thanjavur", "tanjore": "thanjavur",
    "madurai": "madurai", "meenakshi": "madurai",
    "mysore": "mysore", "mysuru": "mysore",
    "mumbai": "mumbai", "bombay": "mumbai",
    "ajanta": "ajanta", "ellora": "ajanta", "caves": "ajanta",
    "khajuraho": "khajuraho",
    "sanchi": "sanchi", "stupa": "sanchi",
    "bhubaneswar": "bhubaneswar", "odisha": "bhubaneswar", "konark": "bhubaneswar",
}

# Word → interest name
NLP_INTEREST_LEXICON = {
    "history": "History", "historical": "History", "mughal": "History",
    "ancient": "History", "medieval": "History", "empire": "History",
    "architecture": "Architecture", "fort": "Architecture", "palace": "Architecture",
    "temple": "Architecture", "monument": "Architecture", "heritage": "Architecture",
    "religion": "Religion", "religious": "Religion", "pilgrimage": "Religion",
    "sacred": "Religion", "spiritual": "Religion", "worship": "Religion",
    "nature": "Nature", "wildlife": "Nature", "lake": "Nature",
    "forest": "Nature", "garden": "Nature", "birds": "Nature",
    "food": "Food", "cuisine": "Food", "eat": "Food", "spicy": "Food",
    "restaurant": "Food", "street food": "Food", "biryani": "Food",
    "photo": "Photography", "photography": "Photography", "sunrise": "Photography",
    "sunset": "Photography", "camera": "Photography",
    "culture": "Culture", "festival": "Culture", "dance": "Culture",
    "music": "Culture", "art": "Culture", "craft": "Culture",
    "adventure": "Adventure", "trek": "Adventure", "camel": "Adventure",
    "safari": "Adventure", "desert": "Adventure", "hike": "Adventure",
    "shopping": "Shopping", "bazaar": "Shopping", "market": "Shopping",
    "handicraft": "Shopping", "souvenir": "Shopping",
    "nightlife": "Nightlife", "night": "Nightlife", "club": "Nightlife",
}

def nlp_parse_query(text: str):
    """
    Returns (city_key_or_None, {interest: score}).
    Uses regex tokenization + keyword matching (no external NLP dependency).
    """
    text_lower = text.lower()
    # --- Entity Extraction: find city ---
    detected_city = None
    # Sort by length desc so "golden temple" matches before "temple"
    for phrase in sorted(NLP_CITY_SYNONYMS, key=len, reverse=True):
        if phrase in text_lower:
            detected_city = NLP_CITY_SYNONYMS[phrase]
            break

    # --- Intent Classification: detect interests ---
    tokens = set(re.findall(r"[a-z ]+", text_lower))
    interests_found = {}
    for phrase in sorted(NLP_INTEREST_LEXICON, key=len, reverse=True):
        if phrase in text_lower:
            interests_found[NLP_INTEREST_LEXICON[phrase]] = 1
    return detected_city, interests_found


# ════════════════════════════════════════════════════════════════════════════
#  ML ENGINE 3 — LINEAR REGRESSION BUDGET PREDICTOR
#  Trains a simple Linear Regression model on synthetic but realistic
#  travel cost data (duration × month → daily budget in INR).
#  Features:  [num_days, month_of_year (1-12)]
#  Target:    estimated daily cost in INR (budget / mid / luxury tier)
#  The model is queried inside _build_itinerary to display an ML-derived
#  cost estimate alongside the static lookup value.
# ════════════════════════════════════════════════════════════════════════════

def _build_budget_training_data():
    """
    Synthetic dataset: 200 rows of (days, month, budget_tier, cost_per_day).
    Costs reflect Indian domestic tourism patterns:
      - Peak season: Oct–Feb (+20%)
      - Off-peak: May–Aug (–15%)
      - Longer trips: marginal daily cost decreases (bulk discount effect)
    """
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(200):
        days   = rng.choice([2, 3, 5, 7, 10])
        month  = rng.integers(1, 13)
        tier   = rng.choice(["budget", "mid", "luxury"])
        base   = {"budget": 3200, "mid": 9500, "luxury": 26000}[tier]
        seasonal = 1.20 if month in (10,11,12,1,2) else (0.85 if month in (5,6,7,8) else 1.0)
        duration_factor = 1.0 - 0.015 * max(0, days - 2)   # slight discount for longer
        noise  = rng.normal(0, base * 0.06)
        cost   = base * seasonal * duration_factor + noise
        rows.append([days, month, tier, max(cost, base * 0.5)])
    return pd.DataFrame(rows, columns=["days","month","tier","cost"])

_BUDGET_DF = _build_budget_training_data()

def _train_budget_model(tier: str):
    sub = _BUDGET_DF[_BUDGET_DF["tier"] == tier]
    X   = sub[["days","month"]].values
    y   = sub["cost"].values
    m   = LinearRegression()
    m.fit(X, y)
    return m

_LR_MODELS = {tier: _train_budget_model(tier) for tier in ("budget","mid","luxury")}

def predict_budget(num_days: int, budget_tier: str) -> str:
    """Return ML-predicted daily cost as formatted string (e.g. 'Rs.9,840')."""
    month = datetime.now().month
    model = _LR_MODELS.get(budget_tier, _LR_MODELS["mid"])
    pred  = model.predict([[num_days, month]])[0]
    pred  = max(pred, 500)
    return f"Rs.{int(pred):,}"


# ── DESTINATION DATA ─────────────────────────────────────────────────────────
DESTINATION_DATA = {
    "agra": {
        "name": "Agra, Uttar Pradesh", "emoji": "Taj",
        "tagline": "The Eternal City of Love & Marble",
        "monuments": ["Taj Mahal","Agra Fort","Fatehpur Sikri","Mehtab Bagh","Itmad-ud-Daulah","Akbar's Tomb"],
        "budget": {"budget": "Rs.3,500", "mid": "Rs.9,500", "luxury": "Rs.28,000"},
        "days": {
            1: {"title": "The Taj & Its World",
                "morning":   {"name":"Taj Mahal at Sunrise","desc":"Enter the east gate at dawn — the marble glows a soft pink as the sun rises over the Yamuna. Built by Mughal Emperor Shah Jahan between 1631–1648 as a mausoleum for his wife Mumtaz Mahal, it is the finest example of Mughal architecture in the world. Entry: Rs.50 (Indians). Allow 2–3 hours.","tags":["UNESCO","Mughal","Photography"]},
                "afternoon": {"name":"Agra Fort","desc":"A UNESCO World Heritage Site and seat of Mughal power for generations. The red sandstone fort, built by Akbar in 1565, houses palaces like Jahangir Mahal, Khas Mahal, and the octagonal Musamman Burj where Shah Jahan spent his last years gazing at the Taj. Entry: Rs.35 (Indians).","tags":["UNESCO","Fort","History"]},
                "evening":   {"name":"Mehtab Bagh at Sunset","desc":"Commissioned by Emperor Babur, this moonlit char bagh garden lies directly north of the Taj on the opposite bank of the Yamuna. At sunset the Taj is perfectly reflected in the river — the only vantage point where this is possible. Entry: Rs.25 (Indians).","tags":["Gardens","Sunset","Photography"]}},
            2: {"title": "Fatehpur Sikri & Hidden Gems",
                "morning":   {"name":"Fatehpur Sikri","desc":"Akbar's abandoned red sandstone capital, built between 1571–1585 and deserted within 15 years — possibly due to water scarcity. A perfectly preserved Mughal ghost city 40km from Agra. The Buland Darwaza stands 54m, the largest gateway in the world. Entry: Rs.35 (Indians).","tags":["UNESCO","Mughal","Architecture"]},
                "afternoon": {"name":"Itmad-ud-Daulah (Baby Taj)","desc":"Built between 1622–1628 by Nur Jahan for her father Mirza Ghiyas Beg, this is the first Mughal structure built entirely in white marble with intricate pietra dura (stone inlay) work — a technique that later defined the Taj Mahal. Entry: Rs.20 (Indians).","tags":["Mughal","Marble","Hidden Gem"]},
                "evening":   {"name":"Sadar Bazaar & Mughal Cuisine","desc":"Explore Agra's famous bazaar for marble inlay souvenirs (pietra dura), leather goods, and petha — the city's iconic translucent white sweet made from ash gourd, a specialty since the Mughal era. Dinner at a rooftop restaurant with floodlit Taj views.","tags":["Food","Shopping","Nightlife"]}},
            3: {"title": "Akbar's Tomb & Mathura Day Trip",
                "morning":   {"name":"Akbar's Tomb, Sikandra","desc":"The mausoleum of Mughal Emperor Akbar, begun by Akbar himself in 1605 and completed by his son Jahangir in 1613, stands in a vast walled garden 10km from Agra. The five-storey red sandstone and white marble structure blends Hindu, Buddhist, Islamic and Christian motifs — a testament to Akbar's policy of religious tolerance (Din-i-Ilahi). Entry: Rs.25 (Indians).","tags":["Mughal","Mausoleum","History"]},
                "afternoon": {"name":"Mathura — Birthplace of Lord Krishna","desc":"50km from Agra, Mathura is one of the seven sacred Hindu cities and the birthplace of Lord Krishna. The Krishna Janmabhoomi complex marks the exact prison cell where Krishna was born c. 3200 BCE according to tradition. The Dwarkadhish Temple (1814) nearby is famed for its intricate Rajasthani paintings and the grand Janmashtami celebrations. Entry: Free.","tags":["Hindu","Sacred","Pilgrimage"]},
                "evening":   {"name":"Vrindavan Temples at Dusk","desc":"12km from Mathura, Vrindavan is the forest where Krishna spent his youth. The medieval Banke Bihari Temple (1862) and the ISKCON temple complex both hold spectacular evening aarti ceremonies. The Prem Mandir (2012), built entirely in Italian marble with LED illumination, is extraordinary at night. Entry: Free.","tags":["Temple","Krishna","Sacred"]}},
            4: {"title": "Agra's Craft Heritage & Moonrise",
                "morning":   {"name":"Taj Museum & Eastern Perspectives","desc":"The Taj Museum inside the Taj complex showcases original Mughal manuscripts, coins, and architectural drawings for the Taj Mahal. Visit the eastern gate for the least crowded view of the Taj, and then explore the riverside terrace along the Yamuna bank for the angle most Mughal miniatures were painted from. Entry included in Taj ticket.","tags":["Museum","Mughal","Photography"]},
                "afternoon": {"name":"Marble Inlay Artisan Workshops","desc":"Agra's pietra dura stone-inlay craft, brought to India by Mughal Emperor Jahangir from Florence, is a living tradition passed down through families in the Taj Ganj neighbourhood. Watch craftsmen cut semi-precious stones — lapis lazuli, malachite, carnelian — to fractions of a millimetre. The best workshops accept commissions. Cost: Free to visit.","tags":["Craft","Art","Mughal"]},
                "evening":   {"name":"Taj Mahal by Moonlight","desc":"On five nights around each full moon, the Archaeological Survey of India opens the Taj Mahal to moonlight viewing between 8:30–12:30pm. Tickets (Rs.510) must be booked at the ASI office in advance and are capped at 400 visitors — the most intimate and ethereal way to experience the monument. Check lunar calendar before booking.","tags":["UNESCO","Moonlight","Photography"]}},
            5: {"title": "Bharatpur Bird Sanctuary & Surrounds",
                "morning":   {"name":"Keoladeo National Park, Bharatpur","desc":"55km from Agra, the Keoladeo Ghana Bird Sanctuary (UNESCO 1985) was once the private duck-hunting reserve of the Maharajas of Bharatpur. The wetland hosts over 370 bird species including the rare Siberian crane (winter visitor). Cycle or take a cycle-rickshaw through the park at dawn for the best sightings. Entry: Rs.75 (Indians).","tags":["UNESCO","Birds","Wildlife"]},
                "afternoon": {"name":"Bharatpur Fort & Lohagarh","desc":"The 'Iron Fort' of Bharatpur was the only fort in India never captured by the British — withstanding two major sieges (1805) by General Lake. The earthen ramparts are surrounded by a wide moat. The Jawahar Burj and Fateh Burj towers commemorate the Jat ruler Suraj Mal's victories against the Mughals and British. Entry: Rs.25 (Indians).","tags":["Fort","Jat","History"]},
                "evening":   {"name":"Deeg Palace & Water Gardens","desc":"35km from Bharatpur, the summer palace of the Jat Maharajas of Bharatpur (1772) is one of Rajasthan's least-visited but most beautiful palaces. The Gopal Bhawan pavilion is flanked by two massive reservoirs; during the monsoon festival, 2,000 fountains are activated simultaneously. Entry: Rs.25 (Indians).","tags":["Palace","Gardens","Hidden Gem"]}},
            6: {"title": "Gwalior — The Gibraltar of India",
                "morning":   {"name":"Gwalior Fort","desc":"120km south of Agra, Gwalior Fort rises 100m above the plain on an isolated sandstone plateau — called the 'pearl amongst fortresses of India' by Mughal Emperor Babur. The fort contains the 8th-century Teli Ka Mandir (one of the tallest early medieval temples), the Sas-Bahu Temples, and the Man Singh Palace (1508) with its vivid blue tilework. Entry: Rs.75 (Indians).","tags":["Fort","Medieval","Architecture"]},
                "afternoon": {"name":"Jai Vilas Palace & Scindias","desc":"The 1874 Italianate palace of the Scindia dynasty houses India's most opulent private museum. The durbar hall has two chandeliers weighing 3.5 tonnes each — before they were hung, ten elephants were placed on the ceiling to test its strength. The dining table has a silver model train that circulates brandy and cigars. Entry: Rs.200 (Indians).","tags":["Palace","Colonial","Museum"]},
                "evening":   {"name":"Tansen's Tomb & Gwalior Music","desc":"Gwalior is the birthplace of the Dhrupad style of Hindustani classical music. The tomb of Tansen — Akbar's most celebrated court musician — is the site of the annual Tansen Music Festival. The adjacent Ghaus Mohammed's Mughal tomb has some of the finest blue tile work outside of Central Asia. Entry: Free.","tags":["Music","Mughal","Culture"]}},
            7: {"title": "Orchha — The Forgotten Royal City",
                "morning":   {"name":"Orchha Fort & Raja Mahal","desc":"230km from Agra, the abandoned Bundela capital of Orchha (founded 1501) rises from the banks of the Betwa river. The Raja Mahal (royal palace) contains some of the finest surviving Mughal-era wall paintings in India — large-scale narratives of Lord Vishnu's avatars painted directly onto walls. Entry: Rs.25 (Indians).","tags":["Mughal","Painting","Hidden Gem"]},
                "afternoon": {"name":"Jahangir Mahal, Orchha","desc":"Built by the Bundela king Bir Singh Deo to celebrate Emperor Jahangir's visit in 1606, the Jahangir Mahal is a 3-storey palace of exquisite stone craftsmanship. Its rooftop offers a panoramic view of Orchha's chhatris (royal cenotaphs) lining the Betwa riverbank — 14 in all, each dedicated to a different Bundela ruler. Entry: Rs.25 (Indians).","tags":["Palace","Bundela","Views"]},
                "evening":   {"name":"Ram Raja Temple & Betwa Riverfront","desc":"The only temple in India where Lord Ram is worshipped as a king (raja) and given a military salute by police guards. The story goes that a palace was consecrated as a temple when a Ram idol brought by Queen Ganesh Kunwari from Ayodhya could not be moved after being set down. The Betwa riverside at sunset is serene and beautiful. Entry: Free.","tags":["Temple","Sacred","History"]}},
            8: {"title": "Agra Deep Dive — Gardens & Ghats",
                "morning":   {"name":"Ram Bagh — India's Oldest Mughal Garden","desc":"Built by Emperor Babur in 1528 on the banks of the Yamuna, Ram Bagh (originally Aram Bagh, Garden of Rest) is the oldest surviving Mughal garden in India and possibly the world. Its char bagh layout directly inspired the Taj Mahal's garden design. Babur himself rested here on his campaigns; his body was temporarily kept here before burial at Kabul. Entry: Rs.25 (Indians).","tags":["Mughal","Garden","History"]},
                "afternoon": {"name":"Dayal Bagh Temple","desc":"The Radha Soami Satsang Temple at Dayal Bagh, under continuous construction since 1904, is considered by many craftsmen to surpass the Taj Mahal in intricacy. The white marble and sandstone structure uses the same pietra dura stone-inlay technique with over 100 craftsmen working on it daily. It has been under construction for over 120 years with no completion date. Entry: Free.","tags":["Temple","Craft","Architecture"]},
                "evening":   {"name":"Yamuna Riverfront at Sunset","desc":"Walk the Yamuna riverfront south of the Taj Mahal past the ruins of the Moonlight Garden to the Arjun Nagar ghats. This stretch offers the most complete view of the Mughal riverside — Agra Fort, the Taj, and the Baby Taj all visible from the river's edge at golden hour. Cost: Free.","tags":["Photography","Sunset","Heritage"]}},
            9: {"title": "Fatehpur Sikri Deep Dive",
                "morning":   {"name":"Salim Chishti's Dargah, Fatehpur Sikri","desc":"Within the Jama Masjid courtyard at Fatehpur Sikri stands the marble tomb (1581) of Sufi saint Sheikh Salim Chishti — whose prayers Emperor Akbar credited for the birth of his son Salim (later Emperor Jahangir). Childless couples from across India tie threads on the marble screens, believing the saint's blessings are still active. Entry: Free.","tags":["Sufi","Sacred","Mughal"]},
                "afternoon": {"name":"Panch Mahal & Diwan-i-Khas","desc":"The Panch Mahal is a five-storey pavilion tapering to a single kiosk at the top — its 176 columns are all unique in design. The adjacent Diwan-i-Khas (Hall of Private Audience) contains Akbar's remarkable central pillar with 36 serpentine brackets — he allegedly sat at the top and debated religion with scholars from all faiths below. Entry: Rs.35 (Indians).","tags":["Mughal","Architecture","History"]},
                "evening":   {"name":"Agra Goodbye Dinner — Mughal Thali","desc":"End your Agra journey with a full Mughal feast at one of the city's heritage restaurants. The Mughal culinary tradition brought to Agra by Babur evolved through Akbar's tandoori techniques to Shah Jahan's refined court cuisine. A proper Agra Mughal thali includes nihari, kebabs, naan, korma, zarda rice, and shahi tukda dessert. Cost: Rs.800–2,000/person.","tags":["Food","Mughal","Culture"]}},
            10: {"title": "Mathura Pilgrimage & Departure",
                "morning":   {"name":"Mathura Museum","desc":"The Government Museum in Mathura houses one of India's finest collections of Mathura School sculpture — the red sandstone Buddha and Jain tirthankara figures (1st–5th century CE) that rivalled the Gandhara School in influencing all later Buddhist art across Asia. The museum's 16,000-piece collection spans 5,000 years. Entry: Rs.20 (Indians).","tags":["Museum","Buddhist","Sculpture"]},
                "afternoon": {"name":"Govardhan Hill Parikrama","desc":"25km from Mathura, Govardhan Hill is sacred as the hill Lord Krishna lifted on his finger to shelter the people of Vrindavan from Indra's rains. Pilgrims perform a 21km barefoot circumambulation (parikrama) of the hill, passing 22 kunds (sacred ponds) and dozens of temples. The summit offers views over the Braj region. Entry: Free.","tags":["Hindu","Pilgrimage","Sacred"]},
                "evening":   {"name":"Final Sunset at Taj Mahal","desc":"Return to the Taj Mahal for a final sunset viewing from the western gate. The late afternoon light turns the white marble through shades of gold, orange and rose — different at every hour. Photography tips: the reflection pool inside the main gate frames the mausoleum perfectly between 4–6pm depending on the season. Entry: Rs.50 (Indians).","tags":["UNESCO","Sunset","Photography"]}},
        }
    },
    "delhi": {
        "name": "Delhi", "emoji": "D",
        "tagline": "8 Cities, 3,000 Years of History",
        "monuments": ["Red Fort","Qutub Minar","Humayun's Tomb","India Gate","Lotus Temple","Purana Qila"],
        "budget": {"budget": "Rs.4,000", "mid": "Rs.12,000", "luxury": "Rs.35,000"},
        "days": {
            1: {"title": "Old Delhi & Mughal Grandeur",
                "morning":   {"name":"Red Fort at Dawn","desc":"Built by Shah Jahan between 1638–1648 as the Mughal capital's centrepiece, this red sandstone fort spans 2km and was the ceremonial and political heart of Mughal India for nearly 200 years. Walk through Diwan-i-Khas, the pearl mosque (Moti Masjid), and the Rang Mahal. Entry: Rs.35 (Indians).","tags":["UNESCO","Mughal","History"]},
                "afternoon": {"name":"Jama Masjid & Chandni Chowk","desc":"India's largest mosque, built by Shah Jahan between 1650–1656, can hold 25,000 worshippers in its vast courtyard. Explore the adjacent Chandni Chowk — Mughal Delhi's royal shopping boulevard, laid out in 1650, today a labyrinth of spice markets, sweets, and street food. Entry: Free.","tags":["Mughal","Islam","Culture"]},
                "evening":   {"name":"Qutub Minar by Evening Light","desc":"The 72.5m minaret was built in 1193 by Qutb ud-Din Aibak — the first tower of victory in Islam in South Asia. The complex also contains the Iron Pillar of Delhi, a 6-tonne 4th-century pillar that has not rusted in over 1,600 years despite outdoor exposure. Entry: Rs.35 (Indians).","tags":["UNESCO","Delhi Sultanate","Architecture"]}},
            2: {"title": "Mughal Tombs & New Delhi",
                "morning":   {"name":"Humayun's Tomb","desc":"Built in 1570 by the Mughal Emperor Humayun's widow Hamida Banu Begum, this UNESCO site is the first garden tomb on the Indian subcontinent and the direct architectural predecessor of the Taj Mahal. The 47m dome inspired generations of Mughal buildings. Entry: Rs.35 (Indians).","tags":["UNESCO","Mughal","Garden"]},
                "afternoon": {"name":"India Gate & Kartavya Path","desc":"The 42m war memorial, designed by Edwin Lutyens and completed in 1931, bears the names of 13,516 soldiers of the British Indian Army killed in World War I. Walk the grand ceremonial boulevard, recently renamed Kartavya Path, to Rashtrapati Bhavan. Entry: Free.","tags":["Colonial","Memorial","Architecture"]},
                "evening":   {"name":"Hauz Khas Village","desc":"Built by Sultan Alauddin Khalji in 1296, the Hauz Khas reservoir once supplied water to Siri, his new capital. Today, its 14th-century madrasa ruins, deer park, and stepwell are surrounded by art galleries, cafes, and rooftop restaurants.","tags":["Medieval","Food","Nightlife"]}},
            3: {"title": "Sultanate Delhi & Mehrauli",
                "morning":   {"name":"Mehrauli Archaeological Park","desc":"A 200-acre park in south Delhi containing over 440 listed heritage monuments spanning 1,000 years — the densest concentration of medieval heritage in any Indian city. Key sites include the Jamali-Kamali mosque and tomb (1536), the Balban's Tomb (1287) — the earliest example of a true arch in India — and the octagonal Metcalfe's Folly.","tags":["Medieval","Sultanate","Archaeology"]},
                "afternoon": {"name":"Purana Qila & Sher Shah Suri","desc":"The Old Fort of Delhi was built by both the Mughal emperor Humayun (who began it) and the Sur Empire's Sher Shah Suri (who completed it, 1538–1545). The site is believed to overlie Indraprastha, the capital of the Pandavas in the Mahabharata. The Qila-i-Kuhna Mosque inside is one of the finest Sur-period buildings. Entry: Rs.25 (Indians).","tags":["Mughal","Sur Empire","History"]},
                "evening":   {"name":"Nizamuddin Dargah — Thursday Qawwali","desc":"The shrine of Sufi saint Hazrat Nizamuddin Auliya (1238–1325) is one of the most spiritually charged places in Delhi. Every Thursday evening, devotional qawwali music rings out in the dargah courtyard — attended by thousands of devotees. The shrine also contains the tombs of poet Amir Khusrau and Mughal princess Jahanara. Entry: Free.","tags":["Sufi","Qawwali","Sacred"]}},
            4: {"title": "Lodi Gardens & Colonial Delhi",
                "morning":   {"name":"Lodi Gardens","desc":"The Lodi Gardens contain the tombs of the Sayyid and Lodi dynasties (15th–16th century) set within 90 acres of manicured gardens — a remarkable open-air museum. The octagonal Sikandar Lodi Tomb (1517) and the double-storey Shish Gumbad are outstanding examples of late Sultanate architecture. The garden is Delhi's most popular morning walk.","tags":["Sultanate","Gardens","Architecture"]},
                "afternoon": {"name":"Rashtrapati Bhavan & Mughal Gardens","desc":"The President's residence, designed by Edwin Lutyens (1929), is one of the last great imperial buildings ever erected. The 340-room palace sits on a 330-acre estate that includes the Mughal Garden — 15 acres of formal gardens open to the public in February. The circular colonnaded Durbar Hall was inspired by the Pantheon in Rome.","tags":["Colonial","Architecture","Heritage"]},
                "evening":   {"name":"Connaught Place & Heritage Stroll","desc":"The circular colonnaded Connaught Place was built between 1929–1933 as New Delhi's commercial heart. Named after the Duke of Connaught, it was designed by RobertTor Russell in the neoclassical style. The underground Palika Bazaar (1978) beneath it is one of Asia's first underground markets. The evening street food of Janpath and Bengali Market are legendary.","tags":["Colonial","Food","Shopping"]}},
            5: {"title": "Sikh Heritage & Walled City",
                "morning":   {"name":"Gurudwara Bangla Sahib","desc":"Built on the site where Sikh Guru Har Krishan (the 8th Guru) stayed in 1664 before his death, Bangla Sahib is the most prominent Sikh temple in Delhi. The sarovar (sacred pool) is believed to have healing properties. The free langar (community kitchen) serves 10,000 meals daily. Entry: Free.","tags":["Sikh","Sacred","Culture"]},
                "afternoon": {"name":"Shahjahanabad Walking Tour","desc":"Shahjahanabad — the walled city built by Shah Jahan between 1638–1648 — is the core of Old Delhi. A guided walk through the Khari Baoli spice market (Asia's largest), the Kinari Bazaar of wedding accessories, and the Chatta Chowk inside Red Fort reveals 400 years of continuous urban life. Hire a local heritage guide for Rs.500–1,000.","tags":["Mughal","Walking","Culture"]},
                "evening":   {"name":"Paranthe Wali Gali & Chandni Chowk","desc":"The narrow alley of Paranthe Wali Gali in Chandni Chowk has been frying stuffed parathas since 1875 — the same families, sometimes the same griddles, for 150 years. End your evening at the Ghantewala sweet shop (1790, Delhi's oldest) for sohan halwa and daulat ki chaat — a seasonal morning dew sweet served only in winter. Cost: Rs.200–400.","tags":["Food","Heritage","Culture"]}},
            6: {"title": "Akshardham & Modern Heritage",
                "morning":   {"name":"Akshardham Temple","desc":"Inaugurated in 2005 and built in just 5 years by 11,000 craftsmen, Akshardham's central monument is a 43m pink sandstone and marble mandir covered in 20,000 carved figures. No metal was used in its construction. The architectural style replicates 8th-century Solanki craftsmanship. The complex also contains India's largest step-well garden. Entry: Free (exhibitions extra).","tags":["Temple","Modern","Architecture"]},
                "afternoon": {"name":"National Museum of India","desc":"India's largest museum, opened in 1949, houses over 200,000 artefacts spanning 5,000 years. Highlights include the Dancing Girl of Mohenjo-daro (2500 BCE), the Gupta gold coins, Gandhara Buddhas, Mughal miniatures, and the complete skeleton of a 3-billion-year-old dinosaur from the Deccan Traps. Entry: Rs.20 (Indians).","tags":["Museum","History","Archaeology"]},
                "evening":   {"name":"Khan Market & Lodhi Colony Street Art","desc":"Lodhi Colony's government housing blocks have been transformed into an open-air street art gallery by the St+art India Foundation — 50 large-scale murals by international and Indian artists across the colony's 100+ walls. Walk through at dusk as the light catches the largest collection of public art in any Indian neighbourhood. Entry: Free.","tags":["Art","Contemporary","Culture"]}},
            7: {"title": "Agra Day Trip from Delhi",
                "morning":   {"name":"Taj Mahal Sunrise from Delhi","desc":"Take the 6am Gatimaan Express (India's fastest train at 160km/h, journey 1hr 40min) to Agra for the ultimate day trip. Arrive at the Taj Mahal's east gate before 8am to experience the monument with minimal crowds. The train departs New Delhi station and reaches Agra Cantt — book tickets in advance on IRCTC. Train: Rs.755 (AC chair car).","tags":["Taj Mahal","Day Trip","Train"]},
                "afternoon": {"name":"Agra Fort & Mughal Monuments","desc":"After the Taj, spend the afternoon at Agra Fort — the red sandstone palace-fortress where Shah Jahan was imprisoned by his son Aurangzeb and spent his last years gazing at the Taj across the Yamuna. Then visit Itmad-ud-Daulah, the 'Baby Taj', for its extraordinary pietra dura inlays. Entry: Rs.35 (Indians).","tags":["UNESCO","Fort","Mughal"]},
                "evening":   {"name":"Return by Taj Express & Dinner","desc":"Return to Delhi on the evening Taj Express. Dine in South Delhi on the way back — the Saket or Malviya Nagar restaurant strips offer everything from traditional Mughlai to contemporary Indian cuisine. The round trip from Delhi perfectly captures the essence of the Taj Mahal experience. Train: Rs.200–500.","tags":["Travel","Food","Heritage"]}},
            8: {"title": "Sultanpur & Sohna Rural Heritage",
                "morning":   {"name":"Sultanpur National Park","desc":"50km from Delhi, Sultanpur is a Ramsar wetland and national park built around a seasonal lake that hosts over 250 bird species including migratory flamingos, pelicans, and Siberian cranes (November–March). Early morning is essential — bird activity drops sharply by 9am. Bring binoculars. Entry: Rs.25 (Indians).","tags":["Birds","Nature","Wildlife"]},
                "afternoon": {"name":"Pataudi Palace & Nawab Heritage","desc":"75km from Delhi, the Pataudi Palace (1935) was the ancestral home of the Nawabs of Pataudi and later the cricket legend Tiger Pataudi. The palace blends Neoclassical and Mughal architecture in a forest setting. It is now a heritage hotel; non-guests can visit for afternoon high tea. Cost: Rs.500–1,000.","tags":["Palace","Heritage","Nature"]},
                "evening":   {"name":"Sohna Sulphur Springs","desc":"Sohna, 55km from Delhi in the Aravalli foothills, has been famous for its sulphur hot springs since Mughal times. Emperor Humayun reportedly halted his campaigns here to bathe. Today it is a popular weekend escape with a warm spring-fed pool. The drive through the Aravalli forest is beautiful at sunset. Cost: Rs.100–300.","tags":["Nature","Wellness","Heritage"]}},
            9: {"title": "Janpath Crafts & Architectural Walks",
                "morning":   {"name":"Crafts Museum, Pragati Maidan","desc":"India's largest crafts museum occupies a rambling campus where 33 full-sized traditional buildings from different Indian states were reconstructed. The collection spans over 35,000 objects — Madhubani paintings, tribal bronzes, Rajasthani puppets, and Kantha embroidery. The Crafts Museum shop is the best place in Delhi to buy authentic handicrafts directly. Entry: Rs.20 (Indians).","tags":["Crafts","Museum","Culture"]},
                "afternoon": {"name":"Dilli Haat & State Emporiums","desc":"Dilli Haat (INA) is a permanent open-air market where artisans from each Indian state rotate on two-week stalls, selling handicrafts directly to buyers. The adjacent state emporiums on Baba Kharak Singh Marg offer fixed-price authentic goods from every Indian state. Together they represent the most complete cross-section of Indian craftsmanship in one place.","tags":["Shopping","Craft","Culture"]},
                "evening":   {"name":"Siri Fort & Tughlaqabad Ruins","desc":"Siri Fort (1303) was the second of Delhi's seven cities, built by Alauddin Khalji to protect against Mongol invasions. The remaining ramparts are scattered across south Delhi's residential areas — a remarkable urban palimpsest. Nearby Tughlaqabad Fort (1321) is one of Delhi's most dramatic ruins — a vast abandoned walled city of 6.5km circumference. Entry: Rs.25 (Indians).","tags":["Sultanate","Fort","History"]}},
            10: {"title": "Delhi Day 10 — Lal Kuan & Departure",
                "morning":   {"name":"Humayun's Tomb Gardens at Dawn","desc":"Return to Humayun's Tomb complex — but this time explore the lesser-visited Isa Khan Tomb (1547), the Arab Serai gateway, and the recently restored Sunder Nursery. This 16th-century park surrounding Humayun's Tomb has been restored to its original Mughal char bagh layout with 90 heritage trees and 300+ flowering species. Entry: Rs.35 (Indians).","tags":["UNESCO","Garden","Mughal"]},
                "afternoon": {"name":"Raj Ghat & Gandhi Smriti","desc":"Raj Ghat is the simple black marble platform marking the exact site of Mahatma Gandhi's cremation on 31 January 1948. An eternal flame burns here. Nearby, Gandhi Smriti is the colonial bungalow (formerly Birla House) where Gandhi spent his last 144 days and where he was assassinated — the footstep path leading to the prayer ground is deeply moving. Entry: Free.","tags":["Gandhi","Independence","Memorial"]},
                "evening":   {"name":"Karim's Restaurant — Farewell Dinner","desc":"Karim's in Gali Kababian, off Jama Masjid, has been serving Mughal court recipes since 1913. Founded by the descendants of imperial Mughal chefs, the restaurant still makes its mutton korma and nahari from slow-cooking recipes that predate the fall of the Mughal Empire. The original outdoor kitchen and clay pots have not changed in a century. Cost: Rs.500–800/person.","tags":["Food","Mughal","Heritage"]}},
        }
    },
    "jaipur": {
        "name": "Jaipur, Rajasthan", "emoji": "JP",
        "tagline": "The Pink City of Rajput Splendour",
        "monuments": ["Amber Fort","Hawa Mahal","City Palace","Jantar Mantar","Nahargarh Fort","Jal Mahal"],
        "budget": {"budget": "Rs.3,200", "mid": "Rs.9,000", "luxury": "Rs.26,000"},
        "days": {
            1: {"title": "Amber Fort & the Hilltops",
                "morning":   {"name":"Amber Fort at Sunrise","desc":"Perched on the Aravalli hills above Maota Lake, Amber Fort was built by Raja Man Singh I in 1592 and expanded by successive Kachhwaha rulers over 150 years. The dazzling Sheesh Mahal (Hall of Mirrors) was created by embedding thousands of tiny convex mirrors in the plasterwork. Entry: Rs.100 (Indians).","tags":["UNESCO","Rajput","Fort"]},
                "afternoon": {"name":"Jaigarh & Nahargarh Forts","desc":"Jaigarh Fort (1726) houses the Jaivana — the world's largest wheeled cannon, weighing 50 tonnes, which was test-fired only once. Just 2km away, Nahargarh (1734) was built by Maharaja Sawai Jai Singh II as a retreat and offers a panoramic view of the entire Pink City. Entry: Rs.35–50 (Indians).","tags":["Fort","Views","History"]},
                "evening":   {"name":"Jal Mahal at Dusk","desc":"The five-storey Water Palace rises from Man Sagar Lake — only the top storey is visible above water. Built in the 18th century by Maharaja Madho Singh I, it was used as a duck-hunting lodge. The palace is best photographed from the lakeside promenade at golden hour. Entry: Viewpoint free.","tags":["Palace","Photography","Sunset"]}},
            2: {"title": "The Pink City Within",
                "morning":   {"name":"Hawa Mahal & Bazaars","desc":"The Palace of Winds was built in 1799 by Maharaja Sawai Pratap Singh in the shape of Lord Krishna's crown. Its 953 jharokha (latticed) windows allowed royal women to observe street festivals while remaining in purdah. The honeycomb facade has no main door — it is entered from the rear. Entry: Rs.50 (Indians).","tags":["Architecture","Shopping","Photography"]},
                "afternoon": {"name":"City Palace & Jantar Mantar","desc":"The City Palace complex, built between 1729–1732, is still the official residence of the Jaipur royal family. The adjacent Jantar Mantar (1734) is the world's largest stone astronomical observatory with 19 instruments, including the Samrat Yantra — a 27m sundial accurate to 2 seconds. Entry: Rs.50–200 (Indians).","tags":["UNESCO","Royalty","Science"]},
                "evening":   {"name":"Chokhi Dhani Village Dinner","desc":"A 10-acre themed Rajasthani village resort on the outskirts of Jaipur. Experience traditional ghoomar and kalbelia folk dances, kathputli (puppet) shows, camel rides, and end the evening with a seated Rajasthani thali dinner on the ground under the stars. Cost: Rs.700–900/person.","tags":["Culture","Food","Music"]}},
            3: {"title": "Albert Hall & Textile Heritage",
                "morning":   {"name":"Albert Hall Museum","desc":"Built in 1887 in the Indo-Saracenic style by Sir Samuel Swinton Jacob, Albert Hall is Rajasthan's oldest museum. The collection spans Egyptian mummies, Persian carpets, ancient coins, Mughal miniatures, and an extraordinary collection of Rajasthani folk art. The building itself — designed to celebrate the Prince of Wales's 1876 visit — is among Jaipur's finest pieces of architecture. Entry: Rs.40 (Indians).","tags":["Museum","Colonial","Art"]},
                "afternoon": {"name":"Block Printing Village — Sanganer","desc":"The village of Sanganer, 16km from Jaipur, is the centre of Jaipur's celebrated hand block-printing tradition. Natural dyes (indigo, turmeric, pomegranate rind) are used on cotton and silk in the dabu (mud-resist) and bagru techniques perfected over centuries. Visit a family workshop, try printing yourself, and shop directly from the artisans. Cost: Free to visit.","tags":["Craft","Textile","Art"]},
                "evening":   {"name":"Bapu Bazaar & Johari Bazaar","desc":"Johari Bazaar (Jewellers' Market) and Bapu Bazaar are the twin hearts of Jaipur's famous gem and jewellery trade. Jaipur is one of the world's top three centres for coloured gemstone cutting and polishing — over 90% of the world's rubies pass through here. Shop for Kundan, meenakari, and lac jewellery under the pink colonnades at dusk. Cost: Free to browse.","tags":["Shopping","Gems","Culture"]}},
            4: {"title": "Ranthambore Tiger Safari",
                "morning":   {"name":"Ranthambore National Park — Morning Safari","desc":"180km from Jaipur, Ranthambore is one of India's best tiger reserves and the most reliable in the country for tiger sightings in the wild. The park's 1,300 sq km of dry deciduous forest contains ruins of the 10th-century Ranthambore Fort — tigers are frequently photographed against the fort's ancient walls. Safari: Rs.1,500–2,500 (jeep share).","tags":["Tiger","Wildlife","Nature"]},
                "afternoon": {"name":"Ranthambore Fort","desc":"The UNESCO Ranthambore Fort (944 CE) stands on a 700m rocky outcrop inside the tiger reserve. Built by the Chahamana dynasty and later held by the Mughals, the fort is now uninhabited — its temples, tanks, and ruined palaces are patrolled by deer, langurs, and tigers. The fort is reachable only during the park's safari hours. Entry: Included in safari.","tags":["Fort","Medieval","Wildlife"]},
                "evening":   {"name":"Sawai Madhopur Village Heritage Walk","desc":"Sawai Madhopur is the gateway town to Ranthambore. An evening walk through the old town reveals the step-well culture of Rajasthan, a Kacheri (old courthouse), and local tribal artisans making Rajasthani blue pottery. Dinner at a heritage haveli guesthouse. Cost: Rs.500–1,500.","tags":["Village","Heritage","Culture"]}},
            5: {"title": "Pushkar — Sacred Lake & Camel Fair",
                "morning":   {"name":"Pushkar — The Sacred Lake","desc":"145km from Jaipur, Pushkar is one of the five sacred dhams of Hinduism. The Pushkar Lake is believed to have been created when Lord Brahma dropped a lotus flower from his hand. The ghats surrounding the lake are where pilgrims bathe to attain moksha. The town contains over 500 temples, including the world's only Brahma Temple (14th century). Entry: Free (donations at ghats).","tags":["Hindu","Sacred","Pilgrimage"]},
                "afternoon": {"name":"Pushkar Camel Fair Grounds & Rose Fields","desc":"Pushkar hosts the world's largest camel fair each November (Kartik Purnima), drawing 50,000 camels and 200,000 people. Even outside the fair, the fair grounds at the edge of the desert make for fascinating exploration. The town's rose farms supply much of India's rose water and attar — visit during the October harvest for the extraordinary fragrance.","tags":["Camel","Culture","Nature"]},
                "evening":   {"name":"Savitri Temple Sunset by Ropeway","desc":"The hilltop Savitri Temple, dedicated to Brahma's first wife, is reached by a ropeway (Rs.100) above the Pushkar valley. The sunset view over the Pushkar Lake, the Thar Desert, and the ancient town below is one of the most spectacular in Rajasthan. Entry: Free (ropeway extra).","tags":["Temple","Sunset","Views"]}},
            6: {"title": "Samode & Royal Havelis",
                "morning":   {"name":"Samode Palace & Painted Haveli","desc":"42km north of Jaipur, the 400-year-old Samode Palace is one of Rajasthan's most intact royal palaces — its Sheesh Mahal (Hall of Mirrors) and Durbar Hall are covered from floor to ceiling with miniature paintings. Now a heritage hotel, the palace grounds are open to visitors who take the morning tour. Entry: Rs.200–500 (tour).","tags":["Palace","Miniature Paintings","Heritage"]},
                "afternoon": {"name":"Abhaneri Stepwell — Chand Baori","desc":"95km from Jaipur, the Chand Baori stepwell at Abhaneri (800 CE) is one of the largest and most spectacular stepwells in India — 13 storeys deep with 3,500 narrow steps arranged in perfect symmetry. The adjacent Harshat Mata Temple has some of the finest 9th-century sandstone sculpture in Rajasthan. Entry: Rs.25 (Indians).","tags":["Stepwell","Medieval","Architecture"]},
                "evening":   {"name":"Jaipur Gem Workshop & Heritage Dinner","desc":"Jaipur's gem-cutting industry employs 100,000 artisans and processes over 80% of the world's emeralds. An evening gem workshop visit (by appointment through a heritage hotel) shows the cutting, polishing, and setting process up close. Dinner at a haveli restaurant in the old city with live Rajasthani folk music. Cost: Rs.1,000–2,000.","tags":["Gems","Craft","Culture"]}},
            7: {"title": "Shekhawati — Painted Towns",
                "morning":   {"name":"Nawalgarh Havelis — Open-Air Museum","desc":"165km from Jaipur in the Shekhawati region, the town of Nawalgarh contains 1,500 havelis from the 19th century, decorated with detailed exterior and interior murals — trains, aeroplanes, European women, and mythological scenes all painted by local artists who had never left Rajasthan. It is called the 'open-air art gallery of Rajasthan.' Entry: Rs.50–200 per haveli.","tags":["Havelis","Murals","Heritage"]},
                "afternoon": {"name":"Mandawa & Fatehpur Shekhawati","desc":"Mandawa (30km from Nawalgarh) has the densest concentration of painted havelis in Shekhawati — the Castle Mandawa and the Hanuman Prasad Goenka Haveli are particularly magnificent. Fatehpur (35km), founded in 1451, has the extraordinary Nadine Le Prince Cultural Centre — a French artist's restoration of a merchant haveli into a heritage museum.","tags":["Art","Heritage","History"]},
                "evening":   {"name":"Sikar's Step-wells & Desert Return","desc":"Sikar, the main Shekhawati town, has several 18th-century step-wells and the Raghunath Temple with its extraordinary painted courtyard. Return to Jaipur via the Shekhawati sunset highway — the landscape of scrub desert, windmills, and painted villages at dusk is quintessential Rajasthan. Cost: Hired car Rs.3,000–4,000 for the day.","tags":["Stepwell","Desert","Heritage"]}},
            8: {"title": "Bundi & Ancient Rock Paintings",
                "morning":   {"name":"Bundi Palace & Taragarh Fort","desc":"210km from Jaipur, Bundi was the old capital of a Rajput kingdom and contains one of the most spectacularly decorated palaces in Rajasthan. The Chitrashala (painting gallery) inside Bundi Palace has 17th-century murals acknowledged by art historians as the finest in Rajasthan. The palace is painted in a distinctive turquoise-blue. Entry: Rs.100 (Indians).","tags":["Palace","Paintings","Rajput"]},
                "afternoon": {"name":"Bundi Step-wells & 84 Pillared Cenotaph","desc":"Bundi has 50+ step-wells (baolies) — more than any other town in India. The Raniji ki Baori (1699), built by the queen of Bundi, is the grandest — 46m deep with extraordinary sculpted panels. The Chaurasi Khambon ki Chhatri (84-pillared cenotaph) is a forest of slender columns overlooking the town and lake.","tags":["Stepwell","Heritage","Architecture"]},
                "evening":   {"name":"Rock Paintings of Garradh","desc":"The area around Bundi contains prehistoric rock paintings at Garradh estimated to be 20,000–30,000 years old — some of the oldest art in India, depicting hunting scenes, animals, and human figures in red and white ochre. A local guide can lead you to the sites in the Vindhya hills. Cost: Guide Rs.500. Entry: Free.","tags":["Prehistoric","Rock Art","Nature"]}},
            9: {"title": "Kota — Industrial Heritage & Wildlife",
                "morning":   {"name":"Kota City Palace & Museum","desc":"240km from Jaipur, Kota's city palace on the banks of the Chambal river houses one of India's most extraordinary palace museums — collections of weapons, elephant howdahs, giant bronze figures, and extraordinary paintings of the Kota School of miniature painting (17th–19th century), which was famous for its hunting scenes painted with exceptional vitality. Entry: Rs.100 (Indians).","tags":["Palace","Museum","Art"]},
                "afternoon": {"name":"Chambal River Crocodile Safari","desc":"The Chambal River between Kota and Dholpur is the cleanest river in the Gangetic plain and the last stronghold of the critically endangered gharial crocodile. A boat safari on the Chambal (Rs.1,500–2,000) offers near-certain sightings of gharials sunbathing on sandbanks, along with Gangetic dolphins, smooth-coated otters, and dozens of bird species.","tags":["Wildlife","Nature","Safari"]},
                "evening":   {"name":"Kota Doria Weaving & Sunset Riverside","desc":"Kota Doria is a fine cotton-silk muslin with a distinctive chequered pattern, woven exclusively in Kota's Kaithoon village. The fabric is so light it was described by Mughal poets as 'woven air.' Visit a Kaithoon cooperative to see the pit-loom weaving and purchase directly. The Chambal riverbank at sunset, with its gharials and migratory birds, is spectacular.","tags":["Craft","Textile","Nature"]}},
            10: {"title": "Jaipur Finale — Perfume & Pottery",
                "morning":   {"name":"Blue Pottery Workshop, Sanganer","desc":"Jaipur's famous blue pottery is not made from clay — it is made from quartz stone powder, powdered glass, Multani mitti (Fuller's earth), borax, and gum, fired at 800°C. The tradition was brought from Persia and Afghanistan by the Mughals. A morning workshop lets you throw and glaze your own piece to take home. Cost: Rs.300–800/session.","tags":["Craft","Blue Pottery","Art"]},
                "afternoon": {"name":"Galtaji Monkey Temple","desc":"The Galtaji Temple complex, 10km from Jaipur in a natural gorge, dates to the 18th century and is a series of sacred kunds (tanks) fed by a natural spring. The main temple is dedicated to the Sun God. The complex is overrun by thousands of rhesus monkeys (earning its nickname 'Monkey Temple') and is free from the tourist crowds of central Jaipur. Entry: Free.","tags":["Temple","Nature","Hidden Gem"]},
                "evening":   {"name":"Farewell at Nahargarh Sunset Café","desc":"Return to Nahargarh Fort for a final sunset over the Pink City. The fort's rampart café serves food and chai with a panoramic view of Jaipur's sea of rose-coloured buildings stretching to the Aravalli horizon. At dusk the city lights up and the Amber Fort's floodlights turn gold. One of Rajasthan's most sublime views. Entry: Rs.50 (Indians).","tags":["Sunset","Views","Farewell"]}},
        }
    },
    "hampi": {
        "name": "Hampi, Karnataka", "emoji": "HMP",
        "tagline": "The Ruined Capital of a Lost Empire",
        "monuments": ["Virupaksha Temple","Vittala Temple","Lotus Mahal","Elephant Stables","Hemakuta Hill","Matanga Hill"],
        "budget": {"budget": "Rs.2,000", "mid": "Rs.6,500", "luxury": "Rs.18,000"},
        "days": {
            1: {"title": "Temples & Sacred River",
                "morning":   {"name":"Virupaksha Temple at Dawn","desc":"One of India's oldest continuously functioning temples, with inscriptions dating to the 7th century. This Shaivite temple at the base of Hemakuta Hill was the royal chapel of the Vijayanagara emperors. Climb the 50m gopuram for a panorama over the boulder-strewn landscape. Entry: Free (donations welcome).","tags":["Temple","Living Heritage","Dravidian"]},
                "afternoon": {"name":"Vittala Temple & Stone Chariot","desc":"The crowning achievement of Vijayanagara architecture, built mainly in the 15th–16th centuries. The 56 musical pillars of the main hall emit musical notes when struck. The iconic stone chariot (ratha) in the courtyard is a solid granite temple dedicated to Garuda. Entry: Rs.40 (Indians).","tags":["UNESCO","Vijayanagara","Architecture"]},
                "evening":   {"name":"Tungabhadra River Sunset","desc":"The sacred Tungabhadra river, believed to be formed by the confluence of two rivers in the Mahabharata, flows through the boulder-strewn Hampi valley. Take a traditional coracle (round wicker boat covered in buffalo hide) ride at sunset and watch the ruins turn amber from the water. Cost: ~Rs.50/person.","tags":["Nature","Sunset","Adventure"]}},
            2: {"title": "Royal Enclosure & Hilltops",
                "morning":   {"name":"Matanga Hill Sunrise","desc":"At 535m, Matanga Hill is the highest point in Hampi. The Hindu scriptures reference this hill as the abode of the sage Matanga. Climb ~600 rock-cut steps before dawn for a 360-degree panorama at sunrise — the only spot from which both the Virupaksha Temple and Vittala Temple are visible simultaneously. Entry: Free.","tags":["Photography","Sunrise","Trekking"]},
                "afternoon": {"name":"Royal Enclosure & Lotus Mahal","desc":"The Royal Enclosure covers 59 hectares and contains the Mahanavami Dibba (King's Platform), the Underground Shiva Temple, and the Queen's Bath. The Lotus Mahal within the adjacent Zenana Enclosure is remarkable — its arches blend Saracenic and Dravidian styles in a two-storeyed pavilion. Entry: Rs.40 (Indians).","tags":["Royalty","Architecture","History"]},
                "evening":   {"name":"Hemakuta Hill Temples","desc":"Hemakuta Hill is dotted with over 30 pre-Vijayanagara temples, most dating to the 9th–11th century Chalukya and Rashtrakuta periods. It is sacred to Shiva and sits directly above the Virupaksha Temple complex. The hilltop is one of the finest sunset viewpoints in Hampi. Entry: Free.","tags":["Jain","Sunset","Architecture"]}},
            3: {"title": "Anegundi — The Ancient Capital",
                "morning":   {"name":"Anegundi Village & Kishkinda","desc":"Across the Tungabhadra, Anegundi is the oldest inhabited site in the Hampi region — believed to be Kishkinda, the monkey kingdom of the Ramayana where Rama met Hanuman. The village predates Hampi as the Vijayanagara capital. Walk through banana plantations to the Ranganatha Temple and the ancient Durga Temple. Coracle crossing: Rs.20.","tags":["Ramayana","Heritage","Village"]},
                "afternoon": {"name":"Anjaneya Hill — Hanuman's Birthplace","desc":"A steep climb of 575 steps leads to the hilltop Anjaneya Temple, said to mark the birthplace of Lord Hanuman. The views from the top — a 360° panorama of the Hampi valley, the Tungabhadra, and the boulder-strewn plains — are among the most spectacular in Karnataka. Bring water; the summit has no shade. Entry: Free.","tags":["Temple","Trekking","Views"]},
                "evening":   {"name":"Riverside Bouldering at Sunset","desc":"The Hampi boulder fields are a world-famous destination for climbers — massive granite rocks sculpted by 3,000 million years of erosion. An evening of scrambling on the riverside boulders near Virupapur Gadde (the hippie island), with the ruins glowing behind you, is one of India's most unique outdoor experiences. Cost: Free.","tags":["Adventure","Bouldering","Nature"]}},
            4: {"title": "Underground Temples & Elephant Stables",
                "morning":   {"name":"Underground Shiva Temple & Bhima's Gate","desc":"Hidden within the Royal Enclosure, the partially submerged Underground Shiva Temple is flooded with groundwater up to waist height (depending on season) — devotees wade in to touch the lingam. The massive Bhima's Gate, one of the main entrances to the Vijayanagara capital, is carved from a single granite monolith 8m tall. Entry: Rs.40 (Indians).","tags":["Temple","Vijayanagara","Hidden Gem"]},
                "afternoon": {"name":"Elephant Stables & Zenana Enclosure","desc":"The Elephant Stables are eleven domed chambers in a single 130m-long building where the royal war elephants were housed. The alternating Indo-Islamic domes (some octagonal, some ribbed) reflect the cosmopolitan nature of Vijayanagara court culture. The Zenana Enclosure's watch tower offers the best aerial view of the stables. Entry: Rs.40 (Indians).","tags":["Architecture","Vijayanagara","History"]},
                "evening":   {"name":"Hampi Bazaar & Archaeological Survey","desc":"The Hampi Bazaar, stretching 750m from the Virupaksha Temple, was once the main commercial street of a city of 500,000 people — larger than Rome at the same period. The Archaeological Museum (Kamalapur) near the site displays Vijayanagara bronzes, coins, and a remarkable scale model of the entire Hampi complex. Entry: Rs.25 (Indians).","tags":["History","Museum","Heritage"]}},
            5: {"title": "Badami — Chalukya Cave Temples",
                "morning":   {"name":"Badami Cave Temples","desc":"130km from Hampi, Badami was the capital of the early Chalukya dynasty (543–757 CE), who carved four rock-cut cave temples into a red sandstone cliff overlooking Agastya Lake. Cave 3 has the finest sculptures — an 18-armed Vishnu as Trivikrama and the great Narasimha relief are masterpieces of early medieval Indian art. Entry: Rs.40 (Indians).","tags":["Chalukya","Cave Temple","UNESCO"]},
                "afternoon": {"name":"Badami Fort & Bhutanatha Temples","desc":"The Badami Fort atop the sandstone cliff houses several Shiva temples and reservoirs and offers extraordinary views over the red rock gorge and Agastya Lake below. The lakeside Bhutanatha Group of temples (7th–11th century) rises directly from the water — some submerged during the monsoon — in one of Karnataka's most photogenic settings. Entry: Rs.25 (Indians).","tags":["Fort","Temple","Photography"]},
                "evening":   {"name":"Pattadakal — World Heritage Temples","desc":"22km from Badami, Pattadakal (UNESCO) was the ceremonial capital of the Chalukya dynasty where kings were crowned. Its 10 temples (7th–8th century) represent the complete transition from early Dravidian to Nagara architecture. The Virupaksha Temple (733 CE) is a direct ancestor of the Kailasa Temple at Ellora. Entry: Rs.40 (Indians).","tags":["UNESCO","Chalukya","Architecture"]}},
            6: {"title": "Aihole — Cradle of Indian Temple Architecture",
                "morning":   {"name":"Aihole Temple Complex","desc":"35km from Badami, Aihole contains over 125 temples from the 6th–12th century — earning it the title 'Cradle of Indian Temple Architecture.' The Durga Temple (7th century) has an apsidal (semicircular) plan unique in India, modelled on Buddhist chaitya halls. The Ladkhan Temple (5th century) is one of the earliest surviving post-Gupta temples. Entry: Rs.40 (Indians).","tags":["Chalukya","Architecture","UNESCO"]},
                "afternoon": {"name":"Mahakuta Temple Cluster","desc":"10km from Badami, the Mahakuta group of Chalukya temples (7th century) around a sacred spring and tank represents a complete medieval temple complex still in active worship. The Mahakuta Temple's ornate shikhara and the flowing spring are unchanged from the Chalukya period. Virtually no other tourists visit. Entry: Free.","tags":["Temple","Chalukya","Hidden Gem"]},
                "evening":   {"name":"Return to Hampi at Dusk","desc":"Drive back to Hampi through the Deccan plateau landscape — red soil, sugarcane fields, and the fading golden light catching the ancient volcanic rock of the Aravalli-Deccan boundary. Dinner at a riverside café on the Tungabhadra with views of the illuminated Virupaksha Temple gopuram.","tags":["Nature","Drive","Heritage"]}},
            7: {"title": "Hospet & Tungabhadra Dam",
                "morning":   {"name":"Tungabhadra Dam & Reservoir","desc":"13km from Hampi, the Tungabhadra Dam (1953) was one of independent India's first major irrigation projects. The reservoir stretches for 65km and is part of the same river that the Vijayanagara Empire diverted for its irrigation canals in the 14th century — still visible as a network of aqueducts and channels throughout the archaeological zone. Entry: Free.","tags":["Engineering","Nature","History"]},
                "afternoon": {"name":"Daroji Sloth Bear Sanctuary","desc":"60km from Hampi, the Daroji Sloth Bear Sanctuary is the world's only sanctuary specifically created for sloth bears. The rugged landscape of granite boulders and lantana scrub is home to 120+ sloth bears, which are routinely seen at the feeding platform near the viewing area in the afternoon. Entry: Rs.200 (Indians).","tags":["Wildlife","Bears","Nature"]},
                "evening":   {"name":"Kampli Fort & Tungabhadra Crossing","desc":"The ruins of Kampli Fort, 15km from Hospet, mark a decisive moment in Vijayanagara history — it was at Kampli that the founders of the Vijayanagara Empire (the Sangama brothers) were initially imprisoned before escaping and founding Hampi. The riverside crossing at sunset is atmospheric. Entry: Free.","tags":["History","Fort","Vijayanagara"]}},
            8: {"title": "Hampi's Water Architecture",
                "morning":   {"name":"Pushkarani Stepped Tank & Waterworks","desc":"The Vijayanagara Empire built an extraordinary hydraulic system — 25km of stone aqueducts, 16 major tanks, and hundreds of canals — to supply water to a city of half a million in the semi-arid Deccan. The Pushkarani stepped tank near the Royal Enclosure and the Queen's Bath hydraulic system are the finest surviving examples. Entry: Free.","tags":["Engineering","History","Architecture"]},
                "afternoon": {"name":"Hampi Utsav Archaeological Walk","desc":"A guided 5km walking tour through the Archaeological Survey's protected zone covering sites most visitors miss: the Ganagitti Jain Temple (1385), the Kadlekalu Ganesha (the largest Ganesha figure in the region, carved from a single boulder), and the Sasivekalu Ganesha with its remarkable naga (serpent) base. Entry: Rs.40 (Indians).","tags":["Jain","Temple","Walking"]},
                "evening":   {"name":"Coracle Ride at Sunset","desc":"A final coracle ride on the Tungabhadra at sunset — gliding silently on the river as the ruins of the world's second-richest medieval city glow orange around you. The UNESCO World Heritage landscape of boulder-strewn hills, ruined palaces, and river-crossing fishermen at dusk is Hampi's most unforgettable image. Cost: Rs.50/person.","tags":["Sunset","River","Photography"]}},
            9: {"title": "Bijapur — Gol Gumbaz & Adil Shahi Tombs",
                "morning":   {"name":"Gol Gumbaz, Bijapur","desc":"200km from Hampi, the Gol Gumbaz (1656) is the mausoleum of Adil Shah II and the second largest dome in the world — only St Peter's in Rome is larger. The 44m dome rests on a 47m-wide base without intermediate support. The Whispering Gallery inside the circular drum is acoustically extraordinary — a whisper can be heard clearly 37m away. Entry: Rs.25 (Indians).","tags":["Adil Shahi","Architecture","UNESCO"]},
                "afternoon": {"name":"Ibrahim Rauza & Bijapur Monuments","desc":"The Ibrahim Rauza (1626) is Bijapur's most refined monument — the tomb of Sultan Ibrahim II and his queen, surrounded by a formal garden. Its slender minarets and delicate stonework are believed to have inspired elements of the Taj Mahal. The city also contains the Jami Masjid (1576), the largest mosque in the Deccan. Entry: Rs.25 (Indians).","tags":["Adil Shahi","Mughal","Architecture"]},
                "evening":   {"name":"Bijapur Fort & Malik-e-Maidan Cannon","desc":"Bijapur Fort's most remarkable object is the Malik-e-Maidan (Lord of the Plains) — the largest medieval cannon in the world, weighing 55 tonnes and cast in 1549. It required 10 elephants, 400 oxen, and hundreds of soldiers to move. The cannon has never been fired at Bijapur — only test-fired once in Ahmadnagar. Entry: Rs.25 (Indians).","tags":["Fort","Cannon","History"]}},
            10: {"title": "Hampi Farewell — Boulder Sunrise",
                "morning":   {"name":"Final Sunrise on Hampi Boulders","desc":"Wake before dawn for a final climb onto the granite boulders east of the Virupaksha Temple. Watch the sun rise across the Vijayanagara plain — the ruins emerging from the darkness one by one, the Tungabhadra glinting silver, the monkeys descending from the boulders, and the first bells of the Virupaksha Temple ringing in the valley below. Entry: Free.","tags":["Sunrise","Photography","Farewell"]},
                "afternoon": {"name":"Kamalapura Archaeological Museum","desc":"The ASI Museum at Kamalapura houses the finest artefacts from the Vijayanagara excavations: bronze Natarajas, copper coins, stone inscriptions, and a detailed scale model of the entire 26 sq km archaeological zone. The collection contextualises everything you've seen across 10 days. Entry: Rs.25 (Indians).","tags":["Museum","Archaeology","History"]},
                "evening":   {"name":"Farewell Dinner — North Karnataka Cuisine","desc":"North Karnataka has a distinctive cuisine rarely found outside the region: jolada rotti (jowar flatbread), ennegayi (stuffed brinjal curry), holige (sweet stuffed flatbread), and the uniquely spiced Dharwad pedha sweet. Dine at a local dhaba in Hospet for an authentic farewell to the Deccan. Cost: Rs.100–300.","tags":["Food","Culture","Farewell"]}},
        }
    },
    "varanasi": {
        "name": "Varanasi, Uttar Pradesh", "emoji": "VNS",
        "tagline": "The Eternal City on the Ganges",
        "monuments": ["Kashi Vishwanath Temple","Dashashwamedh Ghat","Sarnath","Ramnagar Fort","Manikarnika Ghat","Assi Ghat"],
        "budget": {"budget": "Rs.2,500", "mid": "Rs.7,500", "luxury": "Rs.22,000"},
        "days": {
            1: {"title": "The Ghats & Sacred Fire",
                "morning":   {"name":"Sunrise Boat Ride on the Ganges","desc":"Varanasi's 84 ghats stretch for 6.5km along the western bank of the Ganges. A sunrise boat ride (cost ~Rs.300–500) reveals the full spectacle of daily Hindu life — priests, pilgrims bathing, yoga practitioners, and the burning pyres of Manikarnika Ghat. This is considered one of the world's oldest continuously inhabited cities.","tags":["Sacred","Photography","Culture"]},
                "afternoon": {"name":"Kashi Vishwanath Temple","desc":"One of the 12 Jyotirlingas (sacred abodes of Lord Shiva) and among the most important temples in Hinduism. The temple has been rebuilt multiple times after destruction; the current golden spire structure was commissioned by Maratha queen Ahilyabai Holkar in 1780. The corridor was newly expanded in 2021. Entry: Free.","tags":["Temple","Hindu","History"]},
                "evening":   {"name":"Ganga Aarti at Dashashwamedh Ghat","desc":"The grandest of Varanasi's 84 ghats, Dashashwamedh is where Lord Brahma is said to have performed the Dasa Ashwamedh yagna (10-horse sacrifice). Every evening at dusk, a team of priests simultaneously perform a choreographed fire ritual with large brass diyas, conch shells, and Sanskrit chanting — drawing thousands of devotees and visitors.","tags":["Ritual","Fire","Sacred"]}},
            2: {"title": "Sarnath & the Old City",
                "morning":   {"name":"Sarnath — The Deer Park","desc":"Located 10km north of Varanasi, Sarnath is one of the four most sacred Buddhist pilgrimage sites. It was here, in the Isipatana Deer Park, that the Buddha delivered his first sermon (the Dhammacakkappavattana Sutta) in 5th century BCE. The Dhamek Stupa (500 CE) marks the exact spot. Entry: Rs.25 (Indians).","tags":["Buddhist","UNESCO","History"]},
                "afternoon": {"name":"Ramnagar Fort & Museum","desc":"Built in the 18th century by Maharaja Balwant Singh, this cream-coloured sandstone fort across the Ganges is the ancestral seat of the Maharaja of Varanasi. The Veda Vyasa Museum inside houses a remarkable collection of medieval weapons, royal palanquins, vintage automobiles, and rare astronomical instruments.","tags":["Fort","Royalty","Museum"]},
                "evening":   {"name":"Banarasi Silk Weaving Quarter","desc":"Varanasi's silk weaving tradition dates back over 2,000 years. The city's weavers — concentrated in the Peeli Kothi and Madanpura neighbourhoods — produce the celebrated Banarasi silk saris, known for their gold and silver brocade (zari) work. A guided workshop visit reveals the intricate handloom process. Cost: Free–Rs.200.","tags":["Craft","Culture","Shopping"]}},
            3: {"title": "The 84 Ghats — A Complete Walk",
                "morning":   {"name":"Manikarnika Ghat — The Burning Ghat","desc":"Manikarnika is one of the most sacred cremation sites in Hinduism — Hindus believe dying here guarantees moksha (liberation from the cycle of rebirth). Fires have burned here continuously for 3,500 years according to tradition. The ghat is managed by the Dom community, hereditary keepers of the cremation fires. Approach with deep respect and no cameras. Entry: Free.","tags":["Sacred","Hindu","Culture"]},
                "afternoon": {"name":"Assi to Raj Ghat — Full Ghat Walk","desc":"Walk the 6.5km stretch of all 84 ghats from Assi Ghat (south) to Raj Ghat (north) — an extraordinary 3-hour journey through every dimension of Varanasi life. Each ghat has a distinct character: Harishchandra (a second burning ghat), Kedar Ghat (South Indian pilgrims), Darbhanga Ghat (a palace), and Panchaganga Ghat (five sacred rivers).","tags":["Walking","Culture","Heritage"]},
                "evening":   {"name":"Evening Aarti at Multiple Ghats","desc":"Beyond Dashashwamedh, smaller aartis happen simultaneously at Assi Ghat (more intimate, popular with locals), Rajendra Prasad Ghat, and Panchaganga Ghat. Witnessing the same ancient ritual at three different locations on the same evening gives a sense of the city's layered spiritual geography. Cost: Free.","tags":["Ritual","Sacred","Photography"]}},
            4: {"title": "Varanasi's Old City Lanes",
                "morning":   {"name":"Vishwanath Gali & Temple Trail","desc":"The narrow lanes (galis) of Varanasi's old city are medieval in scale — too narrow for cars, they are a labyrinth of temples, ashrams, sweet shops, and flower sellers. The Vishwanath Gali leading to the Kashi Vishwanath Temple passes through 2,000 years of continuous urban occupation. Hire a local guide for the full experience. Cost: Guide Rs.500–1,000.","tags":["Walking","Temple","Heritage"]},
                "afternoon": {"name":"Bharat Mata Temple & Hindi Language Museum","desc":"The Bharat Mata Temple (1936) is unique in India — instead of a deity, it enshrines a marble relief map of undivided India, including mountains and rivers carved to scale. Nearby, the Nagari Pracharini Sabha (Hindi Promotion Society, 1893) is the institution that standardised the Devanagari script used for Hindi, Sanskrit, and Marathi. Entry: Free.","tags":["Culture","Language","History"]},
                "evening":   {"name":"Classical Music of Banaras Gharana","desc":"Varanasi is one of the great seats of Hindustani classical music. The Banaras Gharana of tabla and dhrupad singing has produced masters from Bismillah Khan to Ravi Shankar. An evening concert at the Sankat Mochan Music Festival (spring) or at a private concert hall on the ghats offers a connection to a tradition inseparable from the city itself.","tags":["Music","Culture","Tradition"]}},
            5: {"title": "Bodhgaya — Where Buddha Attained Enlightenment",
                "morning":   {"name":"Bodhi Tree & Mahabodhi Temple, Bodhgaya","desc":"250km from Varanasi (4 hours by car), Bodhgaya is the most sacred site in Buddhism — where Siddhartha Gautama attained enlightenment under the Bodhi Tree in 528 BCE. The Mahabodhi Temple (UNESCO, rebuilt 5th century CE) marks the exact spot. The current Bodhi Tree is a direct descendant of the original. Entry: Free.","tags":["UNESCO","Buddhist","Sacred"]},
                "afternoon": {"name":"Rajgir & Nalanda University Ruins","desc":"80km from Bodhgaya, Nalanda was the greatest university of the ancient world (5th–12th century CE) — at its peak housing 10,000 monks and scholars from across Asia. The Afghan ruler Bakhtiyar Khilji burned its library of 9 million manuscripts in 1193, destroying irreplaceable texts. The excavated ruins span 14 hectares. Entry: Rs.25 (Indians).","tags":["Buddhist","University","History"]},
                "evening":   {"name":"Return via Rajgir Hot Springs","desc":"Rajgir, 12km from Nalanda, is where the Buddha spent several rainy seasons meditating. The Gridhrakuta (Vulture Peak Hill) is where he gave many of his great discourses. The town has natural sulphur hot springs in which pilgrims bathe. Return to Varanasi overnight. Cost: Hot spring Rs.50.","tags":["Buddhist","Sacred","Wellness"]}},
            6: {"title": "Chunar Fort & Vindhya Hills",
                "morning":   {"name":"Chunar Fort on the Ganges","desc":"23km from Varanasi, Chunar Fort stands on a strategic promontory above the Ganges at the point where the Vindhya Hills meet the river. It was a key Mughal garrison and briefly held by Sher Shah Suri. The fort's most distinctive feature is the famous Chunar sandstone quarry — the same stone used to build the Allahabad Fort and many Mughal monuments. Entry: Free.","tags":["Fort","Mughal","History"]},
                "afternoon": {"name":"Vindhyachal Temple & Kali Khoh","desc":"8km from Mirzapur (45km from Varanasi), the Vindhyachal Shakti Peetha is one of the most important goddess pilgrimage sites in North India. The Vindhyavasini Temple, Ashtabhuja Temple, and Kali Khoh Temple form a sacred triangle visited by millions annually. The cave temple of Kali Khoh (the Dark Cave) is particularly atmospheric. Entry: Free.","tags":["Shakti","Pilgrimage","Sacred"]},
                "evening":   {"name":"Varanasi Rooftop — The River by Night","desc":"Return to Varanasi for an evening on a ghat-side rooftop restaurant. The Ganges at night is extraordinary — burning ghats glowing orange in the distance, diyas (oil lamps) floating downstream, the calls of the evening prayer mixing with boat engines. The best rooftop views are from Meer Ghat and Munshi Ghat. Cost: Rs.300–800.","tags":["Atmosphere","Photography","Food"]}},
            7: {"title": "Allahabad — Confluence of Sacred Rivers",
                "morning":   {"name":"Triveni Sangam, Prayagraj","desc":"130km from Varanasi, Prayagraj (Allahabad) is the site of the Triveni Sangam — the sacred confluence of the Ganges, Yamuna, and the mythical underground Saraswati river. Bathing here during the Kumbh Mela (held every 12 years, next in 2025) is considered the holiest act in Hinduism. Even outside Kumbh, a boat to the confluence is a profound experience. Cost: Boat Rs.200–500.","tags":["Hindu","Sacred","Pilgrimage"]},
                "afternoon": {"name":"Allahabad Fort & Ashoka Pillar","desc":"The Allahabad Fort (1583), built by Emperor Akbar, contains within it the Ashoka Pillar (232 BCE) — one of the finest surviving Mauryan pillars. The fort is an active military cantonment, so access is restricted; the Akshaya Vata (immortal banyan tree), said to be 5,000 years old, can be visited separately. Entry: Restricted.","tags":["Mughal","Mauryan","History"]},
                "evening":   {"name":"Khusrau Bagh Mughal Tombs","desc":"Khusrau Bagh contains the magnificent garden tombs of Emperor Jahangir's son Khusrau (1622), his Rajput mother Shah Begum (1616), and his sister Nithar Begum. The red sandstone and marble mausoleums in a walled Mughal garden are remarkably intact and virtually unvisited. Entry: Free.","tags":["Mughal","Garden Tomb","Hidden Gem"]}},
            8: {"title": "Varanasi Craft Deep Dive",
                "morning":   {"name":"Toy Town — Varanasi's Wooden Toys","desc":"Varanasi is one of India's most important centres for wooden toy making — a tradition going back to the Mughal era when toys were made for the royal children of Avadh. The Madanpura and Pilikothi neighbourhoods are full of workshops turning out painted wooden animals, dolls, and kitchen sets. A guided workshop tour lets you watch and participate. Cost: Free–Rs.300.","tags":["Craft","Art","Culture"]},
                "afternoon": {"name":"Banaras Hindu University & Bharat Kala Bhavan","desc":"BHU (founded 1916) is one of Asia's largest residential universities and its campus contains the Bharat Kala Bhavan — a world-class museum of Indian art and archaeology. The collection of over 100,000 objects includes Mughal miniatures, Banaras brocades, and one of India's most significant collections of Hindu temple sculpture. Entry: Rs.20 (Indians).","tags":["Museum","University","Art"]},
                "evening":   {"name":"Sunset at Assi Ghat & Subah-e-Banaras","desc":"The Subah-e-Banaras (Morning of Banaras) is a cultural programme held at Assi Ghat at sunrise, combining yoga, devotional music, and the watching of the sun rise over the Ganges — a newer cultural event that has become an institution. The same ghat is equally beautiful at evening as sadhus, students, and boatmen gather around the ancient banyan tree. Cost: Free.","tags":["Culture","Sacred","Photography"]}},
            9: {"title": "Mirzapur & Carpet Weaving Heritage",
                "morning":   {"name":"Mirzapur Carpet Weaving Centres","desc":"Mirzapur-Bhadohi (75km from Varanasi) produces over 80% of India's hand-knotted carpet exports. The tradition was established in the Mughal era and the region now employs over one million weavers. A guided tour of a weaving centre shows the entire process from wool dyeing (with natural vegetable dyes) to hand-knotting a single knot at a time, at a rate of 3,000 knots per sq inch. Cost: Free.","tags":["Craft","Carpet","Heritage"]},
                "afternoon": {"name":"Vindhya Waterfalls — Rajdari & Devdari","desc":"55km from Varanasi in the Vindhya hills, the Rajdari and Devdari waterfalls descend through dense forest in a nature reserve. The monsoon months (July–September) are the most spectacular, but the streams run year-round. The forest walk from the car park passes through teak, mahua, and tendu groves. Entry: Rs.25 (Indians).","tags":["Nature","Waterfall","Adventure"]},
                "evening":   {"name":"Varanasi Ghee Laddu & Farewell at Ghat","desc":"Varanasi's Sankat Mochan Temple produces a uniquely spiced ghee-based laddu sweet that has been offered to Hanuman here for centuries. Buy a box to take home. End your evening on Dashashwamedh Ghat for a final Ganga Aarti — the ancient ritual feels newer and more miraculous every time. Cost: Rs.50–200.","tags":["Food","Sacred","Farewell"]}},
            10: {"title": "Final Varanasi — Dawn to Departure",
                "morning":   {"name":"Pre-dawn Dip at Har Ki Pauri","desc":"A final pre-dawn boat ride to watch Varanasi wake — the ritual unchanged for millennia. Watch the pandas (hereditary priests) perform the morning sandhya vandana on the ghats, the flower boats float with diyas, and the first light turn the water to molten gold. This is the image of Varanasi that stays forever. Cost: Boat Rs.300–500.","tags":["Sacred","Photography","Sunrise"]},
                "afternoon": {"name":"Kashi Archaeological Museum","desc":"The small but significant Kashi Archaeological Museum in Varanasi displays pre-Mauryan and Mauryan artefacts excavated from the Rajghat mound at the northern end of the city — one of the most continuously inhabited urban sites in South Asia, with occupation layers going back 3,000 years. Entry: Rs.10 (Indians).","tags":["Archaeology","Museum","History"]},
                "evening":   {"name":"Final Evening Aarti & Departure","desc":"Leave Varanasi as it has always been left — from the river. Take a final evening boat ride during the Aarti, watching the ceremony from the water as thousands of diyas float past. The sound of the conch shells, the smell of incense, and the light on the ancient stone ghats is an experience that cannot be replicated anywhere else on earth. Cost: Boat Rs.300–500.","tags":["Sacred","Farewell","Photography"]}},
        }
    },
    "lucknow": {
        "name": "Lucknow, Uttar Pradesh", "emoji": "LKO",
        "tagline": "City of Nawabs, Poetry & Kebabs",
        "monuments": ["Bara Imambara","Chota Imambara","Rumi Darwaza","British Residency","Chattar Manzil","Clock Tower"],
        "budget": {"budget": "Rs.2,200", "mid": "Rs.7,000", "luxury": "Rs.20,000"},
        "days": {
            1: {"title": "Nawabi Splendour",
                "morning":   {"name":"Bara Imambara & Bhulbhulaiya","desc":"Built in 1784 by Nawab Asaf-ud-Daula to provide famine relief, this vast Shia shrine has the largest arched hall in the world not supported by beams. Inside is the famous Bhulbhulaiya — a labyrinthine rooftop maze of 489 identical corridors. Entry: Rs.25 (Indians).","tags":["Nawabi","Shia","Architecture"]},
                "afternoon": {"name":"Rumi Darwaza & Chota Imambara","desc":"The 18m Rumi Darwaza (1784), modelled on a gate in Constantinople, is the iconic symbol of Lucknow. Nearby, the Chota Imambara (1838) is a mausoleum and ceremonial hall dazzling with chandeliers, Belgian glass, and calligraphy. Entry: Rs.25 (Indians).","tags":["Nawabi","Mughal","History"]},
                "evening":   {"name":"Tunday Kababi & Hazratganj","desc":"Lucknow's galouti kebab was invented in the 19th century for a toothless Nawab — so tender it melts on the tongue. Tunday Kababi, founded in 1905, is the city's most celebrated kebab house. Stroll Hazratganj, Lucknow's colonial-era shopping boulevard, in the evening.","tags":["Food","Colonial","Culture"]}},
            2: {"title": "The British Residency & Nawabi Culture",
                "morning":   {"name":"British Residency Ruins","desc":"The Residency complex was the scene of the 87-day Siege of Lucknow during the 1857 uprising. The ruins, still pockmarked with cannonball scars, contain over 2,000 graves of British and Indian soldiers within the compound. The shell-damaged clock tower stopped at 4:53pm — and was never restarted. Entry: Rs.25 (Indians).","tags":["Colonial","1857","History"]},
                "afternoon": {"name":"State Museum Lucknow","desc":"Housed in a grand colonial building, this museum contains one of India's finest collections of terracotta sculptures, Kushana-period statues, ancient coins, and Mughal miniature paintings — many excavated from nearby Kanauj and Sarnath.","tags":["Museum","Archaeology","Art"]},
                "evening":   {"name":"Kathak Recital & Mughlai Dinner","desc":"Lucknow is considered the birthplace of the Lucknow gharana of Kathak dance, developed under the patronage of Nawab Wajid Ali Shah in the 19th century. Attend a classical performance at a cultural centre, then dine on Lucknowi biryani, korma, and sheermal (saffron flatbread).","tags":["Dance","Culture","Food"]}},
            3: {"title": "Lucknow's Living Tehzeeb",
                "morning":   {"name":"Chattar Manzil & Clock Tower","desc":"The Chattar Manzil (Umbrella Palace, 1818) on the Gomti riverbank was the pleasure pavilion of the Nawabs. The adjacent 67m Husainabad Clock Tower (1887) — inspired by Big Ben — was built to mark the arrival of the first Lieutenant Governor of Awadh. Entry: Rs.25 (Indians).","tags":["Nawabi","Architecture","Heritage"]},
                "afternoon": {"name":"Hazratganj & Gomti Riverfront","desc":"Hazratganj, Lucknow's colonial-era main street built in the 1810s, retains its arcaded Nawabi buildings. The Gomti Riverfront promenade (recently restored) traces the river past the ruins of Saadat Ali Khan's palace and the Gomti Nagar ghat. Cost: Free.","tags":["Colonial","Shopping","Heritage"]},
                "evening":   {"name":"Awadhi Food Walk — Chowk & Aminabad","desc":"A dedicated Awadhi food walk through Aminabad and Chowk — trying nihari, sheermal, roomali roti, and the legendary Lucknow dum biryani. The Chowk area has unbroken food culture going back to the Nawabi courts. Cost: Rs.300–600/person.","tags":["Food","Culture","Heritage"]}},
            4: {"title": "Dudhwa & Terai Wildlife",
                "morning":   {"name":"Dudhwa Tiger Reserve — Jeep Safari","desc":"235km from Lucknow on the Nepal border, Dudhwa National Park is home to tigers, one-horned rhinos (reintroduced in 1984), swamp deer, and leopards. The terai forest at dawn is extraordinarily atmospheric. Entry: Rs.400 (Indians).","tags":["Tiger","Wildlife","Nature"]},
                "afternoon": {"name":"Kishanpur Wildlife Sanctuary","desc":"Adjacent to Dudhwa, Kishanpur is particularly known for its swamp deer (barasingha) population — once nearly extinct, now recovering. The tall elephant-grass meadows interspersed with sal forest are ideal barasingha habitat. Entry: Rs.200 (Indians).","tags":["Wildlife","Deer","Nature"]},
                "evening":   {"name":"Return via Awadh's Villages","desc":"The drive back from Dudhwa passes through the agricultural heartland of Awadh — the terai sugarcane and wheat fields that the Nawabs of Avadh taxed and governed. Village stops for fresh sugarcane juice are part of the experience. Cost: Free.","tags":["Village","Nature","Culture"]}},
            5: {"title": "Ayodhya — Birthplace of Lord Rama",
                "morning":   {"name":"Ram Janmabhoomi Temple","desc":"130km from Lucknow, Ayodhya is one of the seven sacred Hindu cities. The new Ram Mandir (inaugurated 2024) stands on the site believed to be the birthplace of Lord Rama, designed in the Nagara style in pink Rajasthani sandstone. Entry: Free.","tags":["Hindu","Sacred","Pilgrimage"]},
                "afternoon": {"name":"Hanuman Garhi & Kanak Bhavan","desc":"Hanuman Garhi (18th century) is a fort-temple atop a mound reached by 76 steps, housing a 6-metre Hanuman idol. Kanak Bhavan has golden idols of Rama and Sita in one of North India's most lavishly decorated temples. Entry: Free.","tags":["Temple","Hindu","Heritage"]},
                "evening":   {"name":"Sarayu River Aarti","desc":"The evening aarti on the banks of the Sarayu river — the sacred river of the Ramayana — is performed at Ram ki Paidi ghat with oil lamps and chanting priests as the sun sets over the river. Entry: Free.","tags":["Ritual","Sacred","Photography"]}},
            6: {"title": "Agra Day Trip",
                "morning":   {"name":"Taj Mahal at Golden Hour","desc":"Take the early Lucknow–Agra express (4.5 hrs) for Agra. Arrive for the Taj Mahal at midmorning — the marble glows brilliantly in midday light and the crowds are manageable by 11am. Entry: Rs.50 (Indians). Train: Rs.400–600.","tags":["UNESCO","Taj Mahal","Travel"]},
                "afternoon": {"name":"Agra Fort & Fatehpur Sikri","desc":"Agra Fort (Rs.35) is a 2-hour experience of Mughal power. If time allows, drive 40km to Fatehpur Sikri — Akbar's abandoned sandstone capital with the world's largest gateway, the 54m Buland Darwaza. Entry: Rs.35 (Indians).","tags":["UNESCO","Mughal","Heritage"]},
                "evening":   {"name":"Petha Shopping & Return to Lucknow","desc":"Agra's famous petha (white translucent ash-gourd sweet) shops in Subhash Bazaar make excellent take-home gifts. Return to Lucknow by evening train. Cost: Rs.200–400.","tags":["Food","Shopping","Travel"]}},
            7: {"title": "Kanpur Heritage",
                "morning":   {"name":"Jain Glass Temple, Kanpur","desc":"80km from Lucknow, the Jain Glass Temple in Kanpur is covered entirely in mosaic glass and mirrors — walls, floors, ceiling. Built in 1956, it houses beautiful Jain marble images. Entry: Rs.10 (Indians).","tags":["Jain","Art","Architecture"]},
                "afternoon": {"name":"1857 Memorial & All Souls Church","desc":"Kanpur's Nana Rao Park contains the reconstructed memorial well and the gothic All Souls' Memorial Church (1875) built for the 1857 Cawnpore massacre victims. Entry: Free.","tags":["1857","Colonial","History"]},
                "evening":   {"name":"Kanpur Leather Market & Thaggu ke Laddu","desc":"Kanpur has been India's leather capital since the 19th century. The famous thaggu ke laddu sweet shop has been making pure-ghee laddus for generations. Cost: Rs.200–500.","tags":["Shopping","Craft","Food"]}},
            8: {"title": "Constantia & Nawabi Architecture Walk",
                "morning":   {"name":"La Martiniere College (Constantia)","desc":"Built by French soldier Major General Claude Martin (1795–1840), Constantia is one of India's most unusual buildings — baroque towers, Mughal chattris, and neo-classical columns. Martin's tomb is in the basement. Now a functioning school. Entry: Rs.100 (pre-arranged).","tags":["Colonial","Architecture","History"]},
                "afternoon": {"name":"Lucknow Zoo & Dilkusha Palace","desc":"The Nawab's Zoo (1921) is set in the grounds of the Dilkusha Palace (1800) — a mock-Tudor hunting lodge whose ruins are hauntingly beautiful. The 1857 battle scars on the walls make it historically layered. Entry: Rs.25 (Indians).","tags":["Colonial","Nawabi","Nature"]},
                "evening":   {"name":"Imambara Sound & Light Show","desc":"The Bara Imambara runs an evening Sound and Light show narrating Nawab Asaf-ud-Daula's famine relief construction of 1784. The illuminated Rumi Darwaza and Imambara at night are extraordinarily dramatic. Entry: Rs.100–200 (Indians).","tags":["Heritage","Nawabi","Culture"]}},
            9: {"title": "Allahabad — Triveni Sangam",
                "morning":   {"name":"Triveni Sangam, Prayagraj","desc":"130km from Lucknow, the Triveni Sangam is the sacred confluence of Ganges, Yamuna, and the mythical Saraswati. A boat to the confluence at dawn (Rs.300–500) is a profound experience for Hindu pilgrims and heritage travellers alike.","tags":["Hindu","Sacred","Pilgrimage"]},
                "afternoon": {"name":"Allahabad Fort & Ashoka Pillar","desc":"The Allahabad Fort (1583), built by Akbar, contains the Ashoka Pillar (232 BCE) — a fine Mauryan column. The Khusrau Bagh has magnificent garden tombs of Mughal prince Khusrau and his mother, virtually unvisited. Entry: Restricted/Rs.25.","tags":["Mughal","Mauryan","Heritage"]},
                "evening":   {"name":"Prayagraj Museum & Return","desc":"The Prayagraj Museum houses outstanding Buddhist, Hindu, and Mughal artefacts from the Gangetic plain. Return to Lucknow by evening. Entry: Rs.10 (Indians).","tags":["Museum","Archaeology","History"]}},
            10: {"title": "Farewell Lucknow — Chowk to Charbagh",
                "morning":   {"name":"Gomti Riverfront Walk","desc":"A final morning walk along the restored Gomti riverfront — the Chattar Manzil reflecting in the river, flower sellers on the ghats, and the soft morning light on the Nawabi skyline. Cost: Free.","tags":["Heritage","Walking","Photography"]},
                "afternoon": {"name":"1857 Trail — Sikander Bagh & Shah Najaf","desc":"Complete the 1857 trail: Sikander Bagh (where 2,000 rebels fell), and Shah Najaf Imambara — used as a British garrison during the Siege and still bearing cannonball damage. Entry: Free–Rs.25.","tags":["1857","History","Heritage"]},
                "evening":   {"name":"Farewell Biryani at Idris","desc":"Idris Biryani in Akbari Gate has been serving Lucknowi dum biryani since 1935 — rice and mutton slow-cooked in sealed earthen pot (dum pukht) for 6 hours, fragrant with kewra water and saffron. Cost: Rs.200–400.","tags":["Food","Nawabi","Farewell"]}},
        }
    },
    "amritsar": {
        "name": "Amritsar, Punjab", "emoji": "AMR",
        "tagline": "The Holy City of the Golden Temple",
        "monuments": ["Golden Temple","Jallianwala Bagh","Wagah Border","Akal Takht","Partition Museum","Ram Bagh"],
        "budget": {"budget": "Rs.2,000", "mid": "Rs.6,500", "luxury": "Rs.18,000"},
        "days": {
            1: {"title": "The Golden Temple & Sacred Sikh Heritage",
                "morning":   {"name":"Golden Temple at Dawn","desc":"The Harmandir Sahib (Golden Temple), the holiest shrine in Sikhism, sits in the sacred Amrit Sarovar (Pool of Nectar) — from which the city takes its name. Founded in 1577 by the fourth Sikh Guru, Ram Das, the current gold-plated structure (using 750kg of pure gold) was completed in 1830. The Langar (community kitchen) feeds 100,000 people daily, free of charge. Entry: Free.","tags":["Sikh","Sacred","Gold"]},
                "afternoon": {"name":"Akal Takht & Sikh Museum","desc":"The Akal Takht (Throne of the Timeless One), built in 1606 by Guru Hargobind, is the highest seat of temporal Sikh authority. It faces the Golden Temple across the sarovar. Within the complex, the Sikh Reference Library and the Central Sikh Museum display historical weapons, manuscripts, and portraits of the Ten Sikh Gurus.","tags":["Sikh","History","Museum"]},
                "evening":   {"name":"Palki Sahib Ceremony","desc":"Every evening, the Guru Granth Sahib (the holy scripture) is ceremonially carried from the Golden Temple to the Akal Takht in a golden palanquin (palki) accompanied by devotional kirtan and thousands of worshippers. This procession, called the Sukhasan ceremony, occurs at sunset and is deeply moving to witness.","tags":["Ritual","Sikh","Sacred"]}},
            2: {"title": "History, Partition & the Border",
                "morning":   {"name":"Jallianwala Bagh Memorial","desc":"On 13 April 1919, British General Dyer ordered troops to open fire on a peaceful gathering of over 20,000 people in this walled garden, killing at least 379 (by British estimates) to over 1,000 (Indian estimates). The bullet marks remain visible in the walls. A moving museum and eternal flame mark the site. Entry: Free.","tags":["Independence","Tragedy","History"]},
                "afternoon": {"name":"Partition Museum","desc":"Opened in 2017, this is the world's first museum dedicated to the 1947 Partition of India, which uprooted 14–18 million people and caused up to 2 million deaths. Built in the historic Town Hall, it uses personal testimonies, photographs, and artefacts to document one of history's largest mass migrations.","tags":["Museum","Partition","Heritage"]},
                "evening":   {"name":"Wagah Border Beating Retreat","desc":"The Wagah-Attari border ceremony, held daily at sunset between India and Pakistan, is a theatrical display of military pageantry involving high-kicking soldiers from the Indian Border Security Force and Pakistani Rangers. Over 15,000 spectators attend on the Indian side. Arrive 90 minutes early for a good seat. Entry: Free.","tags":["Military","Patriotic","Culture"]}},
            3: {"title": "Sikh History & Sacred Sites",
                "morning":   {"name":"Golden Temple — Second Visit & Langar","desc":"Return to the Golden Temple for the midday Langar — the world's largest free community kitchen, serving 100,000 people daily. Volunteers (sevadars) cook, serve, and wash dishes as a form of worship. Sit on the floor to receive the simple dal and chapati meal alongside pilgrims, tourists, and locals alike. The experience of radical equality is profound. Entry: Free.","tags":["Sikh","Langar","Sacred"]},
                "afternoon": {"name":"Ram Bagh Palace & Museum","desc":"The Ram Bagh garden was laid out by Guru Har Gobind Singh (6th Guru) in the 17th century as a rest house for Sikh pilgrims. The adjacent Maharaja Ranjit Singh Museum, in the former summer palace of the Lion of Punjab, contains his personal artefacts, weapons, and the remarkable Sheesh Mahal (Hall of Mirrors). Entry: Rs.20 (Indians).","tags":["Sikh","Museum","Heritage"]},
                "evening":   {"name":"Amritsari Food Trail","desc":"Amritsar is famous for the richest Punjabi food in India: kulcha (stuffed bread) baked in a clay tandoor at Kesar Da Dhaba (est. 1916), amritsari machli (fried fish with ajwain batter), and the legendary lassi served in earthen pots so thick a spoon can stand upright. The food streets around the Golden Temple are electric at night. Cost: Rs.200–400.","tags":["Food","Punjab","Culture"]}},
            4: {"title": "Gurudwaras of the Punjab Trail",
                "morning":   {"name":"Gurudwara Tarn Taran Sahib","desc":"25km from Amritsar, Tarn Taran Sahib was founded by the 5th Sikh Guru, Arjan Dev, in 1590. The sarovar (sacred tank) here is the largest of any Gurudwara — 535 x 360 feet — and the Guru Arjan Dev himself prescribed bathing here as a cure for leprosy. The marble complex is peaceful and largely free of the Amritsar crowds. Entry: Free.","tags":["Sikh","Sacred","Pilgrimage"]},
                "afternoon": {"name":"Gurudwara Goindwal Sahib","desc":"40km from Amritsar, Goindwal was the first planned Sikh settlement, established by the 3rd Guru Amar Das in 1552. The Baoli Sahib here is a sacred stepwell with 84 steps representing the 84 lakh life cycles — pilgrims recite the Japji Sahib prayer on each step. Entry: Free.","tags":["Sikh","Stepwell","Pilgrimage"]},
                "evening":   {"name":"Pul Kanjri Heritage Fort & Village","desc":"12km from Amritsar on the Pakistan border, the Pul Kanjri reservoir and Mughal-era rest house (sarai) was a favourite stopping point for Mughal caravans between Lahore and Delhi. Maharaja Ranjit Singh later built a small fort complex here for his dancing girl, Moran. The village is a remarkable slice of Punjab's layered history. Entry: Free.","tags":["Mughal","Heritage","History"]}},
            5: {"title": "Lahore Day Trip (Attari Border Heritage)",
                "morning":   {"name":"Attari Border Heritage Walk","desc":"The Attari-Wagah border crossing is surrounded by pre-Partition buildings, the remains of the Grand Trunk Road customs post, and a small museum documenting the history of one of the most important borders in Asia. The Grand Trunk Road — 2,400km from Chittagong to Kabul — passes through here and was built by Sher Shah Suri in the 16th century. Entry: Free.","tags":["Partition","Heritage","History"]},
                "afternoon": {"name":"Dera Baba Nanak & Kartarpur Corridor","desc":"50km from Amritsar, the Kartarpur Corridor connects India to Kartarpur Sahib in Pakistan — the site where Guru Nanak spent the last 18 years of his life and died in 1539. Indian citizens with a valid passport can cross for the day (no visa required) to visit the Gurudwara Darbar Sahib. Cost: $20 service fee.","tags":["Sikh","Pilgrimage","Sacred"]},
                "evening":   {"name":"Golden Temple Evening Kirtan","desc":"Return to the Golden Temple for the evening kirtan (devotional music). The Golden Temple's ragis (musicians) play and sing the Guru Granth Sahib continuously day and night — the evening session after the Rehras Sahib prayer is particularly sublime, with the illuminated golden shrine reflected in the glowing sarovar. Entry: Free.","tags":["Sikh","Music","Sacred"]}},
            6: {"title": "Anandpur Sahib — Khalsa Birthplace",
                "morning":   {"name":"Anandpur Sahib — City of Bliss","desc":"120km from Amritsar, Anandpur Sahib is the birthplace of the Khalsa (Sikh warrior brotherhood), founded by Guru Gobind Singh on Baisakhi 1699. The Takht Sri Kesgarh Sahib Gurudwara marks the exact spot. The Virasat-e-Khalsa museum (2011), designed by Moshe Safdie, is one of the finest heritage museums in Asia. Entry: Rs.100 (Indians).","tags":["Sikh","Khalsa","Museum"]},
                "afternoon": {"name":"Kiratpur Sahib & Sutlej River","desc":"20km from Anandpur, Kiratpur Sahib was founded by the 6th Sikh Guru Hargobind in 1627 — the first Sikh town. The ashes of deceased Sikhs are immersed in the Sutlej river here. The riverside Gurudwaras and the flower garland market around the ghat are vivid and atmospheric. Entry: Free.","tags":["Sikh","River","Heritage"]},
                "evening":   {"name":"Ropar Wetland Bird Sanctuary","desc":"The Ropar Wetland on the Sutlej, 15km from Kiratpur, is a Ramsar-designated wetland hosting over 150 bird species including grey-lag geese, bar-headed geese, and sarus cranes in the winter months. The sunset over the wetland with the Shivalik foothills beyond is beautiful. Entry: Free.","tags":["Birds","Nature","Sunset"]}},
            7: {"title": "Amritsar's Old City Deep Dive",
                "morning":   {"name":"Katra Jaimal Singh & Old City Markets","desc":"The historic cloth market of Katra Jaimal Singh in Amritsar's old city has been operating since the 18th century. The narrow lanes are packed with wholesale textile merchants and the ancient havelis that surround the market retain their original Sikh-period facades. The morning light in the lanes is extraordinary for photography. Cost: Free.","tags":["Heritage","Shopping","Photography"]},
                "afternoon": {"name":"Gobindgarh Fort","desc":"At the heart of Amritsar, Gobindgarh Fort (18th century) was the treasury and armoury of Maharaja Ranjit Singh. Recently opened to the public after serving as an Indian Army cantonment for 150 years, it contains Ranjit Singh's personal treasury vault, the cannon Zam-Zama (before its Lahore posting), and a lively heritage theme park inside. Entry: Rs.200 (Indians).","tags":["Sikh","Fort","Heritage"]},
                "evening":   {"name":"Heritage Walk — Sikh-Era Havelis","desc":"Amritsar's old city contains dozens of 18th and 19th century Sikh-era havelis — the mansions of merchants who grew rich supplying the Sikh Empire. The painted and sculpted facades of Katra Ahluwalia and Hall Bazaar are extraordinary. A guided heritage walk reveals how much of Ranjit Singh's capital is still intact. Cost: Guide Rs.500.","tags":["Heritage","Architecture","Walking"]}},
            8: {"title": "Pathankot & Kangra Fort",
                "morning":   {"name":"Kangra Fort — Oldest Dated Fort of India","desc":"100km from Amritsar near Dharamshala, the Kangra Fort was founded by the Katoch dynasty — the oldest surviving royal dynasty in the world, whose lineage stretches back 3,500 years. The fort, sacked by Mahmud of Ghazni in 1009, Timur in 1398, and the Mughals in 1620, still has ramparts stretching 4km. Entry: Rs.25 (Indians).","tags":["Fort","Ancient","History"]},
                "afternoon": {"name":"Masrur Rock-Cut Temple","desc":"35km from Kangra, the Masrur temples (8th century CE) are a group of 15 rock-cut Hindu temples carved from a single sandstone outcrop in the Shivalik foothills — often called the 'Himalayan Ellora.' The main shikhara temple is dedicated to Shiva. The mountain backdrop makes this one of North India's most photogenic temple sites. Entry: Rs.25 (Indians).","tags":["Rock-Cut","Temple","Himalaya"]},
                "evening":   {"name":"Dharamshala & Tibetan Heritage","desc":"Dharamshala is the seat of the Tibetan Government in Exile and the residence of the Dalai Lama since 1959. The Namgyal Monastery, the Tibet Museum, and the narrow lanes of McLeod Ganj create a unique blend of Himalayan and Tibetan culture unlike anywhere else in India. Stay overnight for the best experience. Entry: Free.","tags":["Tibet","Buddhism","Culture"]}},
            9: {"title": "Amritsar to Shimla Heritage Route",
                "morning":   {"name":"Kalka-Shimla Toy Train","desc":"Take the Kalka-Shimla Railway (UNESCO, 2008) — a narrow-gauge mountain railway with 102 tunnels, 864 bridges, and 919 curves across 96km of Himalayan foothills. The 5-hour journey (departs Kalka, 100km from Amritsar) passes through pine forests and the extraordinary Barog Tunnel (1.1km, the longest on the line). Train: Rs.50–300.","tags":["UNESCO","Railway","Heritage"]},
                "afternoon": {"name":"Shimla Heritage Walk — The Ridge","desc":"The ridge of Shimla, at 2,205m, was the summer capital of British India — the Viceroy's residence, Christ Church (1857), and the Gaiety Theatre (1887) all stand intact. The Mall Road's colonial architecture and the panoramic views across the Himalayas make it one of India's finest colonial hill stations. Entry: Free.","tags":["Colonial","Himalaya","Architecture"]},
                "evening":   {"name":"Jakhu Temple & Himalayan Sunset","desc":"The Jakhu Temple (2,455m) dedicated to Hanuman is the highest point of Shimla, reached by a 2km trail or ropeway through dense rhododendron forest. The sunset view across the Himalayan ranges — with Kinnaur's snow peaks visible on clear days — is one of North India's great panoramas. Entry: Free (ropeway Rs.250).","tags":["Temple","Himalaya","Sunset"]}},
            10: {"title": "Final Amritsar — Golden Temple Farewell",
                "morning":   {"name":"Golden Temple Pre-dawn Amrit Vela","desc":"The holiest time at the Golden Temple is amrit vela (3–5am) — the hours before dawn when the Guru Granth Sahib is installed on the throne and the first hymns of the day are sung in the darkness while the illuminated golden shrine shimmers in the sarovar. This experience, shared with thousands of devout Sikhs, is deeply moving. Entry: Free.","tags":["Sikh","Sacred","Dawn"]},
                "afternoon": {"name":"Final Shopping — Phulkari & Jutti","desc":"Amritsar's specialities to take home: phulkari embroidery (the ancient Punjabi needle-work tradition, with orange silk thread on coarse cotton), and jutti (the handmade pointed-toe Punjabi leather shoes, often intricately embroidered). Both are available in the lanes of Hall Bazaar. Cost: Rs.500–5,000.","tags":["Shopping","Craft","Punjab"]},
                "evening":   {"name":"Farewell Kirtan at the Golden Temple","desc":"Attend the final evening kirtan at the Harmandir Sahib as the golden shrine glows on the sacred water and the last pilgrims bathe in the Amrit Sarovar. The cycle of prayer, music, and community has continued here every single day since 1604. Nothing in India equals this continuity of living faith. Entry: Free.","tags":["Sikh","Sacred","Farewell"]}},
        }
    },
    "jodhpur": {
        "name": "Jodhpur, Rajasthan", "emoji": "JDH",
        "tagline": "The Blue City Under the Iron Fort",
        "monuments": ["Mehrangarh Fort","Jaswant Thada","Umaid Bhawan Palace","Mandore Gardens","Toorji Ka Jhalra","Balsamand Lake"],
        "budget": {"budget": "Rs.2,800", "mid": "Rs.8,500", "luxury": "Rs.24,000"},
        "days": {
            1: {"title": "Mehrangarh & the Blue City",
                "morning":   {"name":"Mehrangarh Fort at Sunrise","desc":"One of the largest forts in India, Mehrangarh rises 400ft above the city on a rocky outcrop. Founded by Rao Jodha in 1459, it took two centuries to complete. The fort museum contains the finest collection of Rajput armoury, elephant howdahs, and royal palanquins in India. The views over the blue-washed old city are extraordinary. Entry: Rs.200 (Indians).","tags":["Fort","Rajput","Museum"]},
                "afternoon": {"name":"Jaswant Thada & Clock Tower","desc":"The Jaswant Thada (1899) is a white marble cenotaph built by Maharaja Sardar Singh in memory of his father — often called the 'Taj Mahal of Marwar.' Nearby, the 19th-century Sardar Bazaar around the Clock Tower is the beating heart of the old city, famous for its spices, handicrafts, and street food. Entry: Rs.30.","tags":["Memorial","Shopping","Architecture"]},
                "evening":   {"name":"Sunset from Mehrangarh Ramparts","desc":"Return to Mehrangarh's northern ramparts at dusk for the best view of the blue city turning indigo at sunset. The blue colour (traditionally used by Brahmin households) became the entire city's aesthetic over centuries. Book a rooftop dinner at Indique restaurant for candlelit fort views. Entry: Rs.200 (Indians).","tags":["Sunset","Photography","Views"]}},
            2: {"title": "Royal Gardens & Palace Heritage",
                "morning":   {"name":"Mandore Gardens & Cenotaphs","desc":"Mandore, 9km north of Jodhpur, was the pre-Mehrangarh capital of the Marwar kingdom. The formal gardens contain the imposing deval (cenotaphs) of Jodhpur's rulers — towering multi-storey stone structures unlike any royal memorial in Rajasthan. The Hall of Heroes features 15 life-size sandstone hero figures. Entry: Free.","tags":["Gardens","Cenotaphs","Heritage"]},
                "afternoon": {"name":"Umaid Bhawan Palace","desc":"Built between 1929–1943, this 347-room Chittor sandstone palace was constructed to provide employment during a devastating famine. The palace is one of the world's largest private residences and is divided into three: a royal residence, a heritage hotel (Taj), and a museum. The Art Deco interiors are extraordinary.","tags":["Palace","Art Deco","Royalty"]},
                "evening":   {"name":"Toorji Ka Jhalra Stepwell","desc":"This 18th-century stepwell (baoli) was commissioned by the chief queen of Maharaja Abhay Singh. It descends six storeys and features ornate carved pillars. Restored in 2016, the surrounding area has been transformed into Jodhpur's most atmospheric heritage precinct, with rooftop cafes and art galleries.","tags":["Heritage","Architecture","Culture"]}},
            3: {"title": "Osian — Desert Temples & Sand Dunes",
                "morning":   {"name":"Osian Temple Complex","desc":"65km north of Jodhpur, Osian is an ancient town with 16 Hindu and Jain temples from the 8th–11th century scattered across the desert — earning it the title 'Khajuraho of Rajasthan.' The Sachiya Mata Temple (8th century) and the Mahavira Temple (8th century) are the finest. The drive through the Thar scrubland is beautiful. Entry: Rs.25 (Indians).","tags":["Temple","Jain","Desert"]},
                "afternoon": {"name":"Osian Sand Dunes & Camel Safari","desc":"The crescent sand dunes (barchans) at Osian are among the most accessible in the Thar Desert — 60km from Jodhpur with the backdrop of ancient temples. Camel safaris here are more authentic and far less crowded than Jaisalmer. An hour among the dunes with a camel and a local guide at sunset is unforgettable. Cost: Rs.500–1,500.","tags":["Camel","Desert","Adventure"]},
                "evening":   {"name":"Jodhpur Old City by Night","desc":"The old city of Jodhpur within the ramparts glows a deep blue at night, the fort lit on the hill above. A late evening walk through the Nai Sarak spice market and the Mochi Bazaar leather shoe district — still operating by a single overhead bulb at 9pm — is one of Rajasthan's most atmospheric night experiences. Cost: Free.","tags":["Night","Shopping","Culture"]}},
            4: {"title": "Bishnoi Villages & Wildlife",
                "morning":   {"name":"Bishnoi Village Safari","desc":"The Bishnoi community, 25km south of Jodhpur, follow a 500-year-old ecological faith that forbids harming any living creature. Their villages are surrounded by blackbucks, peacocks, and chinkara gazelles that walk freely among the houses. In 1730, 363 Bishnoi villagers gave their lives to protect trees from being felled — the world's first documented environmental martyrdom. Entry: Free (guide Rs.500–1,000).","tags":["Wildlife","Culture","Ecology"]},
                "afternoon": {"name":"Salawas Block-Print Village","desc":"30km from Jodhpur, Salawas is one of Rajasthan's finest dhurrie (cotton flatweave rug) weaving villages. The village cooperative uses natural dyes and traditional pit-loom techniques. The rugs, with their bold geometric Rajasthani patterns, take 4–8 weeks per piece and are exported worldwide. Visit families directly. Cost: Free.","tags":["Craft","Textile","Village"]},
                "evening":   {"name":"Balsamand Lake Palace","desc":"8km from Jodhpur, the Balsamand Lake was created in 1159 CE as an artificial reservoir by the Parihar rulers — one of the oldest man-made lakes in Rajasthan. The Victorian palace on its bank (now a heritage hotel) has a rose and fruit garden open to visitors at sunset. The flamingos and waterbirds on the lake are spectacular. Entry: Free (garden Rs.200).","tags":["Lake","Heritage","Nature"]}},
            5: {"title": "Nagaur Fort & Camel Fair Heritage",
                "morning":   {"name":"Nagaur Fort — The Agra of Rajasthan","desc":"135km from Jodhpur, Nagaur Fort is one of Rajasthan's most intact and undervisited forts. The 4km ramparts enclose Mughal-era palaces with extraordinary painted interiors, marble fountains, and the oldest surviving Mughal water garden in India — the Shahi Baoli. The fort was restored through a partnership with the Aga Khan Trust. Entry: Rs.100 (Indians).","tags":["Fort","Mughal","Heritage"]},
                "afternoon": {"name":"Nagaur Old City & Ahhichatragarh","desc":"Nagaur's old city, within the ramparts, contains 15th-century mosques, havelis with painted facades, and a remarkable Muslim dargah. The Nagaur Cattle & Camel Fair (held each January–February) is one of India's largest rural fairs — drawing 75,000 animals and 200,000 people. Even outside fair season the town is fascinating. Entry: Free.","tags":["Culture","Heritage","Rural"]},
                "evening":   {"name":"Khimsar Dunes Village & Desert Night","desc":"170km from Jodhpur, Khimsar is a small village beside a medieval fort converted into a heritage hotel, surrounded by the highest sand dunes in eastern Rajasthan. A bonfire dinner in the dunes with local musicians and a clear desert sky full of stars makes for a perfect night in the Thar. Cost: Rs.2,000–4,000.","tags":["Desert","Heritage","Night"]}},
            6: {"title": "Chittorgarh — The Rajput Citadel",
                "morning":   {"name":"Chittorgarh Fort — Rajput Pride","desc":"250km from Jodhpur, Chittorgarh Fort is the largest fort in India (2.8 sq km) and the most historically significant Rajput stronghold. It was the capital of the Mewar kingdom and witnessed three heroic sieges (1303, 1535, 1568) — each ending with the mass self-immolation (jauhar) of Rajput women. The 37m Vijay Stambha (Tower of Victory, 1448) is one of India's greatest monuments. Entry: Rs.25 (Indians).","tags":["Fort","Rajput","UNESCO"]},
                "afternoon": {"name":"Rani Padmini Palace & Kumbha Shyam Temple","desc":"The Padmini Palace on Chittorgarh Fort is where Queen Padmini (Padmavati) gave her reflection to the Delhi Sultan Alauddin Khalji from a mirror — an act of deception leading to the 1303 battle. The 15th-century Kumbha Shyam Temple and Mira Temple (dedicated to the poet-saint Meera Bai) are architectural gems. Entry: Rs.25 (Indians).","tags":["Rajput","Temple","History"]},
                "evening":   {"name":"Sound & Light Show at Chittorgarh","desc":"The evening Sound and Light show at Chittorgarh Fort narrates the three sacks of the fort through dramatic lighting and narration — the stories of Padmini, Jaimal, Patta, and Maharana Pratap brought to life against the actual fort walls. Return to Jodhpur or overnight in Chittorgarh. Entry: Rs.100 (Indians).","tags":["Heritage","Culture","History"]}},
            7: {"title": "Mount Abu — Sacred Hill Station",
                "morning":   {"name":"Dilwara Jain Temples","desc":"The Dilwara temples at Mount Abu (1031–1231 CE) are considered the supreme example of Jain marble architecture in India. The Luna Vasahi temple's ceiling has marble carved into filigree so thin it is translucent. Each panel took a craftsman a lifetime. The Vimal Vasahi temple (1031 CE) has 48 pillars with unique carved figures on each. Entry: Free (no photography).","tags":["Jain","Marble","Architecture"]},
                "afternoon": {"name":"Nakki Lake & Guru Shikhar","desc":"Nakki Lake at Mount Abu (1,220m) is the only lake in Rajasthan fed by natural springs. According to legend it was scooped out by the gods using their nails (nakh). Guru Shikhar at 1,722m is the highest point in the Aravalli range — an easy 20-minute walk from the roadhead offers a 360° Rajasthan panorama. Entry: Free.","tags":["Lake","Nature","Views"]},
                "evening":   {"name":"Achalgarh Fort & Sunset","desc":"Achalgarh Fort (1412 CE), 11km from Mount Abu, was built by Maharana Kumbha and contains some of the finest carved Shaivite temples in Rajasthan. The fort offers a spectacular sunset view across the Aravalli range towards Gujarat. Entry: Rs.10 (Indians).","tags":["Fort","Sunset","Heritage"]}},
            8: {"title": "Jodhpur Food, Craft & Textile Heritage",
                "morning":   {"name":"Jodhpur Textile & Tie-Dye Workshops","desc":"Jodhpur is a major centre of Rajasthani tie-dye (bandhani), block-printing, and hand-woven textiles. The workshops of Kaga Wool Weavers in the old city produce the traditional Marwari pattoo blankets. A morning workshop visit lets you try block-printing on cotton. Cost: Rs.200–500.","tags":["Craft","Textile","Art"]},
                "afternoon": {"name":"Mehrangarh Fort Museum Deep Dive","desc":"Return to Mehrangarh for the museum collections often skipped by first-time visitors: the palanquin gallery (19 royal palanquins, including the tent-palanquin used on campaign), the turban gallery with 40 styles of Rajasthani turbans, and the Daulat Khana (treasury). The fort's audio guide (Rs.200) is outstanding. Entry: Rs.200 (Indians).","tags":["Fort","Museum","Heritage"]},
                "evening":   {"name":"Stepwell Walk & Blue City Photography","desc":"Jodhpur has 7 historic stepwells (baolies) within walking distance of the old city — Toorji ka Jhalra, Gulab Sagar, and Fateh Sagar are the most atmospheric. A sunset photography walk through the Blue City's narrow lanes, catching the indigo walls against the golden-lit Mehrangarh Fort above, is one of Rajasthan's great photography experiences. Cost: Free.","tags":["Photography","Heritage","Architecture"]}},
            9: {"title": "Jodhpur's Rural Hinterland",
                "morning":   {"name":"Rao Jodha Desert Rock Park","desc":"At the foot of Mehrangarh Fort, this 70-acre ecological park was created to restore the native desert ecosystem of the Thar. It contains 300 species of plants found in the Thar, trails through ancient volcanic rock formations, and views up to the fort ramparts. At dawn, before the crowds, it is one of Jodhpur's most peaceful experiences. Entry: Rs.100 (Indians).","tags":["Nature","Ecology","Heritage"]},
                "afternoon": {"name":"Khejarla Fort & Village Lunch","desc":"85km from Jodhpur, the 17th-century Khejarla Fort is one of the best-preserved small Rajput forts in Marwar — privately owned and now a heritage hotel. A village lunch (thali with bajra roti, ker sangri, and buttermilk) at a local family courtyard restaurant is the most authentic Marwari meal possible. Cost: Rs.300–500.","tags":["Fort","Village","Food"]},
                "evening":   {"name":"Ghanta Ghar & Farewell Lassi","desc":"The Clock Tower (Ghanta Ghar) built in the late 19th century stands at the heart of Jodhpur's busiest market. The Prakash Misthan Bhandar shop nearby makes mawa kachori — a deep-fried pastry stuffed with sweetened mawa (reduced milk solids) that is Jodhpur's signature sweet. The rooftop cafes here have the best Mehrangarh view in the city. Cost: Rs.50–200.","tags":["Food","Shopping","Heritage"]}},
            10: {"title": "Final Jodhpur — Fort at First Light",
                "morning":   {"name":"Mehrangarh at Opening Time","desc":"Be at Mehrangarh Fort gates at 9am when they open. With the morning light falling on the elaborate carvings of the Loha Pol (Iron Gate) and the ghost-handprints of the sati queens, and almost no other visitors, the fort reveals itself completely. Spend a final hour on the ramparts watching the blue city wake below. Entry: Rs.200 (Indians).","tags":["Fort","Photography","Farewell"]},
                "afternoon": {"name":"Bal Samand & Farewell Gardens","desc":"A final afternoon at Balsamand Lake's 12th-century reservoir for a peaceful walk under the mango and guava trees on the lake's banks. The flamingos, pelicans, and kingfishers that congregate here make it Jodhpur's most underrated nature spot. Entry: Rs.200 (garden visit).","tags":["Lake","Nature","Farewell"]},
                "evening":   {"name":"Mirchi Bada & Farewell Market Walk","desc":"Jodhpur's mirchi bada — a massive green chilli coated in potato stuffing, battered, and deep fried — is one of Rajasthan's most iconic street foods, eaten with mint chutney at roadside stalls. A final walk through the Nai Sarak spice market at dusk, with its heaps of red chillies, coriander, and fenugreek, is the definitive Jodhpur farewell. Cost: Rs.50–200.","tags":["Food","Shopping","Farewell"]}},
        }
    },
    "udaipur": {
        "name": "Udaipur, Rajasthan", "emoji": "UDR",
        "tagline": "The City of Lakes & Floating Palaces",
        "monuments": ["City Palace","Lake Palace","Jagmandir Island","Saheliyon ki Bari","Kumbhalgarh Fort","Monsoon Palace"],
        "budget": {"budget": "Rs.3,500", "mid": "Rs.11,000", "luxury": "Rs.32,000"},
        "days": {
            1: {"title": "Lake Pichola & the City Palace",
                "morning":   {"name":"City Palace at Dawn","desc":"The largest palace complex in Rajasthan, built over 400 years by the Mewar dynasty beginning with Maharana Udai Singh II in 1559. The sprawling 5-storey complex contains 11 mahals (palaces), museums, gardens, and courtyards. The Crystal Gallery holds the largest private collection of crystal furniture in the world, ordered from F&C Osler of Birmingham in 1877. Entry: Rs.300 (Indians).","tags":["Palace","Mewar","Museum"]},
                "afternoon": {"name":"Lake Pichola Boat Ride","desc":"Lake Pichola was created in the 14th century. A boat ride visits the Jag Mandir island palace — where Mughal prince Khurram (later Shah Jahan) took refuge in 1623 — and passes the Taj Lake Palace, a white marble marvel that appears to float. The lake is surrounded by the Aravalli hills and framed by ghats and temples. Cost: Rs.400–800.","tags":["Lake","Palace","Photography"]},
                "evening":   {"name":"Bagore Ki Haveli Cultural Show","desc":"This 18th-century haveli (townhouse) on Gangaur Ghat houses a folklore museum with the world's largest turban on display. Every evening at 7pm, the Dharohar Dance Show presents ghoomar, bhavai, and kalbelia folk dances in the courtyard. Entry: Rs.90 (Indians).","tags":["Culture","Dance","Heritage"]}},
            2: {"title": "Kumbhalgarh & Mountain Forts",
                "morning":   {"name":"Kumbhalgarh Fort & Great Wall","desc":"Built by Maharana Kumbha in the 15th century, 64km north of Udaipur, this UNESCO fort has a perimeter wall stretching 36km — second only to the Great Wall of China. The fort contains 360 temples and was the birthplace of Maharana Pratap (1540), the great Rajput warrior king. Entry: Rs.40 (Indians).","tags":["UNESCO","Fort","History"]},
                "afternoon": {"name":"Saheliyon ki Bari","desc":"The Garden of Maids of Honour, built by Maharana Sangram Singh II in the 18th century for the royal ladies of the court. The garden contains elephant fountains, marble pavilions, lotus pools, and kiosks with shaded walkways — a rare example of a Rajasthani pleasure garden designed exclusively for women. Entry: Rs.10 (Indians).","tags":["Gardens","Royalty","Architecture"]},
                "evening":   {"name":"Sunset from Monsoon Palace","desc":"The Sajjangarh Palace (Monsoon Palace), built in 1884 by Maharana Sajjan Singh, sits at 944m atop the Aravalli range, 5km from the city. It was designed to observe monsoon clouds approaching across Rajasthan. The panoramic sunset view over Udaipur's lakes is among the most spectacular in India. Entry: Rs.10 (car extra).","tags":["Sunset","Views","Photography"]}},
            3: {"title": "Ranakpur & Jain Marble Temples",
                "morning":   {"name":"Ranakpur Jain Temple","desc":"96km north of Udaipur, the Ranakpur temple (1437 CE) dedicated to Adinatha is one of the five most sacred Jain shrines. The white marble temple has 1,444 intricately carved pillars — no two alike. The main hall's roof is supported by 80 ornate domes, and the carved ceiling above the central shrine is arguably the finest stone carving in India. Entry: Rs.200 (Indians).","tags":["Jain","Marble","Architecture"]},
                "afternoon": {"name":"Kumbhalgarh Wildlife Sanctuary","desc":"Surrounding Kumbhalgarh Fort, the 578 sq km sanctuary is home to wolves, leopards, hyenas, and sloth bears — and one of the last refuges of the Indian wolf in Rajasthan. A wildlife safari at dusk through the dry deciduous forest offers sightings of deer, peacocks, and birds of prey. Entry: Rs.200 (Indians).","tags":["Wildlife","Nature","Fort"]},
                "evening":   {"name":"Narlai Village & Rock Temples","desc":"50km from Udaipur, Narlai is a medieval village built around a massive smooth granite rock surmounted by 32 Jain and Hindu shrines. The rock, called Aravalli Parvat, is climbed via carved steps — the view at sunset over the Aravalli valley is extraordinary. The village guesthouse serves authentic Rajasthani food. Entry: Free.","tags":["Village","Temple","Sunset"]}},
            4: {"title": "Chittorgarh — The Soul of Rajput Pride",
                "morning":   {"name":"Chittorgarh Fort","desc":"112km from Udaipur, Chittorgarh Fort is the largest fort in India (2.8 sq km) and the most sacred site of Mewar heritage. Three jauhars (mass self-immolations) took place here to resist conquest — in 1303, 1535, and 1568. The 37m Vijay Stambha (Tower of Victory, 1448) and the Rana Kumbha Palace are the centrepieces. Entry: Rs.25 (Indians).","tags":["Fort","Rajput","UNESCO"]},
                "afternoon": {"name":"Meera Temple & Padmini Palace","desc":"The Meera Temple at Chittorgarh is dedicated to the Bhakti saint-poetess Mirabai (1498–1547), who chose Krishna over her royal husband and eventually walked into a Krishna idol here. Nearby, the Padmini Palace is where the story of Queen Padmavati that inspired the 2018 film begins. The panoramic views from the fort are spectacular. Entry: Rs.25 (Indians).","tags":["Temple","History","Rajput"]},
                "evening":   {"name":"Sound & Light Show & Return","desc":"The Sound and Light show at Chittorgarh Fort dramatically narrates the three sieges and jauhars. Return to Udaipur through the Aravalli landscape at night — the mountains silhouetted against stars. Cost: Rs.100 (show). Drive: 2 hours.","tags":["Culture","Heritage","History"]}},
            5: {"title": "Dungarpur & Tribal Heritage",
                "morning":   {"name":"Udai Bilas Palace & Juna Mahal","desc":"120km south of Udaipur in Dungarpur, the Udai Bilas Palace (1870s) combines Rajput and Italian Gothic architecture beside the Gaibsagar Lake. The nearby Juna Mahal (13th century) is one of Rajasthan's oldest occupied palaces — seven storeys of apartments covered with extraordinary miniature paintings, mirror work, and coloured glass. Entry: Rs.100–200 (Indians).","tags":["Palace","Miniature","Architecture"]},
                "afternoon": {"name":"Tribal Craft Villages of Dungarpur","desc":"Dungarpur district is home to the Bhil tribal community — one of India's largest indigenous groups. The villages around Dungarpur produce distinctive Pithora ritual paintings (large colourful compositions covering entire interior walls of homes), bamboo weaving, and Gond-influenced jewelry. A guided village visit is deeply illuminating. Cost: Guide Rs.500.","tags":["Tribal","Craft","Culture"]},
                "evening":   {"name":"Gaibsagar Lake Sunset","desc":"The Gaibsagar Lake in Dungarpur, created in the 14th century, is lined with small medieval Rajput temples and ghats. The Shrinathji Temple on its bank reflects in the still water at dusk. A boat ride offers views of the palace beyond the lake. Entry: Free.","tags":["Lake","Temple","Sunset"]}},
            6: {"title": "Nathdwara & Eklingji Pilgrimage",
                "morning":   {"name":"Nathdwara Shrinathji Temple","desc":"48km from Udaipur, Nathdwara is home to the Shrinathji Temple — the most important Vaishnava pilgrimage site in Rajasthan and one of the richest temples in India. The temple's presiding deity is a 7th-century black marble image of the child Krishna brought here from Mathura in 1669. The Pichwai paintings — large fabric depictions of Krishna's life — originated here. Entry: Free.","tags":["Vaishnavism","Pilgrimage","Art"]},
                "afternoon": {"name":"Eklingji & Nagda Temples","desc":"22km north of Udaipur, the Eklingji Temple complex (734 CE) has been the personal deity of the Mewar Maharanas for 1,200 years. The complex contains 108 temples within a walled enclosure. The adjacent Nagda temples (10th century) beside the Nagda lake are among the oldest surviving temples in the region. Entry: Free.","tags":["Temple","Shiva","Mewar"]},
                "evening":   {"name":"Haldi Ghati Battlefield","desc":"40km from Udaipur, Haldi Ghati is the mountain pass where Maharana Pratap fought Akbar's army on 18 June 1576. Though outnumbered, Pratap refused to submit and fought for his kingdom until his death in 1597. The pass has a small museum and the memorial to Pratap's legendary horse Chetak. Entry: Rs.20 (Indians).","tags":["History","Battle","Heritage"]}},
            7: {"title": "Udaipur Lakes — All Five",
                "morning":   {"name":"Fateh Sagar Lake & Nehru Park","desc":"Fateh Sagar lake, north of Pichola, was originally built by Maharana Jai Singh in 1678. The 2.4 sq km lake has a boat-accessible island garden (Nehru Park) with an open-air cinema and aquarium. The hilltop Moti Magri memorial to Maharana Pratap overlooks it. The lake is famous for morning mist and migrating birds. Entry: Free (boat Rs.30).","tags":["Lake","Nature","Heritage"]},
                "afternoon": {"name":"Badi Lake & Bujra Hill Village","desc":"11km from Udaipur, Badi Lake (built 1652) is Udaipur's oldest reservoir — used to supply the city during droughts. The drive passes through the Aravalli countryside to the lake's dam wall, where a modest 17th-century pleasure pavilion still stands. The surrounding hills have panoramic views of the lake system. Entry: Free.","tags":["Lake","Nature","Hidden Gem"]},
                "evening":   {"name":"Lake Pichola by Night — Ghats & Gondolas","desc":"Return to Lake Pichola for an evening boat ride — the City Palace lit and reflected in the water, the Lake Palace glowing white, and the chhatris of Ambrai Ghat silhouetted against the floodlit fort. A dinner at Ambrai restaurant on the lakeshore is one of Udaipur's finest experiences. Cost: Boat Rs.400–800.","tags":["Lake","Photography","Food"]}},
            8: {"title": "Udaipur Crafts & Art Heritage",
                "morning":   {"name":"Shilpgram Rural Arts Festival","desc":"3km from Udaipur, Shilpgram is a crafts village set up by the Indian government where traditional artisans from five western states demonstrate and sell their craft. The annual Shilpgram Festival (December) brings 800 artisans. Even outside festival season, the village workshop demonstrations of puppet making, weaving, and pottery are excellent. Entry: Rs.30 (Indians).","tags":["Craft","Art","Culture"]},
                "afternoon": {"name":"Ahar Museum & Mewar Bronze Heritage","desc":"The Ahar Cenotaphs on the outskirts of Udaipur are the royal cremation ground of the Mewar Maharanas — 250 cenotaphs over 400 years, each marking a different ruler. The adjacent Ahar Archaeological Museum displays prehistoric pottery and tools from the Ahar culture (3rd millennium BCE) — one of India's oldest known cultures. Entry: Rs.10 (Indians).","tags":["Museum","Archaeology","Heritage"]},
                "evening":   {"name":"Bhartiya Lok Kala Museum & Puppet Show","desc":"The Bhartiya Lok Kala Museum (Folk Art Museum) has the finest collection of Rajasthani folk art in Udaipur — folk costumes, tribal jewelry, Pichwai paintings, and over 100 traditional puppets. An evening puppet show using the Rajasthani Kathputli (string puppet) tradition brings the museum's collection to life. Entry: Rs.30 (Indians).","tags":["Folk Art","Puppets","Culture"]}},
            9: {"title": "Kumbalgarh to Ranakpur Road",
                "morning":   {"name":"Kumbhalgarh Fort Sunrise Walk","desc":"Arrive at Kumbhalgarh at opening time (8am) to walk the full 36km wall circuit in early morning light — most of it without other visitors. The views from the northern battlements extend across the Aravalli range to the Thar Desert. The fort's Vedi Temple (15th century) has some of the finest stone carvings in Rajasthan. Entry: Rs.40 (Indians).","tags":["Fort","Walking","Photography"]},
                "afternoon": {"name":"Muchhal Mahavira Temple","desc":"En route from Kumbhalgarh to Ranakpur, the Muchhal Mahavira Temple at Ghanerao is one of Rajasthan's most unusual Jain temples — the idol of Mahavira has a large moustache (muchhal), unique in Jain iconography. The 15th-century temple is set in a tiny village and is almost entirely free of tourists. Entry: Free.","tags":["Jain","Temple","Hidden Gem"]},
                "evening":   {"name":"Ranakpur Valley at Dusk","desc":"The Ranakpur valley at dusk — forested hills, peacocks returning to roost, the white marble temple emerging from the trees — is one of Rajasthan's most serene landscapes. The overnight dharamshala at the temple accepts all visitors. Listen to the evening Jain prayers echoing through the marble halls. Entry: Free.","tags":["Temple","Nature","Serenity"]}},
            10: {"title": "Farewell Udaipur — Lakeside Last Day",
                "morning":   {"name":"City Palace Roof Terrace at Dawn","desc":"Return to the City Palace for the rooftop terrace at opening time — the view across all five lakes, the white city, and the Aravalli hills in the morning mist is the definitive Udaipur panorama. Entry: Rs.300 (Indians). Arrive by 9am.","tags":["Palace","Photography","Farewell"]},
                "afternoon": {"name":"Jagdish Temple & Old City Walk","desc":"The Jagdish Temple (1651), built by Maharana Jagat Singh I, is Udaipur's largest functioning temple and the spiritual centre of the old city. The Vishnu idol in black stone is surrounded by bronze Garuda pillars of remarkable craft. The surrounding old city lanes to Lal Ghat are full of miniature painting workshops and antique dealers. Entry: Free.","tags":["Temple","Vaishnavism","Heritage"]},
                "evening":   {"name":"Final Sunset on Lake Pichola","desc":"Hire a private boat for a final sunset on Lake Pichola — drifting between the Lake Palace and the Jag Mandir island as the sun turns the Aravalli hills amber, the City Palace glows gold on the hill, and the call to prayer from the old city mosque drifts across the water. Arguably India's most romantic evening. Cost: Boat Rs.800–2,000.","tags":["Lake","Sunset","Farewell"]}},
        }
    },
    "jaisalmer": {
        "name": "Jaisalmer, Rajasthan", "emoji": "JSM",
        "tagline": "The Golden Fortress City of the Thar Desert",
        "monuments": ["Jaisalmer Fort","Patwon Ki Haveli","Nathmal Ki Haveli","Gadisar Lake","Sam Sand Dunes","Jain Temples"],
        "budget": {"budget": "Rs.3,000", "mid": "Rs.8,500", "luxury": "Rs.22,000"},
        "days": {
            1: {"title": "The Living Fort & Golden Havelis",
                "morning":   {"name":"Jaisalmer Fort at Sunrise","desc":"One of the world's largest living forts, Sonar Quila (the Golden Fort) rises from the Thar Desert like a mirage. Built in 1156 by Rawal Jaisal, it is still home to about 3,000 people within its walls — a quarter of Jaisalmer's old city. The golden-yellow Trikuta Hill sandstone glows brilliantly at sunrise. Entry: Rs.50 (Indians).","tags":["UNESCO","Fort","Living Heritage"]},
                "afternoon": {"name":"Jain Temples & Patwon Ki Haveli","desc":"Inside the fort, seven elaborately carved Jain temples (12th–15th century) house superb marble sculpture. Outside the fort, the Patwon Ki Haveli (1805) is a cluster of five interconnected mansions built by a wealthy Jain merchant family — 60 elaborately carved balconies make it the most ornate haveli in Rajasthan. Entry: Rs.50–100.","tags":["Jain","Architecture","Heritage"]},
                "evening":   {"name":"Gadisar Lake at Sunset","desc":"This man-made reservoir was built by Maharawal Gadsi Singh in 1367 to collect rainwater and supply the city through its arid desert summers. The lake is surrounded by shrines, chhatris (cenotaphs), and temples. The ornate Tilon Ki Pol gateway, built by a royal courtesan, frames the lake at sunset. Cost: Boat hire Rs.50–100.","tags":["Lake","Sunset","Photography"]}},
            2: {"title": "Sam Sand Dunes & Desert Life",
                "morning":   {"name":"Desert National Park","desc":"Located 40km from Jaisalmer, the 3,162 sq km Desert National Park is one of the largest national parks in India and protects the ecosystem of the Thar Desert. It is the best place to spot the endangered Great Indian Bustard and the chinkara gazelle. Jeep safari from Rs.1,500.","tags":["Wildlife","Nature","Desert"]},
                "afternoon": {"name":"Khaba Fort & Ghost Village","desc":"Just 20km from Jaisalmer, the abandoned village of Kuldhara was deserted overnight in 1825 by its entire Paliwal Brahmin community to avoid the oppressive taxes of the Diwan. The fossilised village of 84 abandoned stone houses and the adjoining ruined Khaba Fort are among Rajasthan's most atmospheric heritage sites. Entry: Rs.10.","tags":["History","Mystery","Architecture"]},
                "evening":   {"name":"Sam Sand Dunes & Camel Safari","desc":"The Sam dunes, 42km from Jaisalmer, are the most spectacular sand dunes in the Thar — great crescent-shaped barchans up to 30m high. A camel safari at sunset across these dunes is one of India's most iconic experiences. Overnight desert camps offer traditional folk music, fire shows, and Rajasthani dinner. Cost: Rs.1,500–5,000 (camp).","tags":["Desert","Camel","Adventure"]}},
            3: {"title": "Havelis & Jaisalmer Fort Deep Dive",
                "morning":   {"name":"Nathmal Ki Haveli & Salim Singh Ki Haveli","desc":"Nathmal Ki Haveli (1885), built for the Prime Minister of Jaisalmer, was constructed by two brothers simultaneously from either side — the halves are almost but not quite symmetrical. The yellow sandstone facade with its peacocks, elephants, and floral arabesques is extraordinary. The Salim Singh Ki Haveli (c. 1815) is famous for its overhanging top floor shaped like a peacock. Entry: Rs.20–50.","tags":["Haveli","Architecture","Heritage"]},
                "afternoon": {"name":"Jaisalmer Fort — Inside the Living Walls","desc":"Return to Sonar Quila to explore the life within the walls: the old Maharaja's palace (now a museum), the 12th-century Laxminath Temple, the water cisterns that supplied the fort during sieges, and the views from the cannon bastions across the desert to the horizon. The fort is still home to 3,000 people — families who have lived here for centuries. Entry: Rs.50 (Indians).","tags":["Fort","Living Heritage","History"]},
                "evening":   {"name":"Gadisar Lake by Night","desc":"Gadisar Lake at night — the chhatris and small temples reflected in the still water, the old city walls glowing against a sky full of desert stars — is one of Jaisalmer's most magical experiences. A small boat ride (Rs.50) crosses to the island shrine. The surrounding ghats are lit with small earthen lamps on auspicious evenings. Cost: Free.","tags":["Lake","Night","Photography"]}},
            4: {"title": "Thar Desert — Fossil Park & Remote Dunes",
                "morning":   {"name":"Akal Wood Fossil Park","desc":"17km from Jaisalmer, the Akal Fossil Park contains 180-million-year-old fossilised wood — fragments of a forest that existed when this desert was tropical jungle. The largest fossil is 13m long and 1.5m in diameter. The park is situated in striking badlands landscape of eroded yellow sandstone. Entry: Rs.25 (Indians).","tags":["Geology","Fossils","Nature"]},
                "afternoon": {"name":"Khuri Sand Dunes & Mud Village","desc":"40km south of Jaisalmer, Khuri has smaller, gentler dunes than Sam but is far more authentic — a Rajasthani mud-brick village where families offer camel rides, folk music performances, and homestays. The Sodha Rajputs of Khuri have maintained their desert hospitality traditions for centuries. Cost: Rs.500–2,000.","tags":["Village","Desert","Culture"]},
                "evening":   {"name":"Desert Overnight Camp — Stargazing","desc":"The Thar Desert, far from city light pollution, has extraordinary dark skies. An overnight camp between the Sam and Khuri dunes offers a bonfire, Rajasthani folk music, and a sky so full of stars that the Milky Way is visible with the naked eye. The silence of the desert at 2am is unforgettable. Cost: Rs.2,000–5,000.","tags":["Desert","Night","Adventure"]}},
            5: {"title": "Barmer & Crafts of the Thar",
                "morning":   {"name":"Barmer Craft Heritage","desc":"100km south of Jaisalmer, Barmer is Rajasthan's craft heartland — producing the finest Ajrak block-printed fabric (geometric resist-print in indigo and madder), Barmer applique (colourful patchwork), and wooden furniture with brass inlay. The town's artisan cooperatives sell directly. The Barmer Arts Trust has a remarkable workshop-museum. Entry: Free.","tags":["Craft","Textile","Art"]},
                "afternoon": {"name":"Kiradu Temples — Rajasthan's Khajuraho","desc":"35km from Barmer, the Kiradu temple complex (11th–12th century) contains five exquisitely carved Shaivite temples in a style closely related to Solanki architecture. The temples are nearly deserted — fewer than 100 tourists visit monthly. The sculptures include erotic panels, celestial musicians, and guardian figures of exceptional quality. Entry: Free.","tags":["Temple","Sculpture","Hidden Gem"]},
                "evening":   {"name":"Return Across the Thar at Sunset","desc":"The 2-hour drive back from Barmer to Jaisalmer at sunset crosses the heart of the Thar — the scrubland, khejri trees, peacocks, and the occasional herd of camels silhouetted against the red horizon. This is the Thar at its most honest — not a tourist attraction but a living desert with its own rhythms. Cost: Hired car Rs.2,500–3,500.","tags":["Desert","Drive","Nature"]}},
            6: {"title": "Longewala & Border Heritage",
                "morning":   {"name":"Longewala Battle Memorial","desc":"120km from Jaisalmer, the Longewala War Memorial marks the site of the Battle of Longewala (December 1971) — where 120 Indian soldiers with one jeep-mounted recoilless rifle and air support held off a Pakistani armoured division of 2,000 soldiers and 45 tanks for an entire night. Pakistani tanks and vehicles still litter the battlefield. Entry: Free.","tags":["Military","1971 War","History"]},
                "afternoon": {"name":"Tanot Mata Temple & Border","desc":"130km from Jaisalmer near the Pakistan border, the Tanot Mata Temple is tended by the Indian Border Security Force. During the 1965 and 1971 wars, the temple allegedly survived over 3,000 bombs dropped by Pakistani aircraft without a single casualty — the unexploded bombs are displayed in the temple museum. Entry: Free.","tags":["Sacred","Military","Border"]},
                "evening":   {"name":"Sunset at Ramkunda","desc":"The Ramkunda step-well and temple complex, 35km from Jaisalmer on the Jodpur road, is a perfectly preserved 14th-century Rajput water heritage site — rarely visited by tourists. The yellow sandstone tank and its carved pavilions at sunset, with the desert silence and a peacock's call, is one of the Thar's most serene moments. Entry: Free.","tags":["Heritage","Stepwell","Sunset"]}},
            7: {"title": "Jaisalmer's Merchant Heritage",
                "morning":   {"name":"Jaisalmer Folk Art Museum","desc":"The small but excellent Jaisalmer Folk Arts Museum near Gadisar Lake displays the domestic material culture of the Thar Desert — decorated chests, embroidered textiles, camel trappings, musical instruments, and turbans from 30 different communities. The collection was assembled from Thar villages and represents an endangered way of life. Entry: Rs.30 (Indians).","tags":["Folk Art","Museum","Culture"]},
                "afternoon": {"name":"Pattwa Haveli Deep Dive","desc":"Return to the Patwon Ki Haveli for the interior — the upper floors contain the restored apartment of a 19th-century Jain merchant family, with original furniture, Belgian glass windows, and painted ceilings. The 60 carved jharokha (cantilevered balconies) on the facade are each unique — a reference set for all Rajasthani haveli architecture. Entry: Rs.50–100.","tags":["Haveli","Architecture","Heritage"]},
                "evening":   {"name":"Jaisalmer Cuisine — Desert Flavours","desc":"Jaisalmer's cuisine evolved from desert necessity: ker sangri (dried wild berries and beans cooked with five spices), bajra roti (millet flatbread), and the mutton laal maas (fiery red meat curry). An evening meal at a haveli rooftop restaurant with views of the golden fort lit against the desert sky — the sand turning copper at dusk — is the perfect Jaisalmer evening. Cost: Rs.400–800.","tags":["Food","Culture","Heritage"]}},
            8: {"title": "Ancient Trade Routes of the Thar",
                "morning":   {"name":"Lodurva Jain Temples","desc":"15km from Jaisalmer, Lodurva was the ancient capital of the Bhati Rajputs before Jaisalmer was founded. The Jain temples here (rebuilt 1615) are remarkable for their unusual kalpavriksha (wish-fulfilling tree) carved in white marble within the inner sanctuary. The isolated desert setting — far from town — adds to the temples' mystical quality. Entry: Free.","tags":["Jain","Temple","Heritage"]},
                "afternoon": {"name":"Kanoi Village & Thar Ecology Walk","desc":"The village of Kanoi, 20km from Jaisalmer, is the starting point for a 3-hour guided walk through the Thar Desert ecology — identifying native plants (khejri, rohida, ber), tracking desert fox, and visiting a traditional Persian wheel well still in use. The khejri tree (Jaisalmer's state tree) was the subject of the Bishnoi martyrdom in 1730. Cost: Guide Rs.500.","tags":["Ecology","Village","Nature"]},
                "evening":   {"name":"Desert Evening Music & Dance","desc":"Jaisalmer's Langas and Manganiyars are hereditary folk musicians from Muslim communities who have performed for Rajput patrons for centuries — their songs are a unique blend of Sufi devotional music, folk ballads, and desert poetry. An intimate evening performance arranged through a haveli guesthouse is the most authentic way to experience this dying tradition. Cost: Rs.500–2,000.","tags":["Music","Folk","Culture"]}},
            9: {"title": "Pokaran & Nuclear Heritage",
                "morning":   {"name":"Pokaran Fort & Museum","desc":"110km from Jaisalmer, Pokaran Fort (14th century) is where India conducted its first nuclear test (Operation Smiling Buddha) on 18 May 1974 — the first nuclear test by a non-WWII power. The fort houses a small but interesting museum of Rajput weapons and local history. The town is also famous for its distinctive yellow sandstone handicrafts. Entry: Rs.25 (Indians).","tags":["Fort","History","Heritage"]},
                "afternoon": {"name":"Ramdeora Temple & Desert Saints","desc":"12km from Pokaran, the Ramdeora Temple is dedicated to Baba Ramdev — a 14th-century Rajput prince who became a revered saint worshipped equally by Hindus and Muslims. His annual fair (August–September) draws 200,000 pilgrims. The temple is the largest pilgrimage site in the western Thar desert. Entry: Free.","tags":["Pilgrimage","Saint","Culture"]},
                "evening":   {"name":"Pokaran Stone Craft Workshops","desc":"Pokaran's pale yellow sandstone is unique in the Thar — softer and lighter than Jaisalmer stone, carved into decorative columns, jali screens, and animal figures by local craftsmen using techniques unchanged for 500 years. A late afternoon workshop visit and the drive back to Jaisalmer through the desert sunset wraps up this deep journey into the Thar. Cost: Free.","tags":["Craft","Stone","Art"]}},
            10: {"title": "Jaisalmer Farewell — Fort & Desert",
                "morning":   {"name":"Final Fort Sunrise — Cannon Bastion","desc":"Return to Jaisalmer Fort for a final sunrise from the Dussehra Chowk and the northern cannon bastion. As the sun rises over the Thar the golden sandstone catches the first light and the fort seems to glow from within — the image that gave Sonar Quila its name. The Jain temples open at 7am for an early morning prayer. Entry: Rs.50 (Indians).","tags":["Fort","Sunrise","Photography"]},
                "afternoon": {"name":"Desert Medicine & Herbals — Thar Market","desc":"Jaisalmer's market stalls around the fort gate sell desert medicinal herbs, spices, and traditional Ayurvedic preparations used by the Bishnoi and Rajput communities — dried sangri, ker, the desert rose, and rare desert gum used in Rajasthani sweets. The market is also the best place for lac bangles and camel-leather goods. Cost: Rs.200–1,000.","tags":["Shopping","Craft","Food"]},
                "evening":   {"name":"Farewell at the Fort — Rooftop Sunset","desc":"End your Jaisalmer journey at a fort rooftop cafe watching the sun descend into the Thar — the Patwon Ki Haveli catching the last light, the city walls golden below, the desert stretching silent to Pakistan. Order a chai and let the golden city hold you for one last hour. Cost: Rs.100–300.","tags":["Sunset","Heritage","Farewell"]}},
        }
    },
    "thanjavur": {
        "name": "Thanjavur, Tamil Nadu", "emoji": "TJR",
        "tagline": "The Temple City of the Chola Empire",
        "monuments": ["Brihadeeswara Temple","Thanjavur Palace","Saraswati Mahal Library","Gangaikonda Cholapuram","Airavatesvara Temple","Art Gallery"],
        "budget": {"budget": "Rs.1,800", "mid": "Rs.5,500", "luxury": "Rs.15,000"},
        "days": {
            1: {"title": "The Great Living Chola Temple",
                "morning":   {"name":"Brihadeeswara Temple at Dawn","desc":"Built by Chola Emperor Raja Raja I between 985–1014 CE, the Brihadeeswara (Big Temple) is one of the great masterpieces of Indian architecture. The 66m vimana (tower) was built without scaffolding using an inclined ramp 6km long. The capstone alone weighs 80 tonnes. The entire complex was built in under 7 years. Entry: Free.","tags":["UNESCO","Chola","Temple"]},
                "afternoon": {"name":"Thanjavur Palace & Art Gallery","desc":"The Nayak Palace (16th century), later expanded by the Marathas, is a huge complex with towers, art galleries, and the Saraswati Mahal Library — one of Asia's oldest libraries, housing 66,000 manuscripts on palm leaf and paper, including a rare complete set of Indian astronomical texts. Entry: Rs.10 (Indians).","tags":["Palace","Library","Art"]},
                "evening":   {"name":"Bronze Sculpture Walk & Carnatic Music","desc":"Thanjavur is the world capital of Chola bronze casting — the Nataraja and Parvati bronzes made here are in museums from London to New York. Visit a traditional foundry using the ancient lost-wax (cire perdue) casting method. Thanjavur is also the birthplace of the Carnatic classical music tradition — attend an evening performance if possible.","tags":["Art","Craft","Music"]}},
            2: {"title": "Gangaikonda Cholapuram & Chola Legacy",
                "morning":   {"name":"Gangaikonda Cholapuram","desc":"Built by Rajendra Chola I in 1035 CE to celebrate his military campaign to the Ganges, this UNESCO temple was designed to surpass even the Brihadeeswara at Thanjavur. Its 53m tower has a concave curvature unique in Indian architecture. The temple is 70km from Thanjavur but essential for understanding the Chola legacy. Entry: Free.","tags":["UNESCO","Chola","Architecture"]},
                "afternoon": {"name":"Airavatesvara Temple, Darasuram","desc":"Built by Chola Emperor Rajaraja II in the 12th century, this UNESCO temple at Darasuram (near Kumbakonam, 40km) is famous for its exquisitely detailed sculptures and the unique stepped vimana with a chariot base — the stone wheels are said to have once rotated. Entry: Free.","tags":["UNESCO","Chola","Sculpture"]},
                "evening":   {"name":"Kaveri Delta Villages & Silk Weaving","desc":"The Kaveri delta villages around Thanjavur produce the famous Thanjavur silk and Tanjore paintings — a classical South Indian style with gold foil and semi-precious stone inlays on a wooden base, dating back to the Maratha rule (c. 1676–1855). Visit an artist's workshop to see the layering process firsthand.","tags":["Art","Craft","Culture"]}},
            3: {"title": "Kumbakonam — Temple Town of the Delta",
                "morning":   {"name":"Kumbakonam Mahamaham Tank","desc":"40km from Thanjavur, Kumbakonam is known as the 'City of Temples' — it has 12 major temples within 3 sq km. The Mahamaham Tank (500m x 250m) is one of South India's holiest — pilgrims believe bathing here during the Mahamaham festival (once every 12 years) is equivalent to bathing in all nine sacred rivers of India. Entry: Free.","tags":["Temple","Pilgrimage","Sacred"]},
                "afternoon": {"name":"Sarangapani & Chakrapani Temples","desc":"The Sarangapani Temple (13th century), dedicated to Vishnu, has a gopuram 45m tall and a remarkable inner sanctum housing a 4m reclining Vishnu image. The Chakrapani Temple's annual chariot festival draws 100,000 devotees. The temples are extraordinarily well-preserved Chola-Nayak structures. Entry: Free.","tags":["Temple","Vaishnavism","Chola"]},
                "evening":   {"name":"Kumbakonam Filter Coffee & Sweets","desc":"Kumbakonam is famous as the source of South India's finest filter coffee — the rich, dark Kumbakonam degree coffee, made from a special coffee-chicory blend. The town's sweet shops produce vethalai (betel leaf), cheedai, and the famous murukku varieties. A walk through the evening market is vivid and fragrant. Cost: Rs.50–200.","tags":["Food","Culture","Heritage"]}},
            4: {"title": "Chidambaram — The Cosmic Dance",
                "morning":   {"name":"Nataraja Temple, Chidambaram","desc":"100km from Thanjavur, the Chidambaram Nataraja Temple is the most important Shaivite temple in South India — the abode of Shiva as Nataraja (Lord of the Cosmic Dance). The temple's golden gopuram marks the spot where Shiva performed the Ananda Tandava (dance of bliss). The Chit Sabha (crystal hall) enshrines the 'space lingam' — a lingam that is visible only by faith. Entry: Free.","tags":["Temple","Shiva","Nataraja"]},
                "afternoon": {"name":"Pichavaram Mangrove Forest","desc":"15km from Chidambaram, the Pichavaram mangrove forest is the second largest mangrove forest in India — a 1,100-hectare labyrinth of waterways between thousands of mangrove islands. A boat ride through the channels at low tide — when the roots arch above the water — is one of Tamil Nadu's most otherworldly natural experiences. Entry: Rs.50 (boat extra).","tags":["Nature","Mangrove","Ecology"]},
                "evening":   {"name":"Chidambaram Evening Puja","desc":"The evening puja at Chidambaram Nataraja Temple is one of Tamil Nadu's most elaborate rituals — 108 classical Bharatanatyam dance poses inscribed on the temple walls. The evening arati with 10,000 oil lamps is extraordinary. Return to Thanjavur. Entry: Free.","tags":["Ritual","Temple","Sacred"]}},
            5: {"title": "Trichy & Srirangam — Vishnu's Greatest Temple",
                "morning":   {"name":"Srirangam — World's Largest Functioning Temple","desc":"55km from Thanjavur, the Sri Ranganathaswamy Temple at Srirangam is the largest functioning Hindu temple in the world — covering 156 acres with 21 gopurams. Seven concentric walls enclose the inner sanctum. The outermost wall alone is over 4km in circumference. 50,000 pilgrims visit daily. Entry: Free (inner sanctum Rs.50).","tags":["Temple","Vaishnavism","Largest"]},
                "afternoon": {"name":"Rock Fort Temple, Trichy","desc":"The Ucchi Pillayar Temple atop the 83m monolithic Rock Fort of Trichy was carved by the Pallava dynasty in the 7th century. The 437 rock-cut steps lead to the summit for a panoramic view of Srirangam, the Kaveri, and the Coleroon rivers. The Vibishana Temple below dates to the 8th century. Entry: Rs.5 (Indians).","tags":["Temple","Rock-Cut","Views"]},
                "evening":   {"name":"Kaveri River at Sunset","desc":"The Kaveri river at Trichy splits into the Coleroon channel and the main river, creating the sacred island of Srirangam. An evening walk along the Kaveri ghats — where fishermen pull in their nets and parakeets roost in the palms — is a gentle, beautiful close to a day of grand temples. Cost: Free.","tags":["Nature","River","Sunset"]}},
            6: {"title": "Tanjore Painting & Bronze Masterclass",
                "morning":   {"name":"Tanjore Painting Workshop","desc":"The Tanjore painting tradition, established under Maratha patronage in the 18th century, uses semi-precious stones (rubies, emeralds, pearls) embedded in gold foil on wooden boards. A full workshop session with a master artist shows all 7 layers: gesso, charcoal sketch, jewel-setting, gold-leaf application, colour, and varnish. Cost: Rs.500–1,500/session.","tags":["Art","Craft","Heritage"]},
                "afternoon": {"name":"Chola Bronze Casting Foundry","desc":"Thanjavur's Swamimalai village (20km) is the centre of the living Chola bronze-casting tradition. The Sthapatya Veda (ancient Tamil sculptural canon) is still followed by hereditary sthapati (sculptor) families using the lost-wax cire perdue process. Watch an entire bronze figure being made from wax model to finished casting. Entry: Free (workshop Rs.300–500).","tags":["Bronze","Craft","Chola"]},
                "evening":   {"name":"Saraswati Mahal Library Night Tour","desc":"The Saraswati Mahal Library in Thanjavur Palace is open for evening visits. Its 66,000 manuscripts include astronomical texts, musical treatises, anatomical drawings, and a complete copy of every major Sanskrit work. The oldest palm-leaf manuscript dates to 1000 CE. Entry: Rs.10 (Indians).","tags":["Library","History","Culture"]}},
            7: {"title": "Velankanni & Coastal Heritage",
                "morning":   {"name":"Velankanni Basilica — Our Lady of Good Health","desc":"60km from Thanjavur, the Basilica of Our Lady of Good Health at Velankanni is one of the most visited Catholic pilgrimage sites in Asia — drawing 2 million pilgrims annually from across India. The white Gothic church (1881) is on the site of three apparitions of the Virgin Mary reported between the 16th and 17th centuries. Entry: Free.","tags":["Christian","Pilgrimage","Heritage"]},
                "afternoon": {"name":"Nagapattinam & Tsunami Heritage","desc":"Nagapattinam, 10km from Velankanni, was the town hardest hit by the 2004 Indian Ocean tsunami. The Tsunami Memorial Museum and the reconstructed fishing village tell the story of the disaster and the remarkable recovery. The medieval Kayarohana Swami Temple here dates to the 3rd century CE. Entry: Free.","tags":["History","Museum","Heritage"]},
                "evening":   {"name":"Point Calimere Wildlife Sanctuary","desc":"45km from Nagapattinam, Point Calimere is a Ramsar-designated coastal wetland and wildlife sanctuary famous for its flamingo flocks (November–January) — up to 30,000 gather here. The sanctuary also protects the last significant herd of blackbuck in Tamil Nadu. Evening is the best time for sightings. Entry: Rs.50 (Indians).","tags":["Wildlife","Birds","Nature"]}},
            8: {"title": "Thanjavur's Musical Heritage",
                "morning":   {"name":"Thanjavur Veena & Musical Instrument Craft","desc":"Thanjavur is the birthplace of the Saraswati veena — the classical South Indian lute, the national instrument of India. The veena makers of Thanjavur are hereditary craftsmen who hollow a single jackwood block using chisels to create the instrument body. A workshop visit shows the 3-week making process. The instrument is also sacred — Goddess Saraswati is always depicted playing it. Cost: Free.","tags":["Music","Craft","Heritage"]},
                "afternoon": {"name":"Bharatanatyam Cultural Performance","desc":"Thanjavur is the birthplace of Bharatanatyam — India's oldest classical dance form, codified from the Devadasi tradition of the Brihadeeswara Temple in the 19th century by the Thanjavur Quartet. An afternoon performance at a cultural academy shows the Bharatanatyam natyarambha (pure dance) and abhinaya (expressive) sections. Cost: Rs.200–500.","tags":["Dance","Culture","Heritage"]},
                "evening":   {"name":"Brihadeeswara Temple — Full Moon Night","desc":"Return to the Brihadeeswara Temple on a full moon night when the white vimana glows silver against a dark sky and the white light transforms the entire complex. The temple is open until 9pm. Evening puja with oil lamps is at 8pm. Entry: Free.","tags":["UNESCO","Temple","Photography"]}},
            9: {"title": "Swamimalai & the Five Sabhas",
                "morning":   {"name":"Swamimalai Murugan Temple","desc":"Swamimalai, 6km from Kumbakonam, is one of the six Aarupadai Veedu — the six sacred abodes of Lord Murugan (Kartikeya) in Tamil Nadu. The temple is built on a hillock of 60 steps representing the 60 Tamil years. The Swamimalai Murugan idol is unique — at Swamimalai, Murugan is said to have taught the Pranava Mantra (Om) to his own father Shiva. Entry: Free.","tags":["Temple","Murugan","Pilgrimage"]},
                "afternoon": {"name":"Chidambaram's Five Cosmic Dance Halls","desc":"Return to Chidambaram to explore the Pancha Sabhas — the five cosmic dance halls of Shiva in Tamil Nadu. Chidambaram is the Chit Sabha (sky/space). The other four (fire, water, earth, wind) are at Thiruvannamalai, Thiruvanaikaval, Kanchipuram, and Kutralam. Understanding this cosmological geography transforms every temple you've visited. Entry: Free.","tags":["Temple","Shiva","Cosmology"]},
                "evening":   {"name":"Kaveri Delta Sunset Boat Ride","desc":"A sunset boat ride on one of the Kaveri delta channels near Kumbakonam — through rice paddies, coconut groves, and temple gopurams rising above the fields — captures the ancient agricultural civilisation that the Chola Empire was built on. The same landscape has fed South India's greatest culture for 2,000 years. Cost: Rs.200–500.","tags":["Nature","River","Heritage"]}},
            10: {"title": "Farewell Thanjavur — Big Temple at First Light",
                "morning":   {"name":"Brihadeeswara at Dawn — Final Visit","desc":"Be at the Brihadeeswara Temple at 7am for opening prayers — the vimana lit gold by the rising sun, the Nandi facing the lingam, and priests performing the dawn puja with camphor flames and Sanskrit chants. This is the 1,000-year-old ritual performed at this exact spot since Raja Raja I completed the temple in 1010 CE. Entry: Free.","tags":["UNESCO","Temple","Farewell"]},
                "afternoon": {"name":"Brihadeeswara Museum & Inscriptions","desc":"The Archaeological Survey museum within the Brihadeeswara complex houses original Chola bronzes, inscribed copper plates recording land grants, and scale architectural drawings of the temple. The temple's outer walls carry 900 inscriptions in Tamil and Grantha — the most extensive epigraphic record of any temple in India. Entry: Free.","tags":["Museum","Archaeology","Chola"]},
                "evening":   {"name":"Farewell Chettinad Dinner","desc":"Chettinad cuisine — from the merchant community of the Kaveri delta — is among India's most complex: slow-cooked with kalpasi (stone flower), marathi mokku (dried flower pods), and freshly ground spice blends. A proper Chettinad meal (chicken kuzhambu, vendakkai poriyal, appam) in Thanjavur is the finest farewell to Tamil Nadu. Cost: Rs.200–500.","tags":["Food","Chettinad","Farewell"]}},
        }
    },
    "madurai": {
        "name": "Madurai, Tamil Nadu", "emoji": "MDU",
        "tagline": "City of the Never-Sleeping Temple",
        "monuments": ["Meenakshi Amman Temple","Thirumalai Nayak Palace","Gandhi Museum","Alagar Kovil","Koodal Azhagar Temple","Vandiyur Mariamman Teppakulam"],
        "budget": {"budget": "Rs.1,800", "mid": "Rs.5,500", "luxury": "Rs.15,000"},
        "days": {
            1: {"title": "The Meenakshi Temple & Sacred City",
                "morning":   {"name":"Meenakshi Amman Temple at Dawn","desc":"One of the largest temple complexes in India, covering 60 acres with 14 gopurams (towers). The tallest gopuram is 52m and covered with 33,000 painted stucco sculptures. The temple is never closed — it operates 24 hours and is administered separately from the state. Over 30,000 pilgrims visit daily. Founded in the 7th century, the current structure dates mainly to the 17th century. Entry: Free (camera Rs.50).","tags":["Temple","Dravidian","Pilgrimage"]},
                "afternoon": {"name":"Thirumalai Nayak Palace","desc":"Built in 1636 by Nayak ruler Thirumalai, this Indo-Saracenic palace was once four times its current size. The remaining Swarga Vilasam (celestial pavilion) has stucco columns 12m tall and carved stucco of exceptional quality. An evening Sound & Light show tells the story of the Nayak dynasty. Entry: Rs.20 (Indians).","tags":["Palace","Nayak","Architecture"]},
                "evening":   {"name":"Meenakshi Temple Night Ceremony","desc":"Every night without exception, the Chithirai festival ceremony is replicated: Lord Alagar's idol is taken in a gold palanquin to sleep in the inner sanctum of Goddess Meenakshi. Witnessing the elaborate procession with music, incense, and devoted priests is one of Tamil Nadu's great religious experiences. Entry: Free.","tags":["Ritual","Temple","Sacred"]}},
            2: {"title": "Gandhi Museum & Temples of the Valley",
                "morning":   {"name":"Gandhi Museum","desc":"Housed in the historic Tamukkam Palace (1670s), this museum is one of India's most comprehensive Gandhi museums. It contains the blood-stained dhoti worn by Gandhi on the day of his assassination (30 January 1948) and extensive exhibits on the independence movement in South India. Entry: Free.","tags":["Independence","History","Museum"]},
                "afternoon": {"name":"Alagar Kovil & Azhagar Hills","desc":"The Kallazhagar Temple at Alagar Kovil, 21km from Madurai, is one of the 108 Divya Desams (sacred Vishnu shrines) of the Vaishnava tradition. Set in the forested Azhagar Hills, the temple and its processional chariot route — used during the annual Chithirai festival — are among Tamil Nadu's most atmospheric heritage sites. Entry: Free.","tags":["Temple","Vaishnavism","Nature"]},
                "evening":   {"name":"Madurai Food Trail & Kothu Parotta","desc":"Madurai has one of Tamil Nadu's most distinctive food cultures. Try jigarthanda (a milk-based drink with nannari syrup and ice cream), mutton kothu parotta (shredded layered bread stir-fried with eggs and meat), and finish with filter coffee at a traditional Brahmin tiffin shop. Cost: Rs.100–300.","tags":["Food","Culture","Street Food"]}},
            3: {"title": "Koodal Azhagar & Floating Temple Festival",
                "morning":   {"name":"Koodal Azhagar Temple","desc":"One of Madurai's 108 Divya Desam temples, the Koodal Azhagar Temple is dedicated to Vishnu in three postures — standing, sitting, and reclining — on three separate floors, unique in South Indian temple architecture. The temple's annual chariot procession runs through the heart of the old city. Entry: Free.","tags":["Temple","Vaishnavism","Architecture"]},
                "afternoon": {"name":"Vandiyur Mariamman Teppakulam","desc":"The vast square tank at Vandiyur is Madurai's most sacred festival site — the Teppam (floating festival) held annually in the Tamil month of Thai sees a raft carrying the idol of Meenakshi and Sundareswarar rowed across the illuminated tank. The 1.5 sq km tank with its central island mandapam is beautiful at all times. Entry: Free.","tags":["Festival","Sacred","Heritage"]},
                "evening":   {"name":"Meenakshi Temple Night Walk","desc":"A night walk through the 4km of corridors surrounding the Meenakshi Temple's inner sanctum — the Hall of 1,000 Pillars (which has 985 actual pillars), the Ashta Shakti Mandapam, and the golden lily tank — all lit at night is extraordinary. The corridors are never empty; pilgrims circumambulate day and night. Entry: Free.","tags":["Temple","Heritage","Night"]}},
            4: {"title": "Rameshwaram — Land's End Pilgrimage",
                "morning":   {"name":"Ramanathaswamy Temple, Rameshwaram","desc":"170km from Madurai, Rameshwaram is one of the Char Dham pilgrimage sites. The Ramanathaswamy Temple has the longest temple corridor in the world — 1,212m, with 1,212 pillars in the outer corridor. The 22 sacred wells (theerthams) within the temple complex require ritual bathing in each — pilgrims emerge drenched in holy water. Entry: Free.","tags":["Pilgrimage","Temple","Char Dham"]},
                "afternoon": {"name":"Pamban Bridge & Adam's Bridge","desc":"The Pamban Bridge (1914) connects Rameshwaram island to mainland India — a 2.3km railway bridge that opens to allow ships through. Adam's Bridge (Ram Setu) — the chain of shoals between India and Sri Lanka — is visible from the Dhanushkodi spit. According to the Ramayana, this was the bridge built by the Vanara army to reach Lanka. Entry: Free.","tags":["Heritage","Geology","Ramayana"]},
                "evening":   {"name":"Dhanushkodi — Land's End","desc":"At the southernmost tip of Rameshwaram island, the ghost town of Dhanushkodi was destroyed by a cyclone in 1964 and never rebuilt. The ruins of the old railway station, church, and customs house stand in the sea. The spit of land between the Bay of Bengal and the Palk Strait at sunset — where two seas meet — is one of India's most dramatic landscapes. Entry: Free.","tags":["Nature","History","Sunset"]}},
            5: {"title": "Chettinad — Merchant Mansions",
                "morning":   {"name":"Karaikudi & Chettinad Palaces","desc":"100km from Madurai, Chettinad is a unique micro-region of 96 villages where the Nattukotai Chettiar merchant community built extraordinary mansions (1800–1940) using materials imported from Burma, Italy, Belgium, and France. The largest, Chettinad Palace in Kanadukathan, has 10 courtyards and 20,000 sq ft of floor space. Entry: Rs.100–200.","tags":["Architecture","Heritage","Merchant"]},
                "afternoon": {"name":"Chettinad Antique Market","desc":"Chettinad's mansions have been steadily stripped of their contents — carved teak pillars, Belgian glass, Burmese teak furniture, and Athangudi tiles — which are sold in antique shops throughout the region. The Athangudi handmade cement tile (made with the Kaveri sand unique to the region) is now internationally recognised as a heritage craft. Cost: Tiles from Rs.50/piece.","tags":["Antiques","Craft","Heritage"]},
                "evening":   {"name":"Chettinad Cuisine Dinner","desc":"Chettinad cuisine is widely regarded as the most complex regional cuisine in India — dishes use over 20 spices including star anise, kalpasi (stone flower), and dried kaya (raw banana). A family-cooked dinner in a Chettinad heritage guesthouse — mutton kuzhambu, prawn masala, idiyappam, and coconut desserts — is the finest meal in South India. Cost: Rs.500–1,000.","tags":["Food","Chettinad","Culture"]}},
            6: {"title": "Kodaikanal — The Princess of Hill Stations",
                "morning":   {"name":"Kodaikanal Lake & Coaker's Walk","desc":"120km from Madurai at 2,133m, Kodaikanal's star-shaped lake was created artificially in 1863 by the British. Coaker's Walk is a 1km cliff promenade with views 2,000m down to the Vaigai plains — on clear days Madurai's Meenakshi Temple gopurams are visible. The Kodaikanal orchid and herb gardens are among South India's best. Entry: Free.","tags":["Nature","Hill Station","Views"]},
                "afternoon": {"name":"Pillar Rocks & Bear Shola Falls","desc":"The Pillar Rocks — three granite pillars 122m tall rising from the Palani Hills — are one of the Kodaikanal's most dramatic natural features. Bear Shola Falls, a seasonal waterfall in a forest grove 5km from town, is a favourite birdwatching spot. The surrounding forests contain over 300 bird species. Entry: Rs.10 (Indians).","tags":["Nature","Geology","Wildlife"]},
                "evening":   {"name":"Kurinji Andavar Temple & Shola Forest","desc":"The Kurinji Andavar Temple, 3km from Kodaikanal, is dedicated to Lord Murugan, whose symbol is the kurinji flower (Strobilanthes kunthiana) — which blooms once every 12 years, covering the entire Palani Hills in blue-purple. The temple's forest setting and the shola (cloud forest) ecosystem are extraordinarily beautiful at dusk. Entry: Free.","tags":["Temple","Nature","Forest"]}},
            7: {"title": "Palani & Murugan Temples Circuit",
                "morning":   {"name":"Palani Dhandayuthapani Temple","desc":"65km from Madurai, the Palani Temple atop a 152m rocky hill is the most visited of the six Murugan abodes — drawing 30,000 pilgrims daily. The idol was made from Navapashanam (nine poisonous substances combined to create a curative substance) by the Siddha sage Bogar in antiquity. Pilgrims climb 693 steps or take the ropeway. Entry: Free.","tags":["Temple","Murugan","Pilgrimage"]},
                "afternoon": {"name":"Sirumalai Hills & Waterfalls","desc":"30km from Dindigul, the Sirumalai Hills (1,600m) are a less visited range of the Western Ghats with coffee and pepper estates, waterfalls, and excellent birding. The Malaiyadipatti rock-cut Jain caves (9th century) are rarely visited heritage gems carved into the hillside. Entry: Free.","tags":["Nature","Jain","Hills"]},
                "evening":   {"name":"Dindigul Thalapakatti Biryani","desc":"Dindigul's seeraga samba rice biryani — made with the tiny local rice variety and Seeraga Samba spices — is universally acknowledged as Tamil Nadu's finest biryani. The Thalapakatti Biryani brand originated here in 1957. A dinner at the original Dindigul outlet before returning to Madurai is a Tamil Nadu culinary pilgrimage. Cost: Rs.150–300.","tags":["Food","Biryani","Culture"]}},
            8: {"title": "Tirunelveli & Kanyakumari",
                "morning":   {"name":"Kanyakumari — India's Southernmost Point","desc":"250km from Madurai, Kanyakumari is where three seas meet — the Arabian Sea, the Bay of Bengal, and the Indian Ocean. The sunrise here shows three colours simultaneously. The Kumari Amman Temple (3rd century BCE) and the Vivekananda Rock Memorial (1970) stand on the promontory. Entry: Free (ferry to rock Rs.34).","tags":["Pilgrimage","Nature","Landmark"]},
                "afternoon": {"name":"Tirunelveli Halwa & Cathedral Heritage","desc":"150km from Madurai, Tirunelveli is famous for the world's first halwa — Tirunelveli Iruttu Kadai Halwa (dark shop halwa), made from wheat and pure ghee since the 18th century. The town also has the 16th-century Christ Church cathedral and the large Nellaiappar Temple complex. Cost: Halwa Rs.200–500/kg.","tags":["Food","Heritage","Culture"]},
                "evening":   {"name":"Courtallam Falls — Niagara of the South","desc":"170km from Madurai, Courtallam (Kutralam) is famous for its medicinal waterfalls — 9 falls over a 12km stretch, each said to have different curative properties from the herbs dissolved in the water. The Main Falls and the Five Falls are accessible year-round and reach their peak during the monsoon. Entry: Free.","tags":["Waterfall","Nature","Wellness"]}},
            9: {"title": "Madurai Deep Dive — Temple Architecture",
                "morning":   {"name":"Meenakshi Temple — Full Archaeological Tour","desc":"A comprehensive 3-hour architectural tour of the Meenakshi complex: the Hall of 1,000 Pillars (built 1569–1572, now a museum), the golden lily tank (Porthamarai Kulam), the Musical Pillars of the Kili Kootu Mandapam, and the Adi Shakti shrine — the original 2nd-century BCE temple within the 16th-century complex. Entry: Free (museum Rs.5).","tags":["UNESCO","Architecture","Archaeology"]},
                "afternoon": {"name":"Pasumalai Hills & Pazhamudhircholai","desc":"The Pazhamudhircholai hilltop temple is one of the six Murugan abodes — set in dense forest 15km from Madurai. The 8-km drive through the forest is home to peacocks, langurs, and spotted deer. The temple at the summit has extraordinary views over the Madurai plain with the Meenakshi Temple visible in the distance. Entry: Free.","tags":["Temple","Nature","Views"]},
                "evening":   {"name":"Madurai Arts & Crafts — Kondapalli Dolls","desc":"Madurai's craft heritage includes brass idol-casting, wooden toys, and the manufacture of the traditional Madurai Sungudi saree (a fine cotton saree with tie-dye dot patterns recognised as a GI product). Visit the Cottage Industries Exposition and the Tamil Nadu Handicrafts Development Corporation for direct purchases. Cost: Free to browse.","tags":["Craft","Art","Shopping"]}},
            10: {"title": "Farewell Madurai — First Light, Last Temple",
                "morning":   {"name":"Meenakshi Temple at 5am — Dawn Abhishekam","desc":"The Meenakshi Amman Temple performs the dawn abhishekam (ritual bathing of the deity) at 5am — the most sacred ritual of the temple day, when only the most devout pilgrims are present. The scent of incense, the sound of Vedic chanting, and the golden light of the oil lamps in the pre-dawn dark is an experience unavailable at any other hour. Entry: Free.","tags":["Temple","Sacred","Farewell"]},
                "afternoon": {"name":"Kazimar Big Mosque & Dargah","desc":"The Kazimar Big Mosque (1284 CE), built by a Muslim merchant from Arabia who settled in Madurai, is one of the oldest mosques in India outside Kerala. The adjacent Dargah of Hazrat Qutubuddin contains the saint's tomb and draws both Muslim and Hindu devotees — a rare example of syncretic devotion that has characterised Madurai for centuries. Entry: Free.","tags":["Islam","Heritage","History"]},
                "evening":   {"name":"Final Aarti & Jigarthanda","desc":"A final evening at the Meenakshi Temple for the closing ceremony — the god and goddess put to rest for the night with a procession, music, and lamps. Then jigarthanda at Rahmath Jigarthanda on North Chitrai Street — the original recipe with China grass (agar-agar), nannari (sarsaparilla) syrup, almond gum, reduced milk, and ice cream — the perfect farewell. Cost: Rs.80–150.","tags":["Temple","Food","Farewell"]}},
        }
    },
    "mysore": {
        "name": "Mysore, Karnataka", "emoji": "MYS",
        "tagline": "The City of Palaces & Sandalwood",
        "monuments": ["Mysore Palace","Chamundeshwari Temple","St Philomena's Church","Brindavan Gardens","Srirangapatna","Jaganmohan Palace"],
        "budget": {"budget": "Rs.2,500", "mid": "Rs.8,000", "luxury": "Rs.22,000"},
        "days": {
            1: {"title": "The Mysore Palace & Royal Splendour",
                "morning":   {"name":"Mysore Palace at Dawn","desc":"The third most visited monument in India after the Taj Mahal and the Red Fort, this Indo-Saracenic palace was built between 1897–1912 for the Wadiyar dynasty after the original wooden palace burned down. It contains 12 Hindu temples, a Durbar Hall, and a Golden Throne displayed during Dasara. Entry: Rs.70 (Indians).","tags":["Palace","Wadiyar","UNESCO"]},
                "afternoon": {"name":"Chamundeshwari Temple & Nandi Bull","desc":"Perched atop the Chamundi Hills (1,065m) above Mysore, this 12th-century temple dedicated to Goddess Chamundeshwari (the royal deity of the Mysore kingdom) is reached via a stairway of 1,008 steps carved into the hillside. Halfway up stands a monolithic Nandi (sacred bull) bull carved in 1659 — 5m tall and 8m long. Entry: Free.","tags":["Temple","Pilgrimage","Views"]},
                "evening":   {"name":"Mysore Palace Illumination","desc":"Every Sunday and on public holidays, Mysore Palace is lit up with 97,000 light bulbs at 7pm. This spectacle, introduced during the reign of Krishnaraja Wadiyar IV, transforms the palace into one of India's most photographed monuments. The palace grounds are the ideal setting for an evening stroll.","tags":["Illumination","Photography","Heritage"]}},
            2: {"title": "Srirangapatna & Tipu's Legacy",
                "morning":   {"name":"Srirangapatna — Tipu Sultan's Capital","desc":"The island fortress of Srirangapatna, 16km from Mysore, was the capital of Mysore's most celebrated ruler, Tipu Sultan (1782–1799). Visit the Daria Daulat Bagh palace with its extraordinary murals of the Battle of Pollilur, and the Gumbaz mausoleum where Tipu Sultan and his parents are buried.","tags":["Tipu Sultan","History","Fort"]},
                "afternoon": {"name":"Brindavan Gardens & Krishnaraja Sagar","desc":"Built in 1932 when the Krishnaraja Sagar dam was completed across the Kaveri river, the Brindavan Gardens are laid out in the Mughal char bagh style with terraced fountains. The illuminated musical fountain show in the evening is a major attraction in Karnataka.","tags":["Gardens","Dam","Nature"]},
                "evening":   {"name":"Devaraja Market & Mysore Pak","desc":"Mysore's main bazaar, built during the reign of Krishnaraja Wadiyar IV, is a riot of jasmine garlands, turmeric, silk, and incense. Do not leave without trying Mysore Pak — the legendary fudge-like sweet made from ghee, sugar, and chickpea flour, invented in the Mysore Palace kitchen in the early 20th century. Cost: Rs.50–150.","tags":["Food","Shopping","Culture"]}},
            3: {"title": "Belur & Halebidu — Hoysala Temples",
                "morning":   {"name":"Belur Chennakeshava Temple","desc":"150km from Mysore, the Belur Chennakeshava Temple (1117 CE) was built by King Vishnuvardhana to celebrate his conversion from Jainism to Vaishnavism. The sculpted lathe-turned pillars and the 42 bracket figures (madanikas) on the outer wall — celestial women in exquisite detail — took 103 years to complete. Entry: Rs.25 (Indians).","tags":["UNESCO","Hoysala","Temple"]},
                "afternoon": {"name":"Halebidu — Twin Temples of Hoysaleswara","desc":"16km from Belur, the Hoysaleswara Temple at Halebidu (12th century) has the most extensive sculptural programme in India — the outer walls carry a continuous band of 240 elephants, 200 lions, 2,000 horses, scrolling foliage, scenes from the Ramayana and Mahabharata, and thousands of divine figures. No two carvings repeat. Entry: Rs.25 (Indians).","tags":["UNESCO","Hoysala","Sculpture"]},
                "evening":   {"name":"Shravanabelagola — Gommateshvara Colossus","desc":"60km from Halebidu, the Gommateshvara (Bahubali) statue at Shravanabelagola is the world's largest monolithic free-standing stone statue — 57 feet carved from a single granite rock in 981 CE. Reaching it requires climbing 614 steps. Every 12 years (next in 2030) it is anointed in the Mahamastakabhisheka ceremony witnessed by millions. Entry: Rs.5 (Indians).","tags":["Jain","Monolith","Pilgrimage"]}},
            4: {"title": "Wayanad — Tribal & Forest Heritage",
                "morning":   {"name":"Edakkal Caves — Neolithic Rock Carvings","desc":"100km from Mysore in Kerala's Wayanad district, the Edakkal Caves contain the only known Neolithic rock engravings in South India — geometric symbols and human figures carved 6,000 years ago. The two caves at 1,200m are reached by a 1km trek through dense forest. Entry: Rs.30 (Indians).","tags":["Prehistoric","Rock Art","Nature"]},
                "afternoon": {"name":"Wayanad Wildlife Sanctuary","desc":"The 344 sq km Wayanad sanctuary is part of the Nilgiri Biosphere Reserve — one of the world's biodiversity hotspots. Home to tigers, elephants, gaurs, and leopards, the sanctuary's teak and rosewood forests are traversed by ancient elephant corridors used for millennia. Safari: Rs.1,200–2,000.","tags":["Wildlife","Nature","Forest"]},
                "evening":   {"name":"Tribal Heritage Walk — Kurichiya Community","desc":"The Kurichiya tribal community of Wayanad are one of Kerala's most well-preserved indigenous cultures. A guided community walk through a Kurichiya settlement reveals their traditional bamboo architecture, arrow-making craft, and the forest-knowledge system they've maintained for centuries. Cost: Community fee Rs.300.","tags":["Tribal","Culture","Heritage"]}},
            5: {"title": "Coorg — Coffee, Forts & Waterfalls",
                "morning":   {"name":"Abbey Falls & Mercara Fort, Coorg","desc":"120km from Mysore, Coorg (Kodagu) is the coffee capital of India — 1,800m coffee and cardamom estates surround the town of Madikeri (Mercara). The Mercara Fort (1681, rebuilt 1812) in the town centre has a small museum and a church built inside a Tipu Sultan-era mosque. Abbey Falls, 10km away, drops 70m through coffee forest. Entry: Rs.25 (Indians).","tags":["Nature","Fort","Coffee"]},
                "afternoon": {"name":"Raja's Seat & Omkareshwara Temple","desc":"Raja's Seat is the hilltop garden from which Coorg's kings watched the sun set over the Western Ghats. The Omkareshwara Temple (1820) in Madikeri is uniquely designed in a blend of Islamic and Gothic styles — Tipu Sultan ordered it built in this hybrid form as a symbol of composite culture. Entry: Rs.10 (Indians).","tags":["Heritage","Nature","Architecture"]},
                "evening":   {"name":"Coorg Pork Curry & Coffee Estate Dinner","desc":"Coorg's Kodava community produce India's finest pork curry (pandi curry, with kachampuli — a black vinegar made from Garcinia fruit) and a distinctive rice bread (kadumbuttu). A farm-to-table dinner at a coffee estate homestay — with the estate's own arabica coffee served after dinner — is among India's most distinctive dining experiences. Cost: Rs.800–2,000.","tags":["Food","Coffee","Culture"]}},
            6: {"title": "Ooty & Nilgiri Hills",
                "morning":   {"name":"Nilgiri Mountain Railway — Blue Mountain Train","desc":"170km from Mysore, the Nilgiri Mountain Railway (UNESCO 2005) is a rack-and-pinion railway that climbs from Mettupalayam to Ooty (2,240m) — a 46km journey through 16 tunnels and across 31 bridges in 5 hours. Built by the British in 1908, it is still steam-hauled on the steepest lower section. Departs 7:10am from Mettupalayam. Train: Rs.30–60.","tags":["UNESCO","Railway","Heritage"]},
                "afternoon": {"name":"Botanical Garden & Toda Village","desc":"The Government Botanical Garden at Ooty (1847) contains 650 plant species across 22 hectares. The Toda tribal settlements on the outskirts of Ooty are home to one of India's most studied indigenous communities — famous for their barrel-shaped stone temples and extraordinary embroidery (pukhoor). Entry: Rs.30 (Indians).","tags":["Tribal","Nature","Heritage"]},
                "evening":   {"name":"Doddabetta Peak & Nilgiri Sunset","desc":"Doddabetta (2,637m) is the highest peak in the Nilgiri Hills and one of the highest in South India. The tea estates stretching to the horizon at sunset — emerald green rows against the blue mountains — are a landscape unlike anywhere else in India. Return to Mysore overnight. Entry: Rs.20 (Indians).","tags":["Nature","Tea","Sunset"]}},
            7: {"title": "Somnathpur & Mysore Silk Heritage",
                "morning":   {"name":"Somnathpur Kesava Temple","desc":"35km east of Mysore, the Kesava Temple at Somnathpur (1268 CE) is the most complete and best-preserved Hoysala temple in existence. Built on a 64-pointed star plan, its 3 vimanas and the exquisitely detailed outer walls represent the apex of Hoysala sculptural achievement. The temple is inside a walled courtyard with no later additions — it looks exactly as it did in 1268. Entry: Rs.25 (Indians).","tags":["UNESCO","Hoysala","Temple"]},
                "afternoon": {"name":"Mysore Silk Weaving & Sandalwood","desc":"Mysore is India's oldest silk weaving city — the Government Silk Weaving Factory (est. 1912) still produces the famous Mysore crepe silk, distinguished by its untwisted thread and pure zari borders. The adjacent Government Sandalwood Oil Factory distils the essential oil from aged Mysore sandalwood — used in perfumery worldwide. Entry: Rs.10 (Indians).","tags":["Craft","Silk","Heritage"]},
                "evening":   {"name":"Karanji Lake Bird Walk","desc":"Karanji Lake, within Mysore city, is a 90-acre wetland with 168 bird species and a giant butterfly enclosure. The evening is the best time for waterbirds — purple herons, painted storks, and open-billed storks nest here. The lakeside butterfly park houses 45 species. Entry: Rs.20 (Indians).","tags":["Birds","Nature","Butterfly"]}},
            8: {"title": "Mysore Palace Deep Dive",
                "morning":   {"name":"Mysore Palace — Restricted Inner Sections","desc":"Return to Mysore Palace for the inner sections not open during general visiting: the private Ambavilasa hall with its stained glass, the Gombhe Thotti (doll palace), and the Kalyani Mantapa marriage hall. The guided inner tour (Saturday–Thursday 9am) includes the golden throne room and the Wadiyar private chambers. Entry: Rs.70 + guide Rs.200.","tags":["Palace","Heritage","Royalty"]},
                "afternoon": {"name":"Jaganmohan Palace & Art Gallery","desc":"Built in 1861, the Jaganmohan Palace served as the main royal court while the main palace was under construction after 1897. The ground floor art gallery houses paintings by Raja Ravi Varma (the father of modern Indian art), 14th-century vijayanagara bronzes, and intricate ivory carvings. Entry: Rs.40 (Indians).","tags":["Palace","Art","Heritage"]},
                "evening":   {"name":"Mysore Dasara Heritage Walk","desc":"Mysore's Dasara festival (October) is one of India's grandest royal spectacles — the city is lit with 100,000 lights for 10 days and culminates in a grand procession of caparisoned elephants. Even outside the festival, walking the Dasara procession route at dusk — from the palace to Bannimantap — reveals the city's ceremonial layout. Cost: Free.","tags":["Festival","Heritage","Royalty"]}},
            9: {"title": "Bandipur & Nagarhole Forests",
                "morning":   {"name":"Bandipur Tiger Reserve — Dawn Safari","desc":"80km from Mysore, Bandipur National Park is part of the Nilgiri Biosphere — the largest protected area in South India. The park has one of India's highest densities of wild tigers, elephants, gaurs, and leopards. A 6am jeep safari through the dry deciduous forest has an excellent chance of elephant and gaur sightings. Entry: Rs.400 (Indians).","tags":["Tiger","Wildlife","Nature"]},
                "afternoon": {"name":"Nagarhole National Park","desc":"40km from Bandipur, Nagarhole (Rajiv Gandhi National Park) is one of India's best-managed tiger reserves. The Kabini reservoir within the park attracts huge concentrations of wildlife at the water's edge — especially from March to June when forests are dry. Boat safaris on the Kabini are one of India's finest wildlife experiences. Entry: Rs.400 (Indians).","tags":["Tiger","Wildlife","Nature"]},
                "evening":   {"name":"Kabini Riverside at Sunset","desc":"The Kabini riverbank at sunset — elephants wading across the river, egrets wheeling overhead, and the forest sounds building as darkness falls — is one of India's most beautiful wildlife evenings. The Kabini Jungle Lodges guesthouse on the bank is among India's finest wildlife stays. Cost: Rs.3,000–8,000.","tags":["Wildlife","Sunset","Nature"]}},
            10: {"title": "Farewell Mysore — Silk & Incense",
                "morning":   {"name":"Chamundi Hills Final Visit","desc":"Return to the Chamundi Hills before 7am for the dawn puja at the Chamundeshwari Temple — the rhythmic chanting, the smell of jasmine, and the golden sun rising over Mysore from the hill's 1,065m vantage point. Descend via the full 1,008 steps to see the Nandi Bull carved in 1659 at close range. Entry: Free.","tags":["Temple","Sunrise","Farewell"]},
                "afternoon": {"name":"Mysore Agarbatti & Handicrafts","desc":"Mysore is India's incense capital — the city produces 80% of India's agarbatti (incense sticks). The Cycle Brand and Hem agarbatti factories offer visits. Mysore is also famous for its rosewood inlay work, lacware, and the distinctive bronze Bidri work. The Cauvery Arts & Crafts Emporium on Sayaji Rao Road is the best fixed-price shop. Cost: Free to visit.","tags":["Craft","Incense","Shopping"]},
                "evening":   {"name":"Palace Illumination — Sunday Farewell","desc":"If your final evening is a Sunday, the Mysore Palace illumination (97,000 bulbs, 7–8pm) is the most spectacular farewell possible. The palace lit up at night against the dark sky — with the jasmine sellers and the crowd of families gathered on the maidan — is the quintessential Mysore experience. Entry: Rs.70 (Indians).","tags":["Palace","Illumination","Farewell"]}},
        }
    },
    "mumbai": {
        "name": "Mumbai, Maharashtra", "emoji": "MUM",
        "tagline": "The City of Dreams & Colonial Grandeur",
        "monuments": ["Gateway of India","Chhatrapati Shivaji Terminus","Elephanta Caves","Haji Ali Dargah","Crawford Market","Banganga Tank"],
        "budget": {"budget": "Rs.4,500", "mid": "Rs.14,000", "luxury": "Rs.40,000"},
        "days": {
            1: {"title": "Colonial Heritage & the Harbour",
                "morning":   {"name":"Gateway of India & Heritage Walk","desc":"The Gateway of India, built to commemorate the 1911 visit of King George V, was completed in 1924 in Indo-Saracenic style. It faces the Apollo Bunder harbour and was the ceremonial last departure point of British troops when they left India in 1948. Take a heritage walk through Colaba to see the Taj Mahal Palace Hotel (1903). Entry: Free.","tags":["Colonial","Architecture","Photography"]},
                "afternoon": {"name":"Chhatrapati Shivaji Terminus (CST)","desc":"Built in 1887 and named after Queen Victoria, this UNESCO World Heritage Site railway station is considered the finest example of Victorian Gothic architecture in India. Its design blends Gothic Revival with Mughal motifs — gargoyles share space with peacocks and tigers. Over 3 million passengers pass through daily. Entry: Free.","tags":["UNESCO","Victorian","Architecture"]},
                "evening":   {"name":"Marine Drive & Chowpatty Beach","desc":"The 3.6km Marine Drive, built in the 1920s, curves from Nariman Point to Chowpatty Beach in a perfect arc nicknamed the 'Queen's Necklace' for its appearance at night. Chowpatty is famous for pav bhaji, bhelpuri, and kulfi. The evening sea breeze and Art Deco buildings lining the boulevard are quintessential Mumbai.","tags":["Promenade","Food","Art Deco"]}},
            2: {"title": "Elephanta Caves & Island Heritage",
                "morning":   {"name":"Elephanta Caves","desc":"A 1-hour ferry from the Gateway of India, Elephanta Island (Gharapuri) houses the UNESCO-listed rock-cut cave temples dating to the 5th–8th century CE. The centrepiece is the Trimurti — a 6m three-faced bust of Lord Shiva representing the Creator, Preserver, and Destroyer — one of the greatest achievements of Indian sculpture. Entry: Rs.40 (Indians).","tags":["UNESCO","Shiva","Rock-Cut"]},
                "afternoon": {"name":"Banganga Tank & Walkeshwar Temple","desc":"Hidden in Malabar Hill, the Banganga Tank is a sacred stepped reservoir believed to have been created when Lord Rama shot an arrow into the ground and water sprang forth. The tank, surrounded by centuries-old temples, is one of Mumbai's oldest and most atmospheric heritage sites. Entry: Free.","tags":["Sacred","Temple","Hidden Gem"]},
                "evening":   {"name":"Haji Ali Dargah at Sunset","desc":"The Haji Ali Dargah, built in 1431, sits on a tidal islet 500m from the shore. It is only accessible at low tide via a narrow 500m causeway. The dazzling white and green Indo-Islamic shrine contains the tomb of Sayyed Peer Haji Ali Shah Bukhari. At sunset, with the Arabian Sea on both sides, it is one of Mumbai's great sights. Entry: Free.","tags":["Sufi","Sacred","Photography"]}},
            3: {"title": "Art Deco Mumbai & Fort District",
                "morning":   {"name":"Art Deco Tour — Marine Drive to Oval Maidan","desc":"Mumbai has the second largest collection of Art Deco buildings in the world (after Miami), concentrated along Marine Drive and around Oval Maidan. The Eros Cinema (1938), New India Assurance Building, and the Regal Cinema are the finest examples of Indian Art Deco — a distinctive tropical blend with Mughal and Hindu motifs. The Oval Maidan Heritage Walk covers 94 listed buildings. Entry: Free.","tags":["Art Deco","Architecture","Heritage"]},
                "afternoon": {"name":"Fort District Heritage Walk","desc":"Mumbai's Fort district — between CST and the Oval Maidan — contains Victorian Gothic buildings of extraordinary quality: the High Court (1878), the Rajabai Clock Tower (1878, modelled on Big Ben), Bombay University's Convocation Hall, and the secretariat buildings. The contrast between Victorian Gothic and Art Deco across the Oval is unique in the world. Entry: Free.","tags":["Colonial","Victorian","Architecture"]},
                "evening":   {"name":"Kala Ghoda Art District","desc":"Kala Ghoda is Mumbai's bohemian arts district centred on the Jehangir Art Gallery (1952) — India's most prestigious commercial gallery. The surrounding streets contain the Chhatrapati Shivaji Maharaj Vastu Sangrahalaya (Prince of Wales Museum, 1922), the Max Mueller Bhavan, and dozens of galleries and cafes. The annual Kala Ghoda Arts Festival (February) is India's largest arts festival. Entry: Free.","tags":["Art","Culture","Heritage"]}},
            4: {"title": "Dharavi & Living Mumbai",
                "morning":   {"name":"Dharavi Craft Heritage Walk","desc":"Dharavi — one of Asia's largest urban settlements — is home to 85,000 small businesses generating an estimated $650 million annually. The leather-tanning, recycling, pottery (kumbharwada), and embroidery industries have operated here for over 100 years. A responsible guided walking tour through the industrial quarters (not the residential areas) is eye-opening. Cost: Guide Rs.500–1,200.","tags":["Heritage","Craft","Culture"]},
                "afternoon": {"name":"Chor Bazaar — Thieves Market","desc":"Established in the 1870s in the Mutton Street area of Mumbai's Bhendi Bazaar, Chor Bazaar is the largest antique market in India. Genuine antiques — Art Deco clocks, Victorian mirrors, Parsi silver, Raj-era furniture, old Bollywood film posters — jostle with clever reproductions. The surrounding Bhendi Bazaar is one of Mumbai's most historically rich Muslim quarters. Entry: Free.","tags":["Antiques","Shopping","Heritage"]},
                "evening":   {"name":"Mohammad Ali Road & Iftar Culture","desc":"Muhammad Ali Road in South Mumbai is the gastronomic heart of Mumbai's Muslim community. During Ramzan, the street comes alive at dusk with hundreds of stalls serving malpua, sheer khurma, sevaiyan, and the extraordinary Mumbai-style nihari. Even outside Ramzan, the restaurants here serve the finest mughlai food in Mumbai. Cost: Rs.200–500.","tags":["Food","Muslim Heritage","Culture"]}},
            5: {"title": "Elephanta Deep Dive & Sanjay Gandhi NP",
                "morning":   {"name":"Sanjay Gandhi National Park — Kanheri Caves","desc":"45km from Mumbai, the Kanheri Caves inside Sanjay Gandhi National Park are 109 Buddhist rock-cut caves (1st century BCE to 11th century CE) — the largest cave complex in India. The Mahayanic caves contain colossal 7m Bodhisattva figures. The park itself is a remarkable 104 sq km forest within a megacity — home to leopards that routinely enter Mumbai's suburbs. Entry: Rs.50 (Indians).","tags":["Buddhist","Cave","Wildlife"]},
                "afternoon": {"name":"Basilica of Mount Mary, Bandra","desc":"The Basilica of Our Lady of the Mount in Bandra (rebuilt 1761) is Mumbai's most venerated Catholic shrine. The annual Bandra Fair (September) draws 500,000 devotees from across faiths. The surrounding Bandra West has the finest collection of Portuguese-era churches, bungalows, and convent buildings in Mumbai. Entry: Free.","tags":["Colonial","Christian","Heritage"]},
                "evening":   {"name":"Bandra-Worli Sea Link & Sunset","desc":"The Rajiv Gandhi Sea Link (2009) — a 5.6km cable-stayed bridge across Mahim Bay — is one of India's great engineering achievements. The Bandra Bandstand at sunset, looking at the sea link and the Mumbai skyline, is one of the city's signature views. The bandstand promenade is also the site of Shah Rukh Khan's Mannat mansion. Cost: Free.","tags":["Engineering","Photography","Sunset"]}},
            6: {"title": "Parsi Heritage — Towers of Silence",
                "morning":   {"name":"Parsi Heritage Walk — Cama Baug & Fire Temples","desc":"Mumbai has one of the world's largest concentrations of Parsis — descendants of Zoroastrian refugees from Persia (7th–10th century). The Agiary (fire temple) of Banaji Limji (1709) is the oldest in Mumbai and one of the few where non-Parsis can observe from outside. The Parsi Colony at Cama Baug contains 19th-century mansions and the first gymnasium in India. Entry: Exterior only.","tags":["Parsi","Zoroastrian","Heritage"]},
                "afternoon": {"name":"Chhatrapati Shivaji Maharaj Vastu Sangrahalaya","desc":"The Prince of Wales Museum (1922), now CSMVS, is Mumbai's premier museum housed in a spectacular Indo-Saracenic building. The collection spans 50,000 objects including Gupta-era stone sculpture, Mughal miniature paintings, Chinese porcelain acquired by the East India Company, and the complete Maratha armour of Shivaji's generals. Entry: Rs.85 (Indians).","tags":["Museum","Art","History"]},
                "evening":   {"name":"Crawford Market & Colaba Nightlife","desc":"Crawford Market (1869, designed by Lockwood Kipling — Rudyard's father) was the first building in India to use hydraulic power. The Norman Gothic structure's bas-reliefs show Indian agricultural and market life. The surrounding Colaba Causeway at night — street vendors, restaurants, and tourists — is quintessential Mumbai. Cost: Free to browse.","tags":["Colonial","Shopping","Food"]}},
            7: {"title": "Versova & Old Mumbai Fishing Villages",
                "morning":   {"name":"Versova Beach & Koli Heritage","desc":"The Koli fishing community are Mumbai's original inhabitants — 500 years before the Portuguese arrived, Koli fishermen lived in the seven islands that became Mumbai. The Versova Koliwada (fishing village) still operates traditional wooden boat building, net drying, and fish auctioning at dawn. The Versova fish market at 6am is one of Mumbai's great sensory experiences. Entry: Free.","tags":["Koli","Heritage","Culture"]},
                "afternoon": {"name":"Essel World & Marve Beach Heritage","desc":"The Marve-Manori beach strip on Mumbai's northwest coast retains Portuguese-era churches, coconut groves, and the last surviving traditional koli village architecture. The Portuguese built their first fort in Maharashtra at Vasai (Bassein, 40km north) in 1534 — its magnificent ruins cover 110 acres and are among the finest Portuguese heritage sites in Asia. Entry: Rs.25 (Indians).","tags":["Portuguese","Church","Heritage"]},
                "evening":   {"name":"Juhu Beach & Bollywood Heritage","desc":"Juhu Beach in suburban Mumbai has been the home of Bollywood's biggest stars since the 1950s — Amitabh Bachchan, Rajesh Khanna, and Hema Malini all lived here. The beach itself at sunset, with its bhelpuri vendors, pani puri carts, and families cooling off after a Mumbai day, is a deeply human Mumbai experience. Cost: Free.","tags":["Bollywood","Food","Culture"]}},
            8: {"title": "Lonar Crater & Aurangabad Day Trip",
                "morning":   {"name":"Lonar Crater Lake","desc":"450km from Mumbai (9 hours), the Lonar Lake is one of only four known hyper-velocity impact craters on earth — formed by a meteorite 50,000 years ago. The 1.8km circular lake is alkaline and saline simultaneously (unique in the world) and surrounded by medieval temples sinking slowly into the crater edge. Flamingos and migratory birds use it as a staging post. Entry: Rs.25 (Indians).","tags":["Geology","Nature","Heritage"]},
                "afternoon": {"name":"Lonar's 17 Temple Ring","desc":"The crater rim and nearby area contain 17 Hindu temples from the 7th–12th centuries — some half-submerged in the rising lake, others perfectly intact in the forest. The Daitya Sudan Temple (12th century) is the finest — a Hemadpanthi style temple with extraordinary black stone carvings of divine figures, celestial musicians, and Shiva legends. Entry: Free.","tags":["Temple","Geology","Hidden Gem"]},
                "evening":   {"name":"Return to Mumbai via Nashik","desc":"Return to Mumbai via Nashik — India's wine capital (150+ vineyards in the Sahyadri foothills) and one of the four Kumbh Mela sites. The Ramkund ghat on the Godavari river in Nashik is where Lord Rama, Sita, and Lakshmana stayed during their exile according to the Ramayana. Cost: Nashik wine tour Rs.500–1,500.","tags":["Heritage","Wine","Pilgrimage"]}},
            9: {"title": "Alibaug & Maratha Coastal Forts",
                "morning":   {"name":"Kulaba Fort, Alibaug — Sea Fort","desc":"100km south of Mumbai by road (or 1 hour by ferry from Gateway), Alibaug's Kulaba Fort (1680) was built by Chhatrapati Shivaji Maharaj's naval admiral Kanhoji Angre. The fort sits on a rocky island accessible on foot at low tide — when the sea draws back, a causeway of ancient stone appears. The fort walls are intact and the views back to the Konkan coast are magnificent. Entry: Rs.15 (Indians).","tags":["Maratha","Sea Fort","History"]},
                "afternoon": {"name":"Murud-Janjira — The Unconquered Fort","desc":"165km south of Mumbai, Janjira Fort on a circular island in the sea was the only coastal fort never captured by the Marathas, British, Portuguese, or Mughals. Built by the Siddis (African naval commanders) of the Adil Shahi dynasty in 1567, it has 22 bastions and its own freshwater tank. Reached by small boat from Murud. Entry: Rs.10 (Indians).","tags":["Sea Fort","Siddi","History"]},
                "evening":   {"name":"Konkan Coast Seafood — Malvani Cuisine","desc":"The Malvani coast between Alibaug and Ratnagiri produces the finest seafood cuisine in Maharashtra — surmai (kingfish) curry, bombil (Bombay duck) fry, tisrya (clam) masala, and the extraordinary sol kadhi (coconut milk and kokum digestif). A fresh catch dinner at a Malvani fishing village near Murud-Janjira is unforgettable. Cost: Rs.300–600.","tags":["Food","Seafood","Konkan"]}},
            10: {"title": "Farewell Mumbai — Sunrise to Sea",
                "morning":   {"name":"Sunrise at Worli Fort & Sea Face","desc":"The 17th-century Worli Fort (built by the British in 1675 to protect Mumbai harbour) is one of the city's hidden treasures — a small Portuguese-era battery now surrounded by the Worli sea face promenade. Sunrise here, with fishing boats heading out through the sea link and the Bandra skyline across the water, is quintessential Mumbai. Entry: Free.","tags":["Fort","Sunrise","Photography"]},
                "afternoon": {"name":"Sassoon Dock Fish Market","desc":"Sassoon Dock in Colaba is one of Mumbai's oldest fish markets — built in 1875 by the Jewish Sassoon family. Early morning is best (6–9am) but afternoon visits still catch the ice-packing, net-mending, and the extraordinary variety of fish species. The Koli women who dominate the market have traded here for generations. Entry: Free.","tags":["Market","Culture","Heritage"]},
                "evening":   {"name":"Sunset at Gateway of India — Farewell","desc":"End your Mumbai journey at the Gateway of India as the sun drops over Elephanta Island and the harbour lights up. The ferry traffic, the fishing boats, the tourist launches — the same view that British Viceroys saw on arrival and the last British troops saw on departure in 1948. Buy a vada pav at the nearby stall. The city never stops. Cost: Free.","tags":["Heritage","Sunset","Farewell"]}},
        }
    },
    "ajanta": {
        "name": "Ajanta & Ellora, Maharashtra", "emoji": "AJT",
        "tagline": "The Greatest Rock-Cut Art in the World",
        "monuments": ["Ajanta Caves","Ellora Caves","Kailasa Temple","Daulatabad Fort","Bibi Ka Maqbara","Aurangabad Caves"],
        "budget": {"budget": "Rs.2,500", "mid": "Rs.8,000", "luxury": "Rs.22,000"},
        "days": {
            1: {"title": "Ajanta — The Painted Caves",
                "morning":   {"name":"Ajanta Caves at Opening","desc":"The 30 rock-cut Buddhist caves at Ajanta (2nd century BCE to 6th century CE) are the finest surviving examples of ancient Indian wall painting in the world. The paintings — depicting the Jataka tales of the Buddha's previous lives — cover 5,000 sq metres of cave walls in natural pigments still vivid after 1,500 years. The caves were rediscovered by British officer John Smith while tiger-hunting in 1819. Entry: Rs.40 (Indians).","tags":["UNESCO","Buddhist","Paintings"]},
                "afternoon": {"name":"Ajanta Caves — The Vihara Monasteries","desc":"Spend the afternoon exploring the later Mahayana caves (Caves 1–2, 16–17), which contain the greatest painting programmes. Cave 1 holds the celebrated Bodhisattva Padmapani — a figure of such beauty it has become the symbol of Indian art internationally. The caves' position in a horseshoe-shaped gorge above the Waghur River adds to their dramatic setting.","tags":["UNESCO","Buddhist","Sculpture"]},
                "evening":   {"name":"Aurangabad City & Mughal Tombs","desc":"Aurangabad was the Mughal base for the Deccan Sultanate campaigns. The Bibi Ka Maqbara (1678), built by Mughal prince Azam Shah for his mother, was a deliberate homage to the Taj Mahal — earning it the nickname 'Taj of the Deccan.' The city also has 6 rock-cut Aurangabad Caves (7th century). Entry: Rs.25 (Indians).","tags":["Mughal","Architecture","History"]}},
            2: {"title": "Ellora — Temples of Three Faiths",
                "morning":   {"name":"Kailasa Temple, Ellora","desc":"Cave 16 at Ellora — the Kailasa Temple — is the world's largest monolithic rock-cut structure. Dedicated to Lord Shiva and modelled on Mount Kailash, the entire 60m x 33m complex was carved top-down from a single basalt rock cliff over 100 years (757–783 CE) under the Rashtrakuta dynasty. Over 200,000 tonnes of rock were removed. Entry: Rs.40 (Indians).","tags":["UNESCO","Hindu","Monolithic"]},
                "afternoon": {"name":"Ellora Buddhist & Jain Caves","desc":"The 34 caves at Ellora span 600 years of religious art from three traditions — Buddhist (Caves 1–12, 5th–7th c.), Hindu (Caves 13–29, 7th–9th c.), and Jain (Caves 30–34, 9th–11th c.). This extraordinary coexistence of faiths on a single hillside makes Ellora unique in world heritage. Entry included with morning ticket.","tags":["UNESCO","Buddhist","Jain"]},
                "evening":   {"name":"Daulatabad Fort","desc":"14km from Aurangabad, the hilltop Daulatabad Fort was considered the most impregnable fortress in medieval India. In 1327, Muhammad bin Tughluq forcibly relocated the entire population of Delhi here to make it his capital. The conical volcanic hill has a completely sealed dark tunnel passage that defenders could flood with smoke.","tags":["Fort","Medieval","History"]}},
            3: {"title": "Ajanta Deep Dive — Caves 1–9",
                "morning":   {"name":"Ajanta Early Buddhist Caves (9–12)","desc":"The earliest Hinayana Buddhist caves at Ajanta (Caves 9, 10, 12, 13) date to the 2nd century BCE — carved by merchants and monks on a trade route through the Deccan. Cave 9 and Cave 10 are pillared chaitya (prayer) halls with remarkable surviving paintings — Cave 10's 'Shaddanta Jataka' painting (2nd century BCE) is the oldest surviving Indian painting in any medium. Entry: Rs.40 (Indians).","tags":["Buddhist","Paintings","Ancient"]},
                "afternoon": {"name":"Ajanta Caves 1–5 — The Mahayana Golden Age","desc":"The Mahayana caves (1st–6th century CE) represent Ajanta's golden age. Cave 1's Bodhisattva Padmapani and Vajrapani are the twin masterpieces of Indian art. Cave 2 has extraordinary ceiling paintings. Cave 17 — the 'picture gallery' — has the most extensive surviving painting programme of any cave. A full afternoon with a knowledgeable guide is essential. Entry: Rs.40 (Indians).","tags":["Buddhist","Paintings","UNESCO"]},
                "evening":   {"name":"Ajanta Viewpoint & Waghur Valley","desc":"The classic viewpoint above the Waghur River gorge gives the full panorama of the 29 cave entrances carved into the horseshoe cliff face — the image that appears in every book on Indian art. In the evening light, with bats emerging from the caves and the gorge filling with shadow, it is one of India's most atmospheric sights. Entry: Free (from viewpoint).","tags":["Photography","Nature","Heritage"]}},
            4: {"title": "Aurangabad's Historic Layers",
                "morning":   {"name":"Bibi Ka Maqbara & Aurangabad Caves","desc":"The Bibi Ka Maqbara (1678) was commissioned by Prince Azam Shah for his mother Dilras Banu Begum. Its obvious homage to the Taj Mahal has made it famous, but it is a distinguished monument in its own right — the interior plasterwork and the lotus fountain tank are exquisite. The six Aurangabad Caves (7th century) on the city's north edge contain outstanding Tantric Buddhist sculptures. Entry: Rs.25 (Indians).","tags":["Mughal","Buddhist","Architecture"]},
                "afternoon": {"name":"Aurangabad's Panchakki & Sufi Heritage","desc":"The Dargah of Baba Shah Musafir at Panchakki has a unique water garden — the 17th-century hydraulic system uses a water wheel (panchakki) powered by an aqueduct to drive a grain mill and power the fountain garden. The expansive tank, shaded by a massive banyan tree, is one of the most peaceful heritage spots in Maharashtra. Entry: Rs.15 (Indians).","tags":["Sufi","Heritage","Engineering"]},
                "evening":   {"name":"Aurangabad Silk & Himroo Craft","desc":"Aurangabad produces the famous Himroo fabric — a hand-woven silk-cotton blend with rich geometric patterns developed during the Mughal period. The weave was originally created to imitate the much costlier Kincab (brocade) used in Mughal courts. An evening visit to the Himroo workshop cooperative is a living connection to Mughal textile culture. Cost: Free.","tags":["Craft","Textile","Mughal"]}},
            5: {"title": "Lonar Crater & Deccan Geology",
                "morning":   {"name":"Lonar Lake — Meteorite Crater","desc":"150km from Aurangabad, Lonar Lake was formed by a meteorite impact 50,000 years ago and is the world's only hypervelocity impact crater in volcanic basalt. The 1.8km diameter lake is simultaneously alkaline and saline. Archaeological evidence shows continuous settlement here since 10,000 BCE. Entry: Rs.25 (Indians).","tags":["Geology","Nature","Heritage"]},
                "afternoon": {"name":"Lonar's 17 Medieval Temples","desc":"The crater rim and its environs contain 17 medieval Hindu temples (7th–12th century CE). The Daitya Sudan Temple is the finest — carved in black basalt with extraordinary Hemadpanthi-style sculptural programme. Some temples are partially submerged as the lake level rises. The combination of geological wonder and medieval art is unique in India. Entry: Free.","tags":["Temple","Geology","Sculpture"]},
                "evening":   {"name":"Khuldabad — Where Aurangzeb Rests","desc":"25km from Aurangabad, Khuldabad (City of Eternity) is called the valley of saints — the dargahs of 1,500 Sufi saints are clustered here. The simple unmarked grave of Mughal Emperor Aurangzeb (died 1707) is here — the great emperor who conquered most of India asked to be buried in a plain unadorned grave funded only by the money he earned sewing prayer caps. Entry: Free.","tags":["Sufi","Mughal","Sacred"]}},
            6: {"title": "Pitalkhora & Deccan Buddhist Trail",
                "morning":   {"name":"Pitalkhora Caves","desc":"78km from Aurangabad, the Pitalkhora rock-cut caves (2nd–1st century BCE) are the oldest excavated rock-cut monuments in Maharashtra — predating Ajanta by 200 years. The caves contain extraordinary yaksha (nature spirit) and elephant sculptures, and a large Buddha figure. The jungle setting and near-total absence of other visitors make this one of the Deccan's most atmospheric heritage sites. Entry: Rs.25 (Indians).","tags":["Buddhist","Ancient","Hidden Gem"]},
                "afternoon": {"name":"Shivneri Fort — Shivaji's Birthplace","desc":"200km from Aurangabad near Junnar, the Shivneri hill fort is where Chhatrapati Shivaji Maharaj was born on 19 February 1630. The 17th-century fort also contains a spring called Ambabai's well, a Ganesh temple, and a small shrine marking Shivaji's birthplace. The Maratha hero is venerated by millions. Entry: Free.","tags":["Maratha","History","Fort"]},
                "evening":   {"name":"Junnar Lenyadri Buddhist Caves","desc":"Adjacent to Junnar, the Lenyadri caves (2nd century BCE) include the famous cave 7 — the only Ganapati (Ganesha) temple within a rock-cut Buddhist cave complex in India. The site is also one of the Ashtavinayak (Eight Ganesha) temples of Maharashtra — a sacred circuit visited by millions. Entry: Free.","tags":["Buddhist","Ganesha","Cave"]}},
            7: {"title": "Ellora Deep Dive — Hindu Caves",
                "morning":   {"name":"Ellora Hindu Caves — Ravana Ki Khai","desc":"Return to Ellora for the Hindu caves most visitors rush past. Cave 14 (Ravana Ki Khai) has panels of extraordinary quality including the famous 'Lakshminarayana' and the 'Shiva defeating Ravana' scenes. Cave 15 (Dashavatara) has a two-storey layout with a superb Nataraja and the 10 incarnations of Vishnu. Cave 29 (Dhumar Lena) is the largest Hindu cave at Ellora. Entry: Rs.40 (Indians).","tags":["UNESCO","Hindu","Sculpture"]},
                "afternoon": {"name":"Ellora Jain Caves — Indra Sabha","desc":"The Jain caves at Ellora (30–34, 9th–11th century) are the least visited but most delicate. Cave 32 (Indra Sabha) is the finest — a two-storey Digambara Jain temple with a life-size elephant standing outside and an upper-storey ceiling of incredible intricacy. Cave 33 (Jagannath Sabha) has an extraordinary assembly of Jain Tirthankaras. Entry: Rs.40 (Indians).","tags":["UNESCO","Jain","Sculpture"]},
                "evening":   {"name":"Ellora Sound & Light Show","desc":"The Ellora Sound and Light show (Tuesday–Sunday, 7pm) uses laser light and narration to illuminate the Kailasa Temple — the 60m monolith coming alive with the story of its 100-year carving and the Rashtrakuta dynasty that created it. Watching the Kailasa Temple lit against the night sky is extraordinary. Entry: Rs.25 (Indians).","tags":["Heritage","Culture","Night"]}},
            8: {"title": "Nasik — Kumbh Mela City",
                "morning":   {"name":"Nashik Ramkund Ghat & Trimbakeshwar","desc":"80km from Aurangabad, Nashik is one of the four Kumbh Mela sites. The Ramkund ghat on the Godavari river is the main bathing ghat — Lord Rama bathed here during his exile according to the Ramayana. The Trimbakeshwar Jyotirlinga Temple (1755), 28km from Nashik, is one of the 12 most sacred Shiva shrines in India. Entry: Free.","tags":["Pilgrimage","Kumbh","Sacred"]},
                "afternoon": {"name":"Nashik Wine Valley","desc":"The Sahyadri foothills around Nashik have been transformed into India's premier wine region since the 1990s — Sula Vineyards, York Winery, and Grover Zampa produce internationally recognised wines. A tasting tour through a Nashik vineyard reveals India's unexpected wine culture, built on the same volcanic basalt soils of the Deccan that surround Ajanta and Ellora. Cost: Rs.500–1,500.","tags":["Wine","Nature","Heritage"]},
                "evening":   {"name":"Nashik Peeth & Old City Walk","desc":"Nashik's old city contains the Kalaram Temple (1792) — a black stone Ram temple of extraordinary quality and the site of Gandhi's 1930 temple entry satyagraha for Dalit rights. The surrounding old city Peeth area is one of Maharashtra's most intact historic urban landscapes. Cost: Free.","tags":["Heritage","Gandhi","History"]}},
            9: {"title": "Ajanta's Full Conservation Story",
                "morning":   {"name":"Ajanta Conservation Centre Visit","desc":"The ASI Conservation Lab near the Ajanta caves is occasionally open to researchers and heritage visitors — it shows the conservation work on detached painting fragments, the analysis of original pigments (lapis lazuli from Afghanistan, malachite from Deccan, organic binders), and the 3D documentation of all 30 caves. Book in advance through ASI Aurangabad. Entry: Rs.25 (Indians).","tags":["Conservation","Buddhist","Heritage"]},
                "afternoon": {"name":"Cave 26 — The Reclining Buddha","desc":"Cave 26 at Ajanta is the final great chaitya hall — containing the 7m Reclining Buddha (parinirvana) with a processional path around it. The cave's carved facade with its double-storey colonnade is the finest at Ajanta. Spend the afternoon without rushing — note the extraordinary detail in the celestial musicians, the naga king with seven cobra hoods, and the Mara's temptation scene. Entry: Rs.40 (Indians).","tags":["Buddhist","Sculpture","Heritage"]},
                "evening":   {"name":"Village Pottery & Terracotta Craft, Aurangabad","desc":"The villages around Aurangabad produce the Bidriware craft — a zinc-copper alloy inlaid with silver, developed during the Bahmani Sultanate period (15th century). The dark oxidised metal with bright silver patterns is one of India's most distinctive craft traditions. A workshop visit and a Deccan dinner complete a rich day. Cost: Free (workshop Rs.200).","tags":["Craft","Bidriware","Heritage"]}},
            10: {"title": "Farewell Aurangabad — Final UNESCO Morning",
                "morning":   {"name":"Ajanta or Ellora — Personal Favourite Cave","desc":"Return for a final morning to the site that moved you most — the Bodhisattva Padmapani at Ajanta Cave 1, or the Kailasa Temple at Ellora Cave 16. Arrive at opening time (8am) before other tourists. Sit quietly for an hour with the ancient art and let the 1,500-year old images of compassion, devotion, or divine power settle into memory. Entry: Rs.40 (Indians).","tags":["UNESCO","Farewell","Heritage"]},
                "afternoon": {"name":"Aurangabad's City Museum","desc":"The Aurangabad Archaeological Museum (closed Fridays) houses a significant collection of Buddhist, Hindu, and Jain sculptures excavated from across the Marathwada region — Satavahana-era terracottas, Vakataka-period Buddhas, and medieval Yadava dynasty sculptures. Entry: Rs.5 (Indians).","tags":["Museum","Archaeology","History"]},
                "evening":   {"name":"Farewell at Aurangabad Heritage Garden","desc":"The Himayat Bagh garden in Aurangabad, laid out in the 17th century by Aurangzeb, is one of the few Mughal pleasure gardens in the Deccan. The garden contains the original Mughal water channels and pavilion foundations, and a diverse collection of fruit trees. An evening stroll here, with the call to prayer from the nearby mosque, is a gentle Mughal farewell. Entry: Free.","tags":["Gardens","Mughal","Farewell"]}},
        }
    },
    "khajuraho": {
        "name": "Khajuraho, Madhya Pradesh", "emoji": "KJR",
        "tagline": "The Temple City of Sacred Art",
        "monuments": ["Kandariya Mahadev Temple","Lakshmana Temple","Chaturbhuja Temple","Duladeo Temple","Archaeological Museum","Raneh Falls"],
        "budget": {"budget": "Rs.2,200", "mid": "Rs.6,500", "luxury": "Rs.18,000"},
        "days": {
            1: {"title": "The Western Temple Group",
                "morning":   {"name":"Kandariya Mahadev Temple at Sunrise","desc":"The largest and finest temple at Khajuraho, built by the Chandela dynasty c. 1025–1050 CE. The 30m shikhara (spire) soars above a 4m-tall platform. The temple has 872 sculptures — approximately 10% are erotic — which scholars link to Tantric Hindu philosophy, the depiction of worldly life before renunciation, and astrological symbolism. Entry: Rs.40 (Indians).","tags":["UNESCO","Chandela","Temple"]},
                "afternoon": {"name":"Western Group Archaeological Walk","desc":"The Western Group contains the densest concentration of Chandela temples (10th–11th century). Visit the Lakshmana Temple (954 CE) — the oldest surviving complete temple — and the Chausath Yogini Temple (9th century), the earliest at Khajuraho, built entirely in granite and dedicated to the 64 Yoginis. Entry: Rs.40 (Indians).","tags":["UNESCO","Sculpture","Archaeology"]},
                "evening":   {"name":"Sound & Light Show","desc":"The Khajuraho Sound and Light show at the Western Group amphitheatre brings the temples' mythological stories to life with dramatic narration and colour lighting against the sculpted facades. It is one of India's best heritage light shows, running in Hindi and English. Entry: Rs.250 (Indians).","tags":["Culture","Heritage","Art"]}},
            2: {"title": "Eastern Group & Jain Temples",
                "morning":   {"name":"Eastern Jain Temple Group","desc":"The Eastern Group contains superb Jain temples from the 10th–11th centuries. The Parsvanatha Temple is the largest of the Jain temples — despite being dedicated to the Jain Tirthankara Parsvanath, it contains many Hindu motifs, reflecting the syncretic culture of the Chandela kingdom. Entry: Rs.10 (Indians).","tags":["Jain","Chandela","Sculpture"]},
                "afternoon": {"name":"Chaturbhuja & Southern Temples","desc":"The Chaturbhuja Temple (c. 1100 CE) in the Southern Group houses a 2.7m monolithic four-armed Vishnu image of extraordinary quality — with no erotic sculpture, it represents the contemplative aspect of Chandela religious art. The nearby Duladeo Temple has the most densely covered sculpture programme of any Khajuraho temple. Entry: Free.","tags":["Temple","Vishnu","Sculpture"]},
                "evening":   {"name":"Raneh Falls & Ken River","desc":"25km from Khajuraho, the Raneh Falls plunge through a 5km gorge of crystalline granite in hues of pink, grey, red, and black. The Ken River here passes through the Ken Gharial Sanctuary, where the endangered gharial (fish-eating crocodilian) can sometimes be spotted from the canyon rim at sunset. Entry: Rs.25.","tags":["Nature","Waterfall","Wildlife"]}},
            3: {"title": "Ken Gharial Sanctuary & Local Heritage",
                "morning":   {"name":"Ken Crocodile Sanctuary — Dawn Boat Ride","desc":"The Ken Gharial Sanctuary, 30km from Khajuraho, is one of the few places in India where the critically endangered gharial — a freshwater crocodilian with a distinctive narrow snout — can be reliably spotted. Early morning boat rides on the Ken River (Rs.500–1,000) offer close sightings, along with mugger crocodiles, Indian skimmers, and river dolphins. Entry: Rs.100 (Indians).","tags":["Wildlife","Gharial","Nature"]},
                "afternoon": {"name":"Khajuraho Tribal Arts & Gond Painting","desc":"The Gond tribal community around Khajuraho produce remarkable pith paintings — large canvases covered in intricate dot-and-line patterns depicting forest spirits and mythological scenes. The Rajwaraha Adivasi Art Gallery in Khajuraho village is run directly by tribal artists. A guided visit connects the ancient erotic sculpture of the temples with the living folk art tradition. Cost: Free.","tags":["Tribal","Art","Gond"]},
                "evening":   {"name":"Archaeological Museum — Temple Fragments","desc":"The Khajuraho Archaeological Museum houses a collection of sculptures removed from damaged temple fragments — including the 10th-century Shaiva triad panel and several life-size erotic figures removed from collapsed structures. Understanding the sculptural canon (sastra) governing the sculptures makes the intact temples far more comprehensible. Entry: Rs.10 (Indians).","tags":["Museum","Sculpture","Chandela"]}},
            4: {"title": "Orchha — Bundela Capital",
                "morning":   {"name":"Orchha Palace Complex","desc":"175km from Khajuraho, Orchha was the capital of the Bundela Rajput kingdom. The Orchha Fort complex (16th–17th century) contains the Raja Mahal with extraordinary surviving fresco paintings, the Jahangir Mahal (built for Emperor Jahangir's visit), and 14 riverside cenotaphs. The entire complex is one of North India's most atmospheric heritage sites. Entry: Rs.25 (Indians).","tags":["Bundela","Fort","Heritage"]},
                "afternoon": {"name":"Ram Raja Temple & Phool Bagh","desc":"The Ram Raja Temple at Orchha is unique in India — Lord Rama is worshipped here with royal honours. The Phool Bagh (flower garden, 1606) is a formal Mughal-influenced garden with a cool underground retreat (Hamam palace) and an octagonal pavilion aligned to catch the monsoon breeze. Entry: Rs.25 (Indians).","tags":["Temple","Gardens","Heritage"]},
                "evening":   {"name":"Orchha Cenotaphs at Sunset","desc":"The 14 royal cenotaphs (chhatris) of Orchha's Bundela rulers line the Betwa riverbank. Each is a multi-storey pavilion topped with a dome, built in the distinct Bundela architectural style that blends Rajput and Mughal elements. Walking among them at sunset, with the Betwa running silver below and the fort silhouetted above, is one of Central India's great sights. Entry: Rs.25 (Indians).","tags":["Heritage","Sunset","Photography"]}},
            5: {"title": "Gwalior Day Trip",
                "morning":   {"name":"Gwalior Fort","desc":"320km from Khajuraho, the Gwalior Fort — called the 'pearl amongst fortresses in India' by Babur — rises 100m above the plains on a 3km sandstone plateau. The Man Singh Palace (1508) with its blue tile work, the 15th-century Jain cave figures carved into the cliff face, and the Sas-Bahu temples are the highlights. Entry: Rs.75 (Indians).","tags":["Fort","Medieval","Architecture"]},
                "afternoon": {"name":"Jai Vilas Palace & Scindia Museum","desc":"The 1874 Jai Vilas Palace has the world's most opulent dining room — two chandeliers of 3.5 tonnes each, tested by 10 elephants placed on the ceiling. The palace museum contains a silver model train that circulates brandy and cigars around a dining table. It remains the home of the Scindia royal family. Entry: Rs.200 (Indians).","tags":["Palace","Colonial","Museum"]},
                "evening":   {"name":"Tansen's Tomb & Gwalior Music Heritage","desc":"Gwalior's greatest legacy is Hindustani classical music — Tansen, the most celebrated musician of Akbar's court, was born here. His tomb and the annual Tansen Samaroh music festival have made Gwalior a pilgrimage for Indian classical musicians. The adjacent Ghaus Mohammed Mughal tomb has the finest blue tile work in Central India. Entry: Free.","tags":["Music","Mughal","Heritage"]}},
            6: {"title": "Panna Tiger Reserve",
                "morning":   {"name":"Panna Tiger Reserve — Morning Safari","desc":"30km from Khajuraho, Panna National Park was famous for losing all its tigers due to poaching by 2009 — and for the remarkable story of their reintroduction from Bandhavgarh, Pench, and Kanha. By 2021 there were 64 tigers. The park's Ken River corridors, gorges, and waterfalls make it one of India's most scenically dramatic tiger reserves. Safari: Rs.1,500–2,500.","tags":["Tiger","Wildlife","Nature"]},
                "afternoon": {"name":"Pandav Falls & Ken River Gorge","desc":"Within Panna National Park, the Pandav Falls drop 30m into a deep gorge formed by the Ken River cutting through the Vindhya limestone. The gorge walls reveal 400 million years of geological strata. The Pandava cave at the base of the falls is said to be where the Pandavas sheltered during their exile according to the Mahabharata. Entry: Rs.100 (Indians).","tags":["Nature","Geology","Heritage"]},
                "evening":   {"name":"Diamond Mining History — Panna","desc":"Panna district is India's only source of natural diamonds — the mines have been producing since ancient times. The Government Diamond Office offers guided tours of alluvial diamond washing in the Ken River (by appointment). The 23.5-carat 'Panna diamond' is among the famous gems mined here. Cost: Tour Rs.500.","tags":["Geology","History","Heritage"]}},
            7: {"title": "Chitrakoot — Sacred Forest Hermitage",
                "morning":   {"name":"Kamadgiri Sacred Hill & Ram Ghat","desc":"170km from Khajuraho, Chitrakoot is where Lord Rama, Sita, and Lakshmana spent 11.5 years of their 14-year exile — and where Bharata came to beg Rama to return. The Kamadgiri hill circumambulation (5km) passes 36 sacred sites in the Vindhya forest. Ram Ghat on the Mandakini river is one of India's most atmospheric pilgrimage ghats. Entry: Free.","tags":["Ramayana","Pilgrimage","Sacred"]},
                "afternoon": {"name":"Hanuman Dhara & Janki Kund","desc":"The Hanuman Dhara waterfall cascades down a cliff face onto a large Hanuman image — it never dries, believed to be Sita's gift to cool Hanuman after he set Lanka ablaze. Janki Kund is where Sita bathed daily during her exile. The Vindhya forest setting and the complete absence of tourist commercialism make this one of North India's most genuine pilgrimage experiences. Entry: Free.","tags":["Ramayana","Sacred","Nature"]},
                "evening":   {"name":"Mandakini Aarti & Village Stay","desc":"The evening aarti at Ram Ghat on the Mandakini river mirrors Varanasi's Ganga Aarti but in a completely different register — simple, unhurried, and watched by local pilgrims rather than tourists. An overnight stay in a dharamshala or village guesthouse is the authentic way to experience this ancient forest pilgrimage. Cost: Rs.200–500.","tags":["Ritual","Sacred","Rural"]}},
            8: {"title": "Khajuraho's Temple Cosmology Deep Dive",
                "morning":   {"name":"Vishvanatha & Nandi Temple — Solar Alignment","desc":"The Vishvanatha Temple (1002 CE) at Khajuraho is astronomically aligned — its central sanctum was designed to be illuminated by the rising sun on specific dates. The carved Nandi bull outside faces the temple's Shiva lingam across an open mandapa. The Vishvanatha's sculptural programme includes the most sophisticated narrative scenes at Khajuraho. Entry: Rs.40 (Indians).","tags":["Temple","Astronomy","Chandela"]},
                "afternoon": {"name":"Khajuraho Folk Art — Chandela Pottery","desc":"The villages surrounding Khajuraho continue the folk pottery tradition of the Chandela period. Terracotta votives, oil lamps, and decorative figures are made using the same hand-built and paddle-beaten techniques shown in the temple carvings. A village pottery workshop near Rajnagar completes the circuit between the living craft and the ancient art. Cost: Guide Rs.300–500.","tags":["Craft","Pottery","Heritage"]},
                "evening":   {"name":"Khajuraho Sound & Light Show — Second Visit","desc":"A second viewing of the Khajuraho Sound and Light show, with the knowledge gained over 8 days, is a completely different experience — the erotic figures, the celestial beings, and the divine couples all now understood in their cosmological context. The illuminated temples at night — the honey-coloured sandstone glowing against black sky — are Khajuraho at its most mysterious. Entry: Rs.250 (Indians).","tags":["Heritage","Culture","Night"]}},
            9: {"title": "Ajaigarh & Kalinjar Forts",
                "morning":   {"name":"Kalinjar Fort — Ancient Citadel","desc":"100km south of Khajuraho, the Kalinjar Fort was one of India's most strategically important hill forts — occupied continuously since 600 BCE. The fort withstood sieges by Mahmud of Ghazni, Humayun, Akbar (who used it as a regional capital), and Aurangzeb. The Chandela-era Shiva temple at the fort's summit and the massive Kirti Stambha (pillar of fame) are remarkable. Entry: Rs.25 (Indians).","tags":["Fort","Chandela","History"]},
                "afternoon": {"name":"Ajaigarh Fort & Vindhya View","desc":"60km from Khajuraho, Ajaigarh Fort (8th century CE) crowns a 700m plateau in the Vindhya hills with panoramic views across the Ken river valley to the Panna forests. The fort is rarely visited and the walk through the forest to the summit passes through the same landscape that the ancient trade routes between the Ganges plain and the Deccan followed for millennia. Entry: Free.","tags":["Fort","Vindhya","Heritage"]},
                "evening":   {"name":"Khajuraho by Night — Fort Road Walk","desc":"The Fort Road and main bazaar of Khajuraho at night reveal the living town behind the temples — the sweet shops, the chai stalls, the cycle-rickshaw drivers, and the local families. The temples are lit at night and visible above the rooftops. This is the easiest place in India to walk from a UNESCO World Heritage temple to a dhaba serving dal-baati-churma in under two minutes. Cost: Free.","tags":["Food","Night","Culture"]}},
            10: {"title": "Farewell Khajuraho — Temple at Dawn",
                "morning":   {"name":"Kandariya Mahadev at Sunrise","desc":"Return to Kandariya Mahadev at 6am — when the guards allow early access before the main crowds. The 30m shikhara in the first light, with dew on the red sandstone and no other tourists, reveals the full power of Chandela temple architecture. Spend an hour with the sculptures — identifying the 36 different apsara (celestial nymph) postures carved on the outer walls. Entry: Rs.40 (Indians).","tags":["UNESCO","Temple","Farewell"]},
                "afternoon": {"name":"Temple Museum & Final Lunch","desc":"Visit the Khajuraho Archaeological Museum for a final overview of the sculptural canon — understanding the 64 yoga positions, the 36 nayika (heroine) types, and the ashtanayika sequence that governs all the erotic sculpture. Then lunch at a local thali restaurant serving the Bundelkhand specialities: poha with jaggery, dal baati churma, and the local kadha (herbal tea). Entry: Rs.10 (Indians).","tags":["Museum","Food","Heritage"]},
                "evening":   {"name":"Final Sunset — Lakshmana Temple","desc":"The Lakshmana Temple (954 CE) is the oldest intact temple at Khajuraho and the one where Chandela artistry was still developing — a perfect comparison with the finished mastery of Kandariya Mahadev. Watch the sun go down over the temple from the garden, as local children play cricket in the shadow of a 1,000-year-old shrine. Entry: Rs.40 (Indians).","tags":["Temple","Sunset","Farewell"]}},
        }
    },
    "sanchi": {
        "name": "Sanchi, Madhya Pradesh", "emoji": "SAN",
        "tagline": "The Cradle of Buddhist Architecture",
        "monuments": ["Great Stupa","Stupa 2 & 3","Ashoka Pillar","Gupta Temples","Archaeological Museum","Udayagiri Caves"],
        "budget": {"budget": "Rs.1,500", "mid": "Rs.5,000", "luxury": "Rs.15,000"},
        "days": {
            1: {"title": "The Great Stupa & Ashokan Legacy",
                "morning":   {"name":"Great Stupa at Dawn","desc":"The Sanchi Stupa, begun by Emperor Ashoka in the 3rd century BCE and expanded over six centuries, is the oldest stone structure in India and the finest preserved Buddhist monument in the world. The four ornately carved toranas (gateways) — built in the 1st century BCE — tell the complete life story of the Buddha through narrative carvings. Entry: Rs.40 (Indians).","tags":["UNESCO","Buddhist","Mauryan"]},
                "afternoon": {"name":"Stupa 2, Stupa 3 & Ashoka Pillar","desc":"Stupa 3 at Sanchi contains the actual relic caskets of two of the Buddha's chief disciples, Sariputra and Mahamogallana, brought to London in 1853 and returned to India in 1953. The Ashokan Pillar (3rd century BCE) at Sanchi is among the finest surviving — its lion capital directly inspired the Lion Capital of Sarnath, now India's national emblem. Entry: Rs.40 (Indians).","tags":["Buddhist","Mauryan","Archaeology"]},
                "evening":   {"name":"Udayagiri Caves & Gupta Temples","desc":"Just 15km from Sanchi, Udayagiri contains 20 rock-cut caves dating to the Gupta dynasty (4th–5th century CE), including the celebrated 5m Varaha (Boar incarnation of Vishnu) relief — the finest surviving Gupta sculpture. The caves were commissioned by Chandragupta II. A nearby Gupta brick temple (5th century CE) is one of the earliest remaining. Entry: Rs.25.","tags":["Gupta","Hindu","Rock-Cut"]}},
            2: {"title": "Buddhist Trail & Vidisha",
                "morning":   {"name":"Vidisha & Heliodorus Pillar","desc":"The town of Vidisha, 10km from Sanchi, was the capital of a powerful Mauryan-era kingdom. It contains the Heliodorus Pillar (110 BCE) — erected by a Greek ambassador from the Indo-Greek kingdom who converted to Vaishnavism, making it the earliest known record of a foreign person converting to Hinduism. Entry: Free.","tags":["Mauryan","History","Heritage"]},
                "afternoon": {"name":"Sanchi Archaeological Museum","desc":"Opened in 1919, the Sanchi Museum houses the original lion capital from the Ashokan Pillar, carved ivory fragments from Stupa 2, and the extraordinary yakshi figure from Stupa 1's eastern torana — one of the most sensuous sculptures of the ancient world. A small but world-class collection. Entry: Rs.10 (Indians).","tags":["Museum","Sculpture","Buddhist"]},
                "evening":   {"name":"Raisen Fort & Sunset View","desc":"The massive Raisen Fort, 23km from Sanchi, was built in the 9th century and commands a dramatic hilltop position. It is famous as the last Rajput fort to fall to the Mughals (1543) — where Rajput women performed jauhar (mass self-immolation) to avoid capture. The sunset view over the Vindhya hills is extraordinary. Entry: Free.","tags":["Fort","History","Sunset"]}},
            3: {"title": "Bhimbetka Rock Shelters",
                "morning":   {"name":"Bhimbetka Prehistoric Rock Art","desc":"45km south of Sanchi, the Bhimbetka rock shelters (UNESCO 2003) contain the oldest known human habitation in India — occupied continuously from 100,000 BCE to the medieval period. The 700+ shelters have over 15,000 rock paintings showing hunting scenes, communal rituals, and animals from every prehistoric period up to the historical era. Entry: Rs.40 (Indians).","tags":["UNESCO","Prehistoric","Rock Art"]},
                "afternoon": {"name":"Bhimbetka Deep Exploration — Lower Shelters","desc":"The lower shelters at Bhimbetka, below the main tourist circuit, contain paintings of extraordinary quality rarely seen by visitors — large bison, boar, and rhinoceros from the Mesolithic period alongside later historical scenes of horse riders and warriors. A dedicated guide can access shelters not on the standard route. Entry: Rs.40 (Indians).","tags":["Prehistoric","Art","Heritage"]},
                "evening":   {"name":"Obaidullaganj & Ratapani Forest","desc":"The Ratapani Tiger Reserve (40km from Sanchi) is one of Madhya Pradesh's newer protected areas — home to leopards, sloth bears, and 55+ tigers. The forest at sunset is atmospheric. The Obaidullaganj dam reservoir attracts waterbirds at dusk. Entry: Rs.200 (Indians).","tags":["Wildlife","Nature","Tiger"]}},
            4: {"title": "Bhopal — City of Lakes",
                "morning":   {"name":"Taj-ul-Masajid — Largest Mosque in India","desc":"46km from Sanchi, Bhopal's Taj-ul-Masajid (begun 1878, completed 1985) is the largest mosque in India — its courtyard can accommodate 175,000 worshippers. The rose-pink sandstone and white marble structure with twin 18m minarets is Bhopal's greatest monument. The adjacent Moti Masjid (1860) is a miniature Jama Masjid. Entry: Free.","tags":["Islam","Architecture","Heritage"]},
                "afternoon": {"name":"State Museum & Bharat Bhavan","desc":"The Bhopal State Museum houses outstanding tribal art from Madhya Pradesh — Gond, Baiga, Bhil, and Korku communities. The Bharat Bhavan (1981), designed by Charles Correa, is India's finest multipurpose arts complex — integrating three museums, two amphitheatres, and galleries into the lakeside hillside. Entry: Rs.20–50 (Indians).","tags":["Museum","Tribal Art","Heritage"]},
                "evening":   {"name":"Upper Lake Sunset & Chowk Bazaar","desc":"The Upper Lake (Bada Talab) in Bhopal, created in the 11th century by the Paramara king Raja Bhoj, is the oldest man-made lake in Central India and still the city's main water supply. The sunset from the Boat Club jetty is one of Madhya Pradesh's finest views. Chowk Bazaar in the old city sells the famous Bhopal embroidery (zardozi) and street food. Cost: Free.","tags":["Lake","Heritage","Food"]}},
            5: {"title": "Pachmarhi — Vindhya Hill Station",
                "morning":   {"name":"Pachmarhi — Queen of Satpuras","desc":"195km from Sanchi, Pachmarhi at 1,067m in the Satpura Hills is Madhya Pradesh's only hill station and contains remarkable prehistoric rock paintings in Mahadeo Hills, ancient Buddhist shrines, and the Jata Shankar cave temple — a natural cave where Shiva is worshipped. The Satpura Tiger Reserve surrounds the town. Entry: Rs.25–100 (Indians).","tags":["Prehistoric","Nature","Heritage"]},
                "afternoon": {"name":"Bee Falls & Satpura Plateau Walk","desc":"The Bee Falls (35m) in the Pachmarhi valley are one of Madhya Pradesh's most beautiful waterfalls — accessible by a 2km walk through deciduous forest. The surrounding Satpura plateau has some of India's finest leopard and wild dog (dhole) habitat. A naturalist-guided walk at dusk is recommended. Entry: Rs.50 (Indians).","tags":["Waterfall","Nature","Wildlife"]},
                "evening":   {"name":"Sunset at Dhoopgarh — MP's Highest Point","desc":"Dhoopgarh (1,352m), the highest point in Madhya Pradesh, is 7km from Pachmarhi. The sunset panorama across the Satpura ranges and the Narmada valley below is one of Central India's greatest views. In winter, the Satpura plateau is cold enough for sweaters. Entry: Rs.20 (Indians).","tags":["Sunset","Views","Nature"]}},
            6: {"title": "Udayagiri & Gupta Art Heritage",
                "morning":   {"name":"Udayagiri Caves — Gupta Art at its Peak","desc":"Return to Udayagiri (15km from Sanchi) for a full morning exploring all 20 rock-cut shrines. The most important: Cave 5 (the great Varaha — Vishnu's boar incarnation lifting the Earth goddess — an 8m panel of extraordinary power), Cave 6 (Shiva Lingam shrine with Gupta-era panels), and Cave 7 (Chandragupta II's personal cave, with an inscription mentioning the Emperor by name). Entry: Rs.25 (Indians).","tags":["Gupta","Rock-Cut","Sculpture"]},
                "afternoon": {"name":"Gyaraspur & Maladevi Temple","desc":"35km from Sanchi, the village of Gyaraspur contains the remarkable Maladevi Temple (9th century) — built into a natural cave and combining the Pratihara and Gurjara-Pratihara architectural styles. The temple's sculptural programme includes some of the finest images of Vishnu and Shiva in Madhya Pradesh. Virtually no other tourists visit. Entry: Free.","tags":["Temple","Medieval","Hidden Gem"]},
                "evening":   {"name":"Eran — Ancient Coin City","desc":"90km from Sanchi, Eran was one of the most prolific coin-minting cities of the Gupta period and earlier Kushana dynasties. The Eran pillar (484 CE) is a Gupta pillar with the only known inscription mentioning sati (widow self-immolation) in ancient India. The site's massive Varaha figure (6th century) is the finest pre-medieval Vishnu image in India. Entry: Free.","tags":["Gupta","Archaeology","History"]}},
            7: {"title": "Bhopal Gas Tragedy Memorial & Modern Heritage",
                "morning":   {"name":"Bhopal Gas Tragedy Memorial Museum","desc":"The Union Carbide gas leak of 2–3 December 1984 killed at least 3,800 people immediately (estimates range to 16,000 total) in the world's worst industrial disaster. The Bhopal Memorial Hospital Research Centre and the Yaad-e-Hayat Memorial document the tragedy and its continuing health impact. Visiting is a deeply sobering and important act of witness. Entry: Free.","tags":["History","Memorial","Modern"]},
                "afternoon": {"name":"Indira Gandhi Rashtriya Manav Sangrahalaya","desc":"The National Museum of Mankind in Bhopal (1977) has the most extensive outdoor collection of vernacular Indian architecture in the world — over 30 full-scale traditional buildings reconstructed on a forested hillside from every region of India. The tribal habitats gallery is extraordinary. Entry: Rs.20 (Indians).","tags":["Museum","Tribal","Architecture"]},
                "evening":   {"name":"Bhopal Chatori Gali & Street Food","desc":"Chatori Gali (Tasty Lane) in old Bhopal is the finest street food precinct in Madhya Pradesh — try the bhopali paya (slow-cooked lamb trotter soup), the seviyan (vermicelli in saffron milk), and the imarti (a sweet deep-fried disc of urad lentil batter). The old Bhopal Begum's city is also full of extraordinary 19th-century haveli architecture. Cost: Rs.100–300.","tags":["Food","Culture","Heritage"]}},
            8: {"title": "Ujjain — City of the Kumbh & Jyotirlinga",
                "morning":   {"name":"Mahakaleshwar Jyotirlinga — Bhasma Aarti","desc":"175km from Sanchi, the Mahakaleshwar Temple at Ujjain houses one of the 12 Jyotirlingas — the abode of Shiva as Mahakal (Lord of Death and Time). The dawn Bhasma Aarti (5am, applied ash puja) is the most dramatic temple ritual in India — the lingam is anointed with fresh ash. Ujjain is one of the four Kumbh Mela sites. Entry: Free.","tags":["Jyotirlinga","Sacred","Kumbh"]},
                "afternoon": {"name":"Vedha Shala Observatory & Kaliadeh Palace","desc":"Ujjain was the prime meridian of ancient Indian astronomy — the zero-degree longitude for Hindu calculations. The Vedha Shala (Jantar Mantar equivalent, 1733) is a set of astronomical instruments built by Maharaja Jai Singh II. The Kaliadeh Palace (1458) on the Shipra river is a Malwa Sultanate pavilion of elegant simplicity. Entry: Rs.10–25 (Indians).","tags":["Astronomy","Heritage","History"]},
                "evening":   {"name":"Shipra River Aarti & Mahakaal Corridor","desc":"The newly completed Mahakal Corridor (2022) transforms the approach to the Mahakaleshwar Temple into a grand heritage boulevard 900m long with sculptures, fountains, and an evening lighting programme. The Shipra river aarti at sunset is a smaller, more intimate version of the Ganga Aarti. Cost: Free.","tags":["Sacred","Heritage","Modern"]}},
            9: {"title": "Satna & Son River Valley Heritage",
                "morning":   {"name":"Bharhut Stupa Sculptures — Indian Museum","desc":"The Bharhut Stupa (2nd century BCE), located near Satna (140km from Sanchi), was one of the earliest and most important Buddhist monuments in India. Though the stupa itself has vanished, its extraordinary carved railings — showing the earliest narrative art in India — are preserved in the Indian Museum, Kolkata. Visiting the site itself (near Satna) reveals the stupa foundations. Entry: Rs.10 (Indians).","tags":["Buddhist","Mauryan","Sculpture"]},
                "afternoon": {"name":"Chitrakoot & Mandakini Valley","desc":"125km from Sanchi, Chitrakoot is where Lord Rama spent 11.5 years of exile. The Kamadgiri hill circumambulation (5km) through dense forest passes 36 sacred sites. The Mandakini river, shaded by kadamba trees, is one of Madhya Pradesh's most beautiful streams — unchanged from the Ramayana's description. Entry: Free.","tags":["Ramayana","Pilgrimage","Nature"]},
                "evening":   {"name":"Son River Valley Sunset","desc":"The Son River valley between Satna and Rewa is one of Central India's most dramatic landscapes — a broad river valley flanked by the Vindhya escarpment. The river flows through a landscape of limestone cliffs, ancient trees, and migratory waterbirds. A sunset on the Son riverbank with the Vindhya hills behind is quintessential Madhya Pradesh. Cost: Free.","tags":["Nature","River","Sunset"]}},
            10: {"title": "Farewell Sanchi — The Great Stupa at Dawn",
                "morning":   {"name":"Great Stupa — Dawn Visit","desc":"Return to the Great Stupa at 8am for the final visit — the morning light coming over the Vindhya hills illuminating the eastern torana's narrative carvings. Spend a final hour identifying the Jataka stories: the Chhadanta Jataka (a white elephant), the Mahakapi Jataka (a monkey king), and the earliest representations of the Buddha as a footprint, a throne, or a parasol — never as a human figure. Entry: Rs.40 (Indians).","tags":["UNESCO","Buddhist","Farewell"]},
                "afternoon": {"name":"Stupa 3 — The Relic Caskets","desc":"Return to Stupa 3, where the relics of Sariputra and Moggallana — the Buddha's two chief disciples — rest in a stone casket inside the dome. The relics were removed by the British in 1851, taken to London, and returned to India in 1952 at the request of Jawaharlal Nehru. The caskets are now in a small chamber inside the stupa. Visiting with this context changes everything. Entry: Rs.40 (Indians).","tags":["Buddhist","History","Heritage"]},
                "evening":   {"name":"Village Farewell — Thali at Sanchi","desc":"The village of Sanchi has a handful of excellent local restaurants serving the Madhya Pradesh village thali — dal bafla (wheat dumplings in dal), kadhi, and the local black sesame chutney. A farewell dinner at sundown, with the silhouette of the Great Stupa's dome against the Vindhya sky, is one of heritage travel's most peaceful endings. Cost: Rs.100–300.","tags":["Food","Heritage","Farewell"]}},
        }
    },
    "bhubaneswar": {
        "name": "Bhubaneswar, Odisha", "emoji": "BBS",
        "tagline": "The Temple City of 1,000 Shiva Shrines",
        "monuments": ["Lingaraja Temple","Mukteshvara Temple","Konark Sun Temple","Udayagiri & Khandagiri Caves","Rajarani Temple","Odisha State Museum"],
        "budget": {"budget": "Rs.2,000", "mid": "Rs.6,000", "luxury": "Rs.17,000"},
        "days": {
            1: {"title": "The Temples of Bhubaneswar",
                "morning":   {"name":"Lingaraja Temple Complex","desc":"The largest and most important temple in Bhubaneswar, the Lingaraja was built c. 1090–1104 CE by Jajati Keshari. The 55m deula (tower) looms over the old town. The temple contains a swayambhu (self-originated) lingam believed to represent both Shiva and Vishnu — unusual in Indian temple tradition. Non-Hindus can view from a special platform. Entry: Free.","tags":["Temple","Shiva","Kalinga"]},
                "afternoon": {"name":"Mukteshvara & Rajarani Temples","desc":"The Mukteshvara Temple (950 CE) is called the 'gem of Odishan architecture' — a small, perfectly proportioned temple notable for its exquisite sculptural programme and a unique ornamental torana (arched gateway) — the only one surviving in Odisha. The adjacent Rajarani Temple (11th c.) has no resident deity but is famed for its erotic sculptures and figures of eight directional guardians. Entry: Rs.40 (Indians).","tags":["UNESCO","Kalinga","Sculpture"]},
                "evening":   {"name":"Odisha State Museum & Crafts Walk","desc":"The Odisha State Museum houses an outstanding collection of Buddhist and Jain manuscripts, tribal artefacts, natural history specimens, and medieval bronzes. Bhubaneswar is also the hub of Odisha's famous handicrafts — silver filigree (tarakasi), applique work (chandua), and pattachitra painting. Visit a government crafts village. Entry: Rs.10 (Indians).","tags":["Museum","Craft","Culture"]}},
            2: {"title": "Konark Sun Temple & Coastal Heritage",
                "morning":   {"name":"Konark Sun Temple","desc":"Built by King Narasimhadeva I in 1250 CE, the Konark Sun Temple is conceived as a gigantic stone chariot for the Sun god Surya — with 24 elaborately carved stone wheels (each 3m in diameter) and 7 stone horses. The 70m tower collapsed centuries ago but the jagamohana (audience hall) survives. The erotic sculptures rival Khajuraho. Entry: Rs.40 (Indians).","tags":["UNESCO","Sun Temple","Kalinga"]},
                "afternoon": {"name":"Udayagiri & Khandagiri Caves","desc":"These twin hills, 6km from Bhubaneswar, contain 33 rock-cut caves dug in the 1st century BCE as residences for Jain monks under King Kharavela of Kalinga. The Hathigumpha (Elephant Cave) inscription on Khandagiri is one of India's most important historical inscriptions — a 17-line eulogy of King Kharavela in Brahmi script. Entry: Rs.25 (Indians).","tags":["Jain","Mauryan","Rock-Cut"]},
                "evening":   {"name":"Puri Jagannath Temple Evening","desc":"The Jagannath Temple at Puri (65km from Bhubaneswar) is one of the Char Dham pilgrimage sites of Hinduism. The 65m tower was built by King Anantavarman Chodaganga Deva in the 12th century. The temple kitchen (the Ananda Bazaar) is said to be the world's largest — feeding 10,000 people daily with the mahaprasad. Non-Hindus may not enter but can view from the Raghunandan Library rooftop. Entry: Free.","tags":["Pilgrimage","Hindu","Cuisine"]}},
            3: {"title": "Puri — Char Dham Pilgrimage City",
                "morning":   {"name":"Puri Beach & Swargadwar Ghat","desc":"Puri is one of India's great pilgrimage cities and its beach — 35km of golden sand — is one of the finest on the east coast. The Swargadwar (Gate of Heaven) at the southern end of the beach is where Hindus cremate the dead. The beach at dawn — with pilgrims bathing in the surf, fishermen pulling in nets, and sand artists at work — is one of Odisha's most vivid scenes. Entry: Free.","tags":["Pilgrimage","Beach","Culture"]},
                "afternoon": {"name":"Gundicha Temple & Narendra Tank","desc":"The Gundicha Temple is the 'Garden House of God' — during the annual Rath Yatra (Chariot Festival), Lord Jagannath's idol is pulled here from the main temple and stays for 9 days before returning. The 5km route of the Rath Yatra (the world's largest chariot procession, drawing 1 million devotees) passes through the main road. The Narendra tank hosts the annual Chandan Yatra boat festival. Entry: Free.","tags":["Temple","Festival","Heritage"]},
                "evening":   {"name":"Raghurajpur Heritage Village — Pattachitra","desc":"14km from Puri, Raghurajpur is Odisha's most famous heritage craft village — every family in the village is a hereditary artist. The pattachitra (cloth or palm-leaf painting) tradition dates to the 12th century and its subject matter is inseparable from the Jagannath cult. The village's decorated walls and the sound of artists working in every home are extraordinary. Entry: Free.","tags":["Craft","Art","Heritage"]}},
            4: {"title": "Chilika Lake — Asia's Largest Lagoon",
                "morning":   {"name":"Chilika Lake — Bird Watching","desc":"100km from Bhubaneswar, Chilika is Asia's largest coastal lagoon (1,100 sq km) — a Ramsar-designated wetland visited by one million migratory birds every winter. The open water and reed beds host flamingos, bar-headed geese, spoonbills, and the globally endangered Irrawaddy dolphin. A boat from Mangalajodi village (November–February) is the best experience. Entry: Rs.50 + boat.","tags":["Birds","Wildlife","Ramsar"]},
                "afternoon": {"name":"Nalabana Island Bird Sanctuary","desc":"The Nalabana Island in Chilika Lake's heart is a seasonal bird sanctuary — accessible by boat when the island emerges from the water (October–April). Up to 500,000 birds can be present simultaneously. The island is surrounded by the largest concentration of flamingos in eastern India. Entry: Rs.100 (Indians, permit required).","tags":["Birds","Nature","Wildlife"]},
                "evening":   {"name":"Kalijai Island Temple","desc":"The Kalijai Island in Chilika Lake has a temple to the goddess Kalijai — said to be a young girl who died here before her wedding. Devotees float terracotta horses, candles, and flowers on the lake. The boat journey across the lagoon at sunset, with the pink sky reflected in the shallow water and birds roosting in every tree, is unforgettable. Cost: Boat Rs.200–400.","tags":["Sacred","Nature","Photography"]}},
            5: {"title": "Simlipal Tiger Reserve",
                "morning":   {"name":"Simlipal National Park — Waterfall Safari","desc":"250km from Bhubaneswar, Simlipal is Odisha's largest national park (2,750 sq km) and one of India's finest biosphere reserves. The park has 12 waterfalls including the spectacular Joranda Falls (150m) and Barehipani Falls (400m — among India's tallest). The park is home to 100+ tigers, 400 elephants, and the rare melanistic (black) tigers unique to Simlipal. Safari: Rs.1,500–2,500.","tags":["Tiger","Wildlife","Waterfall"]},
                "afternoon": {"name":"Joranda & Barehipani Falls Trek","desc":"The Barehipani Falls (400m) are reached by a 7km forest trek from the road. The falls cascade in two stages down a sandstone escarpment into a deep gorge — the largest waterfall in Odisha. The forest trail passes through sal, bamboo, and mahua groves full of birds and butterflies. Trek time: 3–4 hours return. Entry: Included with park ticket.","tags":["Waterfall","Trekking","Nature"]},
                "evening":   {"name":"Tribal Village Stay — Baiga Community","desc":"The Baiga tribal community within the Simlipal buffer zone are one of Odisha's most traditional forest communities. An overnight stay in a Baiga ecotourism guesthouse includes a traditional meal (wild greens, mahua flower dal, bamboo rice) and a performance of the Chhau martial dance tradition. Cost: Rs.800–1,500.","tags":["Tribal","Culture","Chhau"]}},
            6: {"title": "Dhauli & Buddhist Odisha",
                "morning":   {"name":"Dhauli Peace Pagoda & Rock Edict","desc":"8km from Bhubaneswar, Dhauli hill is where Emperor Ashoka's decisive battle of Kalinga (261 BCE) took place — and where, horrified by the carnage, he converted to Buddhism. The Ashokan Rock Edict here predates Sanchi's stupa and is the first known proclamation of dharma (moral law) as state policy. The modern Shanti Stupa (1972) overlooks the Daya river where 100,000 died in the battle. Entry: Free.","tags":["Buddhist","Mauryan","Peace"]},
                "afternoon": {"name":"Ratnagiri, Udayagiri & Lalitgiri — Diamond Triangle","desc":"90km from Bhubaneswar, the Buddhist 'Diamond Triangle' contains three major excavated sites. Ratnagiri (7th–11th century CE) is the most impressive — enormous Bodhisattva figures and a complete monastery complex. The Ratnagiri museum displays hundreds of Buddhist sculptures including the extraordinary Vajrayana Tantric figures unique to this region. Entry: Rs.25 (Indians).","tags":["Buddhist","UNESCO","Museum"]},
                "evening":   {"name":"Puri Sunset Beach Return","desc":"Return to Puri for the sunset beach — the famous Puri red-golden sunset, when the Bay of Bengal turns crimson and the fishermen's silhouettes line the shoreline. Buy coconut water from the vendors and sit on the sand as families, pilgrims, and sadhus watch the same sunset together. Cost: Free.","tags":["Beach","Sunset","Heritage"]}},
            7: {"title": "Odisha Tribal Heritage",
                "morning":   {"name":"Odisha Crafts Village & Tribal Museum","desc":"The Tribal Research Institute Museum in Bhubaneswar has the finest collection of Odisha tribal art and material culture — Dongria Kondh headdresses, Gond paintings, Saura ikons (traditional votive wall paintings), and Juang textiles. The adjacent IDCOL Crafts Village has artisans making applique work (Pipli chandua), silver filigree (Cuttack tarakasi), and stone carving in workshop settings. Entry: Rs.10–20 (Indians).","tags":["Tribal","Museum","Craft"]},
                "afternoon": {"name":"Koraput & Dhimsa Dance Villages","desc":"300km south of Bhubaneswar, Koraput district is one of India's most biodiverse and culturally rich tribal regions — home to the Kondh, Gadaba, and Bondas communities. The weekly Koraput tribal market draws dozens of communities to trade forest produce, textiles, and livestock. The Dhimsa circle dance of the Kondh is one of Odisha's most distinctive performing arts. Cost: Tour Rs.1,500–3,000.","tags":["Tribal","Market","Culture"]},
                "evening":   {"name":"Chhau Dance Performance","desc":"The Chhau martial dance — performed in elaborate masks representing gods, demons, and animals — is Odisha's most spectacular performing art and was inscribed on UNESCO's Intangible Cultural Heritage list in 2010. The Seraikela and Mayurbhanj styles both originate in Odisha. An evening performance at a cultural centre in Bhubaneswar or Puri is extraordinary. Cost: Rs.200–500.","tags":["Dance","UNESCO","Culture"]}},
            8: {"title": "Konark Deep Dive & Coastal Walk",
                "morning":   {"name":"Konark Sun Temple — Archaeological Deep Dive","desc":"Return to Konark for a full morning with a specialist guide. The temple's 24 wheels are not merely decorative — they are sundials marking the hours. The 7 horses represent the days of the week. The erotic panels on the outer walls represent the worldly life from which one enters the divine inner sanctum. The temple's astronomical alignments track solstices and equinoxes. Entry: Rs.40 (Indians).","tags":["UNESCO","Astronomy","Architecture"]},
                "afternoon": {"name":"Konark Beach & Chandrabhaga Festival","desc":"Konark beach, 3km from the temple, is one of Odisha's finest — the sand here is used for sand sculpture by Padma Shri artist Sudarsan Pattnaik, who has won world championships. The Chandrabhaga Festival (February full moon) sees thousands of pilgrims bathe at the beach adjacent to where the Chandrabhaga river met the sea in ancient times. Entry: Free.","tags":["Beach","Art","Festival"]},
                "evening":   {"name":"Curls of the Mahanadi — Boat Ride","desc":"The Mahanadi delta east of Bhubaneswar is a labyrinth of channels, mangroves, and sandbanks. An evening boat ride on the Mahanadi near Cuttack reveals the river system that has sustained Odisha's civilisation for 3,000 years — the same waterway that carried the Buddhist missionaries of Ashoka to Southeast Asia. Cost: Rs.300–500.","tags":["River","Nature","Heritage"]}},
            9: {"title": "Cuttack — Silver City of Odisha",
                "morning":   {"name":"Cuttack Tarakasi Silver Filigree","desc":"25km from Bhubaneswar, Cuttack has been India's silver filigree capital since the 12th century. The tarakasi craft — creating objects from twisted silver wire as fine as human hair — requires years of training. The silver jewellery, miniature temples, and decorative objects produced here are known worldwide. Visit the silver artisan quarter in the old city. Cost: Free.","tags":["Craft","Silver","Heritage"]},
                "afternoon": {"name":"Cuttack Barabati Fort & Durgapuja Heritage","desc":"The Barabati Fort (14th century) in Cuttack was the seat of the Ganga dynasty and later the Mughals and the Marathas. Its moat and ruined gateway are the only remnants. Cuttack is also the most important city for Odisha's Durga Puja — the city's theme-based puja mandaps draw millions of visitors annually (October). Entry: Rs.5 (Indians).","tags":["Fort","Heritage","Festival"]},
                "evening":   {"name":"Kathajodi River & Cuttack Floating Market","desc":"The floating market on the Kathajodi river in Cuttack — vegetables and fish sold from boats at the Maa Mahanadi Ghat — has operated for centuries. An evening visit reveals the living river-based economy of Odisha's former capital. The sunset over the Mahanadi from the Netaji Bridge is atmospheric. Cost: Free.","tags":["Market","River","Heritage"]}},
            10: {"title": "Farewell Bhubaneswar — Temple at Dawn",
                "morning":   {"name":"Lingaraja Temple — Dawn Puja","desc":"Return to the Lingaraja Temple complex for the dawn puja (5am, for Hindus) — the priests bathe the Shiva lingam in milk, honey, and rose water before the sun rises. Non-Hindus can observe from the platform outside the main entrance. The temple's 55m tower lit at dawn against the eastern sky, with the priests chanting and the scent of incense filling the air, is a deeply moving farewell. Entry: Free.","tags":["Temple","Sacred","Farewell"]},
                "afternoon": {"name":"Museum of Tribal Arts & Final Crafts","desc":"A final visit to the IDCOL Crafts Village to purchase Odisha's finest: a pattachitra scroll (pilgrimage to Jagannath depicted in successive frames), a silver tarakasi pendant, and a piece of Sambalpuri ikat weaving — the double-resist dyeing process recognised as a GI product. Odisha crafts are among India's finest and most undervalued. Cost: Rs.500–5,000.","tags":["Craft","Shopping","Heritage"]},
                "evening":   {"name":"Farewell Dinner — Odisha Cuisine","desc":"Odisha has one of India's most distinctive and least known cuisines — dalma (toor dal with vegetables and coconut), pakhala (fermented rice water, eaten cold, a summer staple for 2,000 years), chhena poda (charred cheese cake, the only Indian sweet cooked in an oven), and the Konark prawn curry. A farewell dinner celebrating Odisha's flavours ends one of India's most rewarding heritage journeys. Cost: Rs.200–500.","tags":["Food","Culture","Farewell"]}},
        }
    },
}

DESTINATIONS_LIST = [
    ("-- Select a city --", ""),
    ("Agra, Uttar Pradesh", "agra"),
    ("Delhi", "delhi"),
    ("Varanasi, Uttar Pradesh", "varanasi"),
    ("Lucknow, Uttar Pradesh", "lucknow"),
    ("Amritsar, Punjab", "amritsar"),
    ("Jaipur, Rajasthan", "jaipur"),
    ("Jodhpur, Rajasthan", "jodhpur"),
    ("Udaipur, Rajasthan", "udaipur"),
    ("Jaisalmer, Rajasthan", "jaisalmer"),
    ("Hampi, Karnataka", "hampi"),
    ("Thanjavur, Tamil Nadu", "thanjavur"),
    ("Madurai, Tamil Nadu", "madurai"),
    ("Mysore, Karnataka", "mysore"),
    ("Mumbai, Maharashtra", "mumbai"),
    ("Ajanta & Ellora, Maharashtra", "ajanta"),
    ("Khajuraho, Madhya Pradesh", "khajuraho"),
    ("Sanchi, Madhya Pradesh", "sanchi"),
    ("Bhubaneswar, Odisha", "bhubaneswar"),
]

LOADING_STEPS = [
    "Running KNN interest matching...",
    "Scoring cities by cosine similarity...",
    "Extracting NLP intent from preferences...",
    "Predicting budget via regression model...",
    "Ranking matched heritage sites...",
    "Selecting personalised day activities...",
    "Curating ML-filtered highlights...",
    "Finalising your AI itinerary...",
]

BUDGET_COST_DEFAULT = {"budget": "Rs.3,000",  "mid": "Rs.9,000",  "luxury": "Rs.25,000"}

def get_budget_cost(dest_key, budget_key):
    city = DESTINATION_DATA.get(dest_key, {})
    return city.get("budget", BUDGET_COST_DEFAULT).get(budget_key, BUDGET_COST_DEFAULT[budget_key])

INTERESTS = ["History","Architecture","Religion","Nature","Food",
             "Photography","Culture","Adventure","Shopping","Nightlife"]
STYLES    = [("Walk","Relaxed"), ("Fast","Packed"), ("Focus","Focused"), ("Free","Flexible")]
DURATIONS = [2, 3, 5, 7, 10]


# ── HELPERS ──────────────────────────────────────────────────────────────────
def default_days(city_name):
    return {
        1: {"title": "Arrival & First Monuments",
            "morning":   {"name":f"{city_name} Heritage Walk","desc":"Start your journey with a guided heritage walk through the old city. Take in the architectural layers of history on every street corner.","tags":["Heritage","Walking","Architecture"]},
            "afternoon": {"name":"Main Monument Visit","desc":"Explore the primary UNESCO site or fort of the city. Hire a local guide for deeper historical context and hidden corners most visitors miss.","tags":["UNESCO","History","Culture"]},
            "evening":   {"name":"Local Bazaar & Cuisine","desc":"Immerse yourself in the local market culture. Try regional street food and explore artisanal crafts unique to this part of India.","tags":["Food","Shopping","Culture"]}},
        2: {"title": "Deep Dive & Hidden Gems",
            "morning":   {"name":"Sunrise at Sacred Site","desc":"Begin early at the most sacred monument — the quality of morning light and the near-empty grounds create an unforgettable experience.","tags":["Sacred","Sunrise","Photography"]},
            "afternoon": {"name":"Museum & Archaeological Site","desc":"Visit the local archaeological museum to understand the artifacts in context, then explore a lesser-known ruin or temple nearby.","tags":["Museum","Archaeology","Learning"]},
            "evening":   {"name":"Cultural Performance","desc":"Attend an evening of classical dance or music — a living connection to the traditions that built these monuments.","tags":["Culture","Music","Dance"]}},
    }


# ════════════════════════════════════════════════════════════════════════════
#  PDF GENERATOR
# ════════════════════════════════════════════════════════════════════════════
def generate_pdf(filepath, city_name, tagline, total_days, budget_key,
                 style_name, days_data, n_monuments, start_date, budget_display=None):
    doc = SimpleDocTemplate(
        filepath, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title=f"{city_name} Heritage Itinerary",
        author="Walk Through History"
    )
    W = A4[0] - 36*mm

    def ps(name, size, color=PDF_TEXT, bold=False, align=TA_LEFT, leading=None):
        return ParagraphStyle(
            name, fontSize=size, textColor=color,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            alignment=align, leading=leading or size*1.4, spaceAfter=0)

    story = []

    # Brand / title
    story.append(Paragraph("Walk Through History",
                            ps("brand", 9, PDF_GOLD, bold=True, align=TA_CENTER)))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width=W, thickness=0.5, color=PDF_GOLD, spaceAfter=4*mm))
    story.append(Paragraph(f"{city_name} Heritage Journey",
                            ps("title", 22, PDF_GOLD, bold=True, align=TA_CENTER)))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(tagline, ps("sub", 10, PDF_TEXT2, align=TA_CENTER)))
    story.append(Spacer(1, 5*mm))

    # Badges
    badges = [f"{total_days} Days", f"{budget_key.capitalize()} Budget", f"{style_name} Pace"]
    bd = [[Paragraph(b, ps(f"b{i}", 8, PDF_DARK if i==0 else PDF_GOLD,
                           bold=True, align=TA_CENTER)) for i, b in enumerate(badges)]]
    bt = Table(bd, colWidths=[W/3]*3)
    bt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(0,0), PDF_GOLD),
        ("BACKGROUND",    (1,0),(2,0), PDF_CARD),
        ("ALIGN",         (0,0),(-1,-1),"CENTER"),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(bt)
    story.append(Spacer(1, 4*mm))

    # Stats
    cost = budget_display or BUDGET_COST_DEFAULT.get(budget_key, "Rs.9,000")
    stats = [(str(total_days),"DAYS"),(str(total_days*3),"EXPERIENCES"),
             (str(n_monuments),"MONUMENTS"),(cost,"EST./DAY")]
    sd = [[Paragraph(n, ps(f"sn{si}",14,PDF_GOLD,bold=True,align=TA_CENTER)) for si,(n,_) in enumerate(stats)],
          [Paragraph(l, ps(f"sl{si}", 7,PDF_TEXT2,align=TA_CENTER)) for si,(_,l) in enumerate(stats)]]
    st = Table(sd, colWidths=[W/4]*4)
    st.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), PDF_CARD),
        ("ALIGN",         (0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LINEBELOW",     (0,0),(-1,0), 0.3, PDF_GOLD),
    ]))
    story.append(st)
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width=W, thickness=0.4, color=PDF_GOLD, spaceAfter=6*mm))

    # Day cards
    day_keys = list(days_data.keys())
    slots_cfg = [
        ("Morning",   "morning",   "7 AM",  PDF_MORN),
        ("Afternoon", "afternoon", "1 PM",  PDF_GOLD),
        ("Evening",   "evening",   "6 PM",  PDF_EVE),
    ]
    for i in range(total_days):
        dk       = day_keys[i % len(day_keys)]
        di       = days_data[dk]
        date_str = (start_date + timedelta(days=i)).strftime("%A, %d %b %Y")

        dh = Table([[
            Paragraph(f"D{i+1}", ps(f"dc{i}", 9, PDF_DARK, bold=True, align=TA_CENTER)),
            Paragraph(f"Day {i+1}  -  {di['title']}", ps(f"dt{i}", 12, PDF_GOLD, bold=True)),
            Paragraph(date_str, ps(f"dd{i}", 8, PDF_TEXT2)),
        ]], colWidths=[12*mm, W*0.6, None])
        dh.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(0,0), PDF_GOLD),
            ("BACKGROUND",    (1,0),(-1,0), PDF_CARD),
            ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (1,0),(-1,-1), 8),
        ]))
        story.append(KeepTogether([dh]))
        story.append(Spacer(1, 1*mm))

        for pname, skey, tstr, col in slots_cfg:
            slot = di[skey]
            tags_str = "  |  ".join(slot["tags"])
            inner = Table([
                [Paragraph(f"{pname.upper()}  -  {tstr}", ps(f"per{i}{skey}", 7, PDF_TEXT2, bold=True))],
                [Paragraph(slot["name"],  ps(f"an{i}{skey}", 10, PDF_TEXT, bold=True))],
                [Paragraph(slot["desc"],  ps(f"ad{i}{skey}",  8, PDF_TEXT2, leading=11))],
                [Paragraph(tags_str,      ps(f"at{i}{skey}",  7, PDF_TEXT2))],
            ], colWidths=[W - 14*mm])
            inner.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), PDF_CARD),
                ("TOPPADDING",    (0,0),(-1,-1), 3),
                ("BOTTOMPADDING", (0,0),(-1,-1), 3),
                ("LEFTPADDING",   (0,0),(-1,-1), 8),
                ("RIGHTPADDING",  (0,0),(-1,-1), 8),
                ("LINEABOVE",     (0,0),(-1,0), 1.5, col),
            ]))
            row = Table([[
                Paragraph(tstr, ps(f"ts{i}{skey}", 7, col, align=TA_CENTER)),
                inner
            ]], colWidths=[14*mm, W - 14*mm])
            row.setStyle(TableStyle([
                ("VALIGN",        (0,0),(-1,-1),"TOP"),
                ("TOPPADDING",    (0,0),(-1,-1), 0),
                ("BOTTOMPADDING", (0,0),(-1,-1), 2),
            ]))
            story.append(row)
            story.append(Spacer(1, 1.5*mm))

        story.append(Spacer(1, 4*mm))

    # Footer
    story.append(HRFlowable(width=W, thickness=0.4, color=PDF_GOLD, spaceAfter=3*mm))
    story.append(Paragraph(
        f"Generated by Walk Through History  -  {datetime.now().strftime('%d %b %Y, %H:%M')}",
        ps("ft", 7, PDF_TEXT2, align=TA_CENTER)))

    doc.build(story)


# ════════════════════════════════════════════════════════════════════════════
#  HTML SHARE GENERATOR
# ════════════════════════════════════════════════════════════════════════════
def generate_share_html(city_name, tagline, total_days, budget_key,
                        style_name, days_data, n_monuments, start_date, budget_display=None):
    period_icons  = {"morning":"Sunrise  7 AM", "afternoon":"Sun  1 PM", "evening":"Moon  6 PM"}
    period_colors = {"morning":"#fcc419", "afternoon":"#d4a843", "evening":"#818cf8"}
    day_keys = list(days_data.keys())
    days_html = ""
    for i in range(total_days):
        dk      = day_keys[i % len(day_keys)]
        di      = days_data[dk]
        ds      = (start_date + timedelta(days=i)).strftime("%A, %d %b %Y")
        slots_h = ""
        for sk in ("morning","afternoon","evening"):
            s    = di[sk]
            ic   = period_icons[sk]
            col  = period_colors[sk]
            tags = "".join(f'<span class="tag">{t}</span>' for t in s["tags"])
            slots_h += f"""
            <div class="slot">
              <div class="slot-time" style="color:{col}">{ic}</div>
              <div class="slot-body">
                <div class="period" style="color:{col}">{sk.upper()}</div>
                <div class="act-name">{s['name']}</div>
                <div class="act-desc">{s['desc']}</div>
                <div class="tags">{tags}</div>
              </div>
            </div>"""
        days_html += f"""
        <div class="day-card">
          <div class="day-header">
            <span class="day-circle">D{i+1}</span>
            <div>
              <div class="day-title">Day {i+1} &mdash; {di['title']}</div>
              <div class="day-date">{ds}</div>
            </div>
          </div>
          <div class="day-body">{slots_h}</div>
        </div>"""

    cost = budget_display or BUDGET_COST_DEFAULT.get(budget_key, "Rs.9,000")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{city_name} Heritage Journey</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e8e0d4;font-family:'Segoe UI',Arial,sans-serif;font-size:15px;line-height:1.6}}
.hero{{text-align:center;padding:3rem 1rem 1.5rem;background:radial-gradient(ellipse 70% 50% at 50% 0%,rgba(212,168,67,.07),transparent 70%)}}
.badge{{display:inline-block;border:1px solid rgba(212,168,67,.4);border-radius:20px;padding:5px 16px;font-size:12px;color:#d4a843;margin-bottom:1rem}}
h1{{font-size:clamp(1.8rem,5vw,3rem);font-weight:700;color:#e8e0d4;margin-bottom:.4rem}}
.tagline{{color:#9a9080;font-size:1rem;margin-bottom:1.5rem}}
.badges{{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:1rem}}
.b-gold{{background:#d4a843;color:#0d1117;border-radius:20px;padding:3px 14px;font-size:12px;font-weight:700}}
.b-out{{border:1px solid rgba(212,168,67,.4);color:#d4a843;border-radius:20px;padding:3px 12px;font-size:11px}}
.stats{{display:flex;justify-content:center;gap:2.5rem;padding:1rem 0;border-top:1px solid rgba(255,255,255,.07);border-bottom:1px solid rgba(255,255,255,.07);margin:1rem 0}}
.stat-n{{font-size:1.6rem;font-weight:700;color:#d4a843;display:block}}
.stat-l{{font-size:10px;color:#6a6055;text-transform:uppercase;letter-spacing:.08em}}
.content{{max-width:780px;margin:0 auto;padding:0 1rem 4rem}}
.day-card{{background:#161c26;border:1px solid rgba(255,255,255,.07);border-radius:12px;margin-bottom:1rem;overflow:hidden}}
.day-header{{display:flex;align-items:center;gap:12px;padding:14px 18px;background:#13181f}}
.day-circle{{width:36px;height:36px;border-radius:50%;background:rgba(212,168,67,.15);border:1px solid rgba(212,168,67,.3);display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#d4a843;flex-shrink:0}}
.day-title{{font-size:14px;font-weight:600;color:#e8e0d4}}
.day-date{{font-size:11px;color:#6a6055}}
.day-body{{padding:14px 18px;display:flex;flex-direction:column;gap:10px}}
.slot{{display:flex;gap:12px}}
.slot-time{{min-width:80px;font-size:11px;padding-top:2px}}
.slot-body{{flex:1;background:#1a2130;border-radius:8px;padding:10px 14px;border:1px solid rgba(255,255,255,.07)}}
.period{{font-size:9px;font-weight:700;letter-spacing:.1em;margin-bottom:3px}}
.act-name{{font-size:13px;font-weight:600;color:#e8e0d4;margin-bottom:3px}}
.act-desc{{font-size:11px;color:#9a9080;line-height:1.5}}
.tags{{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}}
.tag{{font-size:9px;padding:2px 8px;border:1px solid rgba(255,255,255,.1);border-radius:10px;color:#6a6055}}
footer{{text-align:center;padding:2rem;color:#6a6055;font-size:11px;border-top:1px solid rgba(255,255,255,.07)}}
</style>
</head>
<body>
<div class="hero">
  <div class="badge">* Heritage Itinerary</div>
  <h1>{city_name} Heritage Journey</h1>
  <p class="tagline">{tagline}</p>
  <div class="badges">
    <span class="b-gold">{total_days} Days</span>
    <span class="b-out">{budget_key.capitalize()} Budget</span>
    <span class="b-out">{style_name} Pace</span>
  </div>
  <div class="stats">
    <div><span class="stat-n">{total_days}</span><span class="stat-l">Days</span></div>
    <div><span class="stat-n">{total_days*3}</span><span class="stat-l">Experiences</span></div>
    <div><span class="stat-n">{n_monuments}</span><span class="stat-l">Monuments</span></div>
    <div><span class="stat-n">{cost}</span><span class="stat-l">Est./Day</span></div>
  </div>
</div>
<div class="content">
{days_html}
</div>
<footer>Generated by Walk Through History &nbsp;&bull;&nbsp; {datetime.now().strftime('%d %b %Y, %H:%M')}</footer>
</body>
</html>"""

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False,
                                      encoding="utf-8", prefix="travel_itinerary_")
    tmp.write(html)
    tmp.flush()
    tmp.close()
    return tmp.name


# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════
class TravelPlannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Travel Planner — Walk Through History")
        self.configure(bg=BG)
        self.geometry("1200x820")
        self.minsize(900, 700)

        self.selected_duration = tk.IntVar(value=2)
        self.selected_budget   = tk.StringVar(value="mid")
        self.selected_style    = tk.StringVar(value="relaxed")
        self.dest_var          = tk.StringVar(value="")
        self.interests_state   = {i: tk.BooleanVar(value=i in ("History","Architecture")) for i in INTERESTS}

        self._dur_btns    = {}
        self._budget_btns = {}
        self._style_btns  = {}
        self._int_btns    = {}
        self._current     = None   # holds last generated itinerary data

        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_hero()
        self._build_main_area()

    def _build_hero(self):
        hero = tk.Frame(self, bg=BG, pady=30)
        hero.pack(fill="x")
        bf = tk.Frame(hero, bg=BG2, bd=1, relief="solid")
        bf.pack()
        tk.Label(bf, text="* AI-Powered Travel Planner", bg=BG2, fg=GOLD,
                 font=("Segoe UI", 10), padx=14, pady=5).pack()
        tk.Label(hero, text="Plan Your Journey", bg=BG, fg=TEXT,
                 font=("Georgia", 34, "bold")).pack(pady=(12, 6))
        tk.Label(hero,
                 text="Tell us where you want to go and our AI will craft a\npersonalised day-by-day heritage travel itinerary.",
                 bg=BG, fg=TEXT2, font=("Georgia", 12), justify="center").pack()

    def _build_main_area(self):
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        # ── Left: scrollable form ──
        fo = tk.Frame(outer, bg=BG)
        fo.pack(side="left", fill="y", pady=4, padx=(0, 12))
        fc = tk.Canvas(fo, bg=BG, bd=0, highlightthickness=0, width=480)
        fs = tk.Scrollbar(fo, orient="vertical", command=fc.yview)
        fc.configure(yscrollcommand=fs.set)
        fs.pack(side="right", fill="y")
        fc.pack(side="left", fill="both", expand=True)
        fi = tk.Frame(fc, bg=BG2, bd=1, relief="solid", padx=22, pady=22)
        fw = fc.create_window((0,0), window=fi, anchor="nw")
        fi.bind("<Configure>", lambda e: fc.configure(scrollregion=fc.bbox("all")))
        fc.bind("<Configure>", lambda e: fc.itemconfig(fw, width=e.width))
        self._build_form(fi)

        # ── Right: scrollable result ──
        ro = tk.Frame(outer, bg=BG)
        ro.pack(side="left", fill="both", expand=True, pady=4)
        self.result_canvas = tk.Canvas(ro, bg=BG, bd=0, highlightthickness=0)
        rs = tk.Scrollbar(ro, orient="vertical", command=self.result_canvas.yview)
        self.result_canvas.configure(yscrollcommand=rs.set)
        rs.pack(side="right", fill="y")
        self.result_canvas.pack(side="left", fill="both", expand=True)
        self.result_inner = tk.Frame(self.result_canvas, bg=BG)
        rw = self.result_canvas.create_window((0,0), window=self.result_inner, anchor="nw")
        self.result_inner.bind("<Configure>",
            lambda e: self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all")))
        self.result_canvas.bind("<Configure>",
            lambda e: self.result_canvas.itemconfig(rw, width=e.width))
        self._build_empty_state(self.result_inner)

    # ── FORM ─────────────────────────────────────────────────────────────────
    def _build_form(self, parent):
        def section(text):
            tk.Label(parent, text=text, bg=BG2, fg=TEXT3,
                     font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", pady=(14,4))

        section("DESTINATION")
        self.dest_combo = ttk.Combobox(parent, textvariable=self.dest_var,
                                       values=[d[0] for d in DESTINATIONS_LIST],
                                       state="readonly", font=("Segoe UI", 11))
        self.dest_combo.current(0)
        self.dest_combo.pack(fill="x", ipady=6)
        sty = ttk.Style()
        sty.theme_use("clam")
        sty.configure("TCombobox", fieldbackground=BG3, background=BG3, foreground=TEXT,
                       selectbackground=BG3, selectforeground=TEXT,
                       bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=TEXT3)
        sty.map("TCombobox", fieldbackground=[("readonly", BG3)])

        # ── NLP QUERY FIELD (ML Engine 2) ────────────────────────────────────
        tk.Label(parent, text='NLP: describe your trip (optional)',
                 bg=BG2, fg=TEXT3, font=("Segoe UI", 8)).pack(anchor="w", pady=(6,2))
        self._nlp_var = tk.StringVar()
        nlp_entry = tk.Entry(parent, textvariable=self._nlp_var,
                             bg=BG3, fg=TEXT2, font=("Segoe UI", 10),
                             insertbackground=GOLD, bd=0, relief="flat",
                             highlightthickness=1, highlightbackground=BORDER,
                             highlightcolor=GOLD)
        nlp_entry.pack(fill="x", ipady=6)
        nlp_entry.insert(0, 'Enter place Name')
        nlp_entry.bind("<FocusIn>",
            lambda e: nlp_entry.delete(0, "end")
                      if nlp_entry.get().startswith('e.g.') else None)
        nlp_entry.bind("<FocusOut>",
            lambda e: nlp_entry.insert(0, 'e.g. "forts and spicy food near Delhi"')
                      if not nlp_entry.get().strip() else None)

        section("DURATION (DAYS)")
        dr = tk.Frame(parent, bg=BG2)
        dr.pack(fill="x")
        for d in DURATIONS:
            btn = tk.Button(dr, text=str(d), width=4, height=2,
                            bg=BG3, fg=TEXT2, relief="flat", bd=1,
                            font=("Segoe UI", 11, "bold"), cursor="hand2",
                            highlightbackground=BORDER, highlightthickness=1,
                            command=lambda v=d: self._select_duration(v))
            btn.pack(side="left", padx=4)
            self._dur_btns[d] = btn
        self._select_duration(2)

        section("BUDGET")
        br = tk.Frame(parent, bg=BG2)
        br.pack(fill="x")
        for icon, name, val, rng in [("$","Budget","budget","< Rs.5k/day"),
                                      ("**","Mid","mid","Rs.5-15k/day"),
                                      ("Crown","Luxury","luxury","Rs.25k+/day")]:
            f = tk.Frame(br, bg=BG3, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1, cursor="hand2")
            f.pack(side="left", expand=True, fill="x", padx=4, ipady=6)
            tk.Label(f, text=icon, bg=BG3, fg=TEXT, font=("Segoe UI", 14)).pack()
            tk.Label(f, text=name, bg=BG3, fg=TEXT, font=("Segoe UI", 10, "bold")).pack()
            tk.Label(f, text=rng,  bg=BG3, fg=TEXT3, font=("Segoe UI", 8)).pack()
            f.bind("<Button-1>", lambda e, v=val, fr=f: self._select_budget(v, fr))
            for w in f.winfo_children():
                w.bind("<Button-1>", lambda e, v=val, fr=f: self._select_budget(v, fr))
            self._budget_btns[val] = f
        self._select_budget("mid", self._budget_btns["mid"])

        section("INTERESTS")
        igf = tk.Frame(parent, bg=BG2)
        igf.pack(fill="x")
        row_f = None
        for i, interest in enumerate(INTERESTS):
            if i % 5 == 0:
                row_f = tk.Frame(igf, bg=BG2)
                row_f.pack(fill="x", pady=2)
            btn = tk.Button(row_f, text=interest, bg=BG3, fg=TEXT2,
                            relief="flat", bd=1, font=("Segoe UI", 9),
                            padx=8, pady=4, cursor="hand2",
                            highlightbackground=BORDER, highlightthickness=1,
                            command=lambda x=interest: self._toggle_interest(x))
            btn.pack(side="left", padx=3)
            self._int_btns[interest] = btn
        for interest in INTERESTS:
            if self.interests_state[interest].get():
                self._int_btns[interest].configure(fg=GOLD, highlightbackground=GOLD)

        section("TRAVEL STYLE")
        sr = tk.Frame(parent, bg=BG2)
        sr.pack(fill="x")
        for icon, name in STYLES:
            val = name.lower()
            f = tk.Frame(sr, bg=BG3, bd=1, relief="solid",
                         highlightbackground=BORDER, highlightthickness=1, cursor="hand2")
            f.pack(side="left", expand=True, fill="x", padx=4, ipady=6)
            tk.Label(f, text=icon, bg=BG3, fg=TEXT, font=("Segoe UI", 12)).pack()
            tk.Label(f, text=name, bg=BG3, fg=TEXT2, font=("Segoe UI", 9)).pack()
            f.bind("<Button-1>", lambda e, v=val, fr=f, n=name: self._select_style(v, fr, n))
            for w in f.winfo_children():
                w.bind("<Button-1>", lambda e, v=val, fr=f, n=name: self._select_style(v, fr, n))
            self._style_btns[val] = (f, name)
        self._select_style("relaxed", self._style_btns["relaxed"][0], "Relaxed")

        section("TRAVELLERS")
        tr = tk.Frame(parent, bg=BG2)
        tr.pack(fill="x")
        self._trav_btns = {}
        self._selected_trav = tk.StringVar(value="couple")
        for t in ("Solo","Couple","Family","Group"):
            val = t.lower()
            btn = tk.Button(tr, text=t, bg=BG3, fg=TEXT2, relief="flat", bd=1,
                            font=("Segoe UI", 9), padx=12, pady=6, cursor="hand2",
                            highlightbackground=BORDER, highlightthickness=1,
                            command=lambda v=val: self._select_trav(v))
            btn.pack(side="left", padx=4)
            self._trav_btns[val] = btn
        self._select_trav("couple")

        tk.Frame(parent, bg=BG2, height=16).pack()
        self.gen_btn = tk.Button(parent, text="*  Generate AI Itinerary",
                                  bg=GOLD, fg=BG, relief="flat", bd=0,
                                  font=("Segoe UI", 13, "bold"), pady=12,
                                  cursor="hand2", command=self._on_generate)
        self.gen_btn.pack(fill="x")
        self.gen_btn.bind("<Enter>", lambda e: self.gen_btn.configure(bg=GOLD2))
        self.gen_btn.bind("<Leave>", lambda e: self.gen_btn.configure(bg=GOLD))

    # ── TOGGLES ──────────────────────────────────────────────────────────────
    def _select_duration(self, val):
        self.selected_duration.set(val)
        for d, btn in self._dur_btns.items():
            btn.configure(bg=GOLD if d==val else BG3, fg=BG if d==val else TEXT2,
                          highlightbackground=GOLD if d==val else BORDER)

    def _select_budget(self, val, frame):
        self.selected_budget.set(val)
        for v, f in self._budget_btns.items():
            f.configure(highlightbackground=GOLD if v==val else BORDER)

    def _select_style(self, val, frame, name):
        self.selected_style.set(val)
        for v, (f, n) in self._style_btns.items():
            f.configure(highlightbackground=GOLD if v==val else BORDER)
            for w in f.winfo_children():
                if isinstance(w, tk.Label):
                    w.configure(fg=GOLD if (v==val and w.cget("text")==n) else TEXT2)

    def _toggle_interest(self, interest):
        cur = self.interests_state[interest].get()
        self.interests_state[interest].set(not cur)
        self._int_btns[interest].configure(
            fg=GOLD if not cur else TEXT2,
            highlightbackground=GOLD if not cur else BORDER)

    def _select_trav(self, val):
        self._selected_trav.set(val)
        for v, btn in self._trav_btns.items():
            btn.configure(bg=GOLD if v==val else BG3, fg=BG if v==val else TEXT2,
                          highlightbackground=GOLD if v==val else BORDER)

    # ── EMPTY / LOADING ───────────────────────────────────────────────────────
    def _build_empty_state(self, parent):
        self._clear_result()
        f = tk.Frame(parent, bg=BG2, bd=1, relief="solid", padx=30, pady=40)
        f.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Label(f, text="Map", bg=BG2, font=("Segoe UI", 52), fg=TEXT2).pack(pady=(0,10))
        tk.Label(f, text="Your AI Itinerary Awaits", bg=BG2, fg=TEXT,
                 font=("Georgia", 18, "bold")).pack()
        tk.Label(f, text="Fill in your travel preferences and let AI craft\nyour perfect India heritage journey.",
                 bg=BG2, fg=TEXT2, font=("Segoe UI", 11), justify="center").pack(pady=(6,16))
        tb = tk.Frame(f, bg=BG3, bd=1, relief="solid", padx=14, pady=10)
        tb.pack(fill="x", padx=30)
        for tip in ["Choose a destination with rich monument history",
                    "Select interests tailored to your preferences",
                    "Longer durations unlock hidden gems nearby",
                    "Budget affects hotel and dining suggestions"]:
            row = tk.Frame(tb, bg=BG3)
            row.pack(fill="x", pady=3)
            tk.Label(row, text="*", bg=BG3, fg=GOLD, font=("Segoe UI", 12)).pack(side="left")
            tk.Label(row, text=tip, bg=BG3, fg=TEXT2, font=("Segoe UI", 9),
                     wraplength=360, justify="left").pack(side="left", padx=6)

    def _clear_result(self):
        for w in self.result_inner.winfo_children():
            w.destroy()

    def _show_loading(self):
        self._clear_result()
        f = tk.Frame(self.result_inner, bg=BG2, bd=1, relief="solid", padx=30, pady=60)
        f.pack(fill="both", expand=True, padx=4, pady=4)
        tk.Label(f, text="...", bg=BG2, fg=GOLD, font=("Segoe UI", 36)).pack()
        tk.Label(f, text="Crafting your itinerary...", bg=BG2, fg=TEXT2,
                 font=("Segoe UI", 13)).pack(pady=(12,4))
        self._loading_step_lbl = tk.Label(f, text=LOADING_STEPS[0], bg=BG2, fg=TEXT3,
                                           font=("Segoe UI", 10))
        self._loading_step_lbl.pack()

    # ── GENERATE ─────────────────────────────────────────────────────────────
    def _on_generate(self):
        dest_name = self.dest_var.get()
        if not dest_name or dest_name.startswith("--"):
            messagebox.showwarning("No Destination", "Please select a destination first.")
            return
        dest_key = next((k for n,k in DESTINATIONS_LIST if n==dest_name), "")

        # ── ML ENGINE 2: NLP query parsing (if user typed in the hint field) ──
        nlp_city    = None
        nlp_ints    = {}
        if hasattr(self, "_nlp_var") and self._nlp_var.get().strip():
            nlp_city, nlp_ints = nlp_parse_query(self._nlp_var.get())
            # Pre-tick detected interests
            if nlp_ints:
                for interest, val in nlp_ints.items():
                    if interest in self.interests_state:
                        self.interests_state[interest].set(bool(val))
                        self._int_btns[interest].configure(
                            fg=GOLD if val else TEXT2,
                            highlightbackground=GOLD if val else BORDER)
            # Override destination if NLP found a city
            if nlp_city:
                dest_key = nlp_city
                matched_name = next((n for n,k in DESTINATIONS_LIST if k==nlp_city), dest_name)
                dest_name = matched_name

        # ── ML ENGINE 1: KNN recommendation ──────────────────────────────────
        interest_scores = {i: (1 if self.interests_state[i].get() else 0)
                           for i in self.interests_state}
        knn_top3 = knn_recommend(interest_scores, top_n=3)
        self._knn_suggestion = knn_top3   # store for display in result header

        # If selected city is NOT in top-3, note the best KNN alternative
        self._knn_best = knn_top3[0] if dest_key not in knn_top3 else None

        self.gen_btn.configure(state="disabled", text="  Generating...")
        self._show_loading()
        self.update()

        def cycle():
            for msg in LOADING_STEPS:
                time.sleep(0.42)
                try:
                    self._loading_step_lbl.configure(text=msg)
                    self._loading_step_lbl.update()
                except Exception:
                    return
            time.sleep(0.3)
            self.after(0, lambda: self._finish_generate(dest_key, dest_name))

        threading.Thread(target=cycle, daemon=True).start()

    def _finish_generate(self, dest_key, dest_name):
        self._build_itinerary(dest_key, dest_name)
        self.gen_btn.configure(state="normal", text="Regenerate Itinerary")

    # ── ITINERARY ─────────────────────────────────────────────────────────────
    def _build_itinerary(self, dest_key, dest_name):
        self._clear_result()
        data       = DESTINATION_DATA.get(dest_key)
        city_name  = data["name"] if data else dest_name
        days_data  = data["days"] if data else default_days(city_name.split(",")[0])
        total_days = self.selected_duration.get()
        budget_key = self.selected_budget.get()
        style_name = self.selected_style.get().capitalize()
        tagline    = data["tagline"] if data else f"A curated {total_days}-day heritage experience"
        n_mons     = len(data["monuments"]) if data else 6
        start_date = datetime.now() + timedelta(days=7)

        # ── ML ENGINE 3: Linear Regression budget prediction ─────────────────
        ml_budget_display = predict_budget(total_days, budget_key)
        static_budget     = get_budget_cost(dest_key, budget_key)
        # Use ML-predicted value as the primary display
        budget_display = ml_budget_display

        # Store for PDF/share
        self._current = dict(city_name=city_name, tagline=tagline,
                             total_days=total_days, budget_key=budget_key,
                             style_name=style_name, days_data=days_data,
                             n_monuments=n_mons, start_date=start_date,
                             budget_display=budget_display)

        container = tk.Frame(self.result_inner, bg=BG)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        # ── ML INSIGHT PANEL ────────────────────────────────────────────────
        # Show KNN top-3 matches and NLP/budget predictions in a compact bar
        knn_top3   = getattr(self, "_knn_suggestion", [])
        knn_best   = getattr(self, "_knn_best", None)
        knn_names  = [next((n for n,k in DESTINATIONS_LIST if k==c), c.title())
                      for c in knn_top3]

        ml_panel = tk.Frame(container, bg=BG3, bd=1, relief="solid", padx=14, pady=10)
        ml_panel.pack(fill="x", pady=(0,6))

        tk.Label(ml_panel, text="AI/ML ANALYSIS", bg=BG3, fg=GOLD,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")

        row1 = tk.Frame(ml_panel, bg=BG3)
        row1.pack(fill="x", pady=(4,0))
        tk.Label(row1, text="KNN Top Matches:", bg=BG3, fg=TEXT3,
                 font=("Segoe UI", 8)).pack(side="left")
        for i, nm in enumerate(knn_names):
            is_sel = (knn_top3[i] == dest_key) if i < len(knn_top3) else False
            tk.Label(row1, text=f"  {nm.split(',')[0]}",
                     bg=BG3, fg=GOLD if is_sel else TEXT2,
                     font=("Segoe UI", 8, "bold" if is_sel else "normal")).pack(side="left")

        row2 = tk.Frame(ml_panel, bg=BG3)
        row2.pack(fill="x", pady=(3,0))
        tk.Label(row2, text="ML Budget (Linear Regression):", bg=BG3, fg=TEXT3,
                 font=("Segoe UI", 8)).pack(side="left")
        tk.Label(row2, text=f"  {ml_budget_display}/day",
                 bg=BG3, fg=GOLD, font=("Segoe UI", 8, "bold")).pack(side="left")
        tk.Label(row2, text=f"  (static lookup: {static_budget})",
                 bg=BG3, fg=TEXT3, font=("Segoe UI", 8)).pack(side="left")

        if knn_best:
            best_name = next((n for n,k in DESTINATIONS_LIST if k==knn_best), knn_best.title())
            row3 = tk.Frame(ml_panel, bg=BG3)
            row3.pack(fill="x", pady=(3,0))
            tk.Label(row3,
                     text=f"KNN suggests '{best_name.split(',')[0]}' better matches your interests.",
                     bg=BG3, fg=MORNING, font=("Segoe UI", 8)).pack(anchor="w")

        # Header
        hdr = tk.Frame(container, bg=BG2, bd=1, relief="solid", padx=18, pady=16)
        hdr.pack(fill="x")
        br = tk.Frame(hdr, bg=BG2)
        br.pack(fill="x", pady=(0,8))
        for txt, gold in [(f"{total_days} Days",True),(f"{budget_key.capitalize()} Budget",False),(f"{style_name} Pace",False)]:
            tk.Label(br, text=txt, bg=GOLD if gold else BG3, fg=BG if gold else GOLD,
                     font=("Segoe UI",9,"bold"), padx=10, pady=3).pack(side="left", padx=4)
        short = city_name.split(",")[0]
        tk.Label(hdr, text=f"{short} Heritage Journey", bg=BG2, fg=TEXT,
                 font=("Georgia",16,"bold"), anchor="w").pack(fill="x")
        tk.Label(hdr, text=tagline, bg=BG2, fg=TEXT2,
                 font=("Segoe UI",10), anchor="w").pack(fill="x", pady=(3,10))
        tk.Frame(hdr, bg=BORDER, height=1).pack(fill="x", pady=(0,8))
        sr = tk.Frame(hdr, bg=BG2)
        sr.pack(fill="x")
        for num, lbl in [(str(total_days),"Days"),(str(total_days*3),"Experiences"),
                          (str(n_mons),"Monuments"),(budget_display,"Est./Day")]:
            s = tk.Frame(sr, bg=BG2)
            s.pack(side="left", padx=18)
            tk.Label(s, text=num, bg=BG2, fg=GOLD, font=("Georgia",16,"bold")).pack()
            tk.Label(s, text=lbl.upper(), bg=BG2, fg=TEXT3, font=("Segoe UI",8)).pack()

        # Days
        day_keys = list(days_data.keys())
        self._day_bodies = {}
        for i in range(total_days):
            dk   = day_keys[i % len(day_keys)]
            di   = days_data[dk]
            ds   = (start_date + timedelta(days=i)).strftime("%a, %d %b")
            card = tk.Frame(container, bg=CARD, bd=1, relief="solid")
            card.pack(fill="x")
            dh = tk.Frame(card, bg=CARD, cursor="hand2", pady=10)
            dh.pack(fill="x", padx=16)
            tk.Label(dh, text=f"D{i+1}", bg=BG2, fg=GOLD, font=("Segoe UI",10,"bold"),
                     width=3, padx=6, pady=4).pack(side="left", padx=(0,10))
            inf = tk.Frame(dh, bg=CARD)
            inf.pack(side="left")
            tk.Label(inf, text=f"Day {i+1} - {di['title']}", bg=CARD, fg=TEXT,
                     font=("Segoe UI",11,"bold")).pack(anchor="w")
            tk.Label(inf, text=ds, bg=CARD, fg=TEXT3, font=("Segoe UI",9)).pack(anchor="w")
            chev = tk.Label(dh, text="v", bg=CARD, fg=TEXT3, font=("Segoe UI",9))
            chev.pack(side="right")
            body = tk.Frame(card, bg=CARD)
            self._day_bodies[i] = {"body":body, "open":False, "chev":chev}

            for period, sk, icon, ts, col in [
                ("Morning","morning","Sunrise","7 AM",MORNING),
                ("Afternoon","afternoon","Sun","1 PM",GOLD),
                ("Evening","evening","Moon","6 PM",EVENING)]:
                self._build_slot(body, period, sk, icon, ts, di[sk], col)

            def toggle(e, idx=i):
                entry = self._day_bodies[idx]
                if entry["open"]:
                    entry["body"].pack_forget()
                    entry["chev"].configure(text="v")
                    entry["open"] = False
                else:
                    entry["body"].pack(fill="x", padx=14, pady=(0,12))
                    entry["chev"].configure(text="^")
                    entry["open"] = True

            dh.bind("<Button-1>", toggle)
            for w in dh.winfo_children():
                w.bind("<Button-1>", toggle)

            if i == 0:
                body.pack(fill="x", padx=14, pady=(0,12))
                chev.configure(text="^")
                self._day_bodies[i]["open"] = True

        # Action bar
        ab = tk.Frame(container, bg=BG2, bd=1, relief="solid", padx=14, pady=12)
        ab.pack(fill="x")

        pdf_btn = tk.Button(ab, text="Download PDF",
                            bg=GOLD, fg=BG, relief="flat", bd=0,
                            font=("Segoe UI",10,"bold"), padx=18, pady=8,
                            cursor="hand2", command=self._download_pdf)
        pdf_btn.pack(side="left", padx=4)
        pdf_btn.bind("<Enter>", lambda e: pdf_btn.configure(bg=GOLD2))
        pdf_btn.bind("<Leave>", lambda e: pdf_btn.configure(bg=GOLD))

        share_btn = tk.Button(ab, text="Share Link",
                              bg=BG3, fg=TEXT2, relief="flat", bd=1,
                              font=("Segoe UI",10), padx=18, pady=8,
                              cursor="hand2",
                              highlightbackground=BORDER, highlightthickness=1,
                              command=self._share_link)
        share_btn.pack(side="left", padx=4)
        share_btn.bind("<Enter>", lambda e: share_btn.configure(fg=TEXT, highlightbackground=GOLD))
        share_btn.bind("<Leave>", lambda e: share_btn.configure(fg=TEXT2, highlightbackground=BORDER))

        save_btn = tk.Button(ab, text="Save",
                             bg=BG3, fg=TEXT2, relief="flat", bd=1,
                             font=("Segoe UI",10), padx=18, pady=8,
                             cursor="hand2",
                             highlightbackground=BORDER, highlightthickness=1,
                             command=self._save)
        save_btn.pack(side="left", padx=4)

    def _build_slot(self, parent, period, slot_key, icon, time_str, data, color):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=4)
        tc = tk.Frame(row, bg=CARD, width=56)
        tc.pack(side="left", fill="y", padx=(0,8))
        tc.pack_propagate(False)
        tk.Label(tc, text=time_str, bg=CARD, fg=TEXT3, font=("Segoe UI",8)).pack(pady=(2,2))
        tk.Label(tc, text=icon, bg=CARD, fg=color, font=("Segoe UI",10)).pack()
        ct = tk.Frame(row, bg=BG3, bd=1, relief="solid", padx=12, pady=8)
        ct.pack(side="left", fill="both", expand=True)
        tk.Label(ct, text=period.upper(), bg=BG3,
                 fg={"Morning":MORNING,"Afternoon":GOLD,"Evening":EVENING}.get(period,GOLD),
                 font=("Segoe UI",8,"bold")).pack(anchor="w")
        tk.Label(ct, text=data["name"], bg=BG3, fg=TEXT,
                 font=("Segoe UI",11,"bold")).pack(anchor="w", pady=(2,0))
        tk.Label(ct, text=data["desc"], bg=BG3, fg=TEXT2,
                 font=("Segoe UI",9), wraplength=400, justify="left").pack(anchor="w")
        tr = tk.Frame(ct, bg=BG3)
        tr.pack(anchor="w", pady=(4,0))
        for tag in data["tags"]:
            tk.Label(tr, text=tag, bg=BG3, fg=TEXT3,
                     font=("Segoe UI",8), bd=1, relief="solid", padx=6, pady=2).pack(side="left", padx=2)

    # ── ACTIONS ──────────────────────────────────────────────────────────────
    def _download_pdf(self):
        if not self._current:
            messagebox.showwarning("No Itinerary", "Please generate an itinerary first.")
            return
        c = self._current
        default_name = f"{c['city_name'].split(',')[0].replace(' ','_')}_Heritage_Itinerary.pdf"
        filepath = filedialog.asksaveasfilename(
            title="Save Itinerary as PDF",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF Files","*.pdf"),("All Files","*.*")]
        )
        if not filepath:
            return
        try:
            generate_pdf(
                filepath=filepath,
                city_name=c["city_name"],
                tagline=c["tagline"],
                total_days=c["total_days"],
                budget_key=c["budget_key"],
                style_name=c["style_name"],
                days_data=c["days_data"],
                n_monuments=c["n_monuments"],
                start_date=c["start_date"],
                budget_display=c.get("budget_display"),
            )
            if messagebox.askyesno("PDF Saved",
                    f"Itinerary PDF saved to:\n{filepath}\n\nOpen it now?"):
                webbrowser.open(f"file:///{filepath.replace(os.sep, '/')}")
        except Exception as ex:
            messagebox.showerror("PDF Error", f"Could not generate PDF:\n{ex}")

    def _share_link(self):
        if not self._current:
            messagebox.showwarning("No Itinerary", "Please generate an itinerary first.")
            return
        c = self._current
        try:
            html_path = generate_share_html(
                city_name=c["city_name"],
                tagline=c["tagline"],
                total_days=c["total_days"],
                budget_key=c["budget_key"],
                style_name=c["style_name"],
                days_data=c["days_data"],
                n_monuments=c["n_monuments"],
                start_date=c["start_date"],
                budget_display=c.get("budget_display"),
            )
            file_url = f"file:///{html_path.replace(os.sep, '/')}"
            webbrowser.open(file_url)

            # Share dialog
            win = tk.Toplevel(self)
            win.title("Share Itinerary")
            win.configure(bg=BG2)
            win.geometry("540x230")
            win.resizable(False, False)
            win.grab_set()

            tk.Label(win, text="Share Your Itinerary", bg=BG2, fg=GOLD,
                     font=("Georgia",14,"bold")).pack(pady=(18,6))
            tk.Label(win,
                     text="Your itinerary is now open in your browser.\nCopy the file path below to share:",
                     bg=BG2, fg=TEXT2, font=("Segoe UI",10), justify="center").pack()

            lf = tk.Frame(win, bg=BG3, bd=1, relief="solid")
            lf.pack(fill="x", padx=24, pady=10)
            link_var = tk.StringVar(value=html_path)
            le = tk.Entry(lf, textvariable=link_var, bg=BG3, fg=TEXT,
                          font=("Segoe UI",9), bd=0,
                          readonlybackground=BG3, state="readonly")
            le.pack(fill="x", padx=10, pady=8)

            def copy_path():
                self.clipboard_clear()
                self.clipboard_append(html_path)
                messagebox.showinfo("Copied", "File path copied to clipboard!", parent=win)

            brow = tk.Frame(win, bg=BG2)
            brow.pack(pady=4)
            tk.Button(brow, text="Copy Path", bg=GOLD, fg=BG,
                      font=("Segoe UI",10,"bold"), relief="flat",
                      padx=14, pady=6, cursor="hand2",
                      command=copy_path).pack(side="left", padx=6)
            tk.Button(brow, text="Open in Browser", bg=BG3, fg=TEXT2,
                      font=("Segoe UI",10), relief="flat", bd=1,
                      padx=14, pady=6, cursor="hand2",
                      highlightbackground=BORDER, highlightthickness=1,
                      command=lambda: webbrowser.open(file_url)).pack(side="left", padx=6)
            tk.Button(brow, text="Close", bg=BG3, fg=TEXT2,
                      font=("Segoe UI",10), relief="flat", bd=1,
                      padx=14, pady=6, cursor="hand2",
                      command=win.destroy).pack(side="left", padx=6)
        except Exception as ex:
            messagebox.showerror("Share Error", f"Could not generate share page:\n{ex}")

    def _save(self):
        messagebox.showinfo("Saved", "Itinerary bookmarked!\n(Connect a database to persist saves across sessions.)")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TravelPlannerApp()
    app.mainloop()
