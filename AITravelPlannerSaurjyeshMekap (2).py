"""
Travel Planner — Walk Through History
Python/Tkinter desktop app with real PDF export and HTML share link.

Requirements:
    pip install reportlab

Run:
    python travel_planner.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, time, os, webbrowser, tempfile
from datetime import datetime, timedelta

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

# ── DESTINATION DATA ─────────────────────────────────────────────────────────
DESTINATION_DATA = {
    "agra": {
        "name": "Agra, Uttar Pradesh", "emoji": "Taj",
        "tagline": "The Eternal City of Love & Marble",
        "monuments": ["Taj Mahal","Agra Fort","Fatehpur Sikri","Mehtab Bagh","Itmad-ud-Daulah","Akbar's Tomb"],
        "days": {
            1: {"title": "The Taj & Its World",
                "morning":   {"name":"Taj Mahal at Sunrise","desc":"Enter the east gate at dawn — the marble turns a soft pink as the sun rises over the Yamuna. Spend 2 hours exploring the main mausoleum, mosque, and gardens.","tags":["UNESCO","Mughal","Photography"]},
                "afternoon": {"name":"Agra Fort","desc":"A UNESCO site and the power centre of the Mughal Empire. Walk through Diwan-i-Am, Diwan-i-Khas, and the octagonal tower where Shah Jahan spent his last years.","tags":["UNESCO","Fort","History"]},
                "evening":   {"name":"Mehtab Bagh at Sunset","desc":"Cross the Yamuna to this moonlit garden — the only place to see the Taj Mahal reflected perfectly in the river at golden hour.","tags":["Gardens","Sunset","Photography"]}},
            2: {"title": "Fatehpur Sikri & Hidden Gems",
                "morning":   {"name":"Fatehpur Sikri","desc":"Akbar's abandoned sandstone capital — a perfectly preserved 16th-century Mughal ghost city. The Buland Darwaza stands 54m tall, one of the world's great gateways.","tags":["UNESCO","Mughal","Architecture"]},
                "afternoon": {"name":"Itmad-ud-Daulah (Baby Taj)","desc":"The first Mughal structure built entirely in white marble with intricate pietra dura inlays — a precursor and inspiration for the Taj Mahal itself.","tags":["Mughal","Marble","Hidden Gem"]},
                "evening":   {"name":"Sadar Bazaar & Mughal Cuisine","desc":"Explore the bustling market for marble souvenirs and petha sweets. Dinner at a rooftop restaurant with a lit Taj Mahal view.","tags":["Food","Shopping","Nightlife"]}},
        }
    },
    "delhi": {
        "name": "Delhi", "emoji": "🔴",
        "tagline": "8 Cities, 3,000 Years of History",
        "monuments": ["Red Fort","Qutub Minar","Humayun's Tomb","India Gate","Lotus Temple","Purana Qila"],
        "days": {
            1: {"title": "Old Delhi & Mughal Grandeur",
                "morning":   {"name":"Red Fort at Dawn","desc":"Beat the crowds to the Mughal emperor's palace-fortress. Walk through Diwan-i-Khas, the pearl mosque, and the royal apartments.","tags":["UNESCO","Mughal","History"]},
                "afternoon": {"name":"Jama Masjid & Old Delhi","desc":"India's largest mosque, built by Shah Jahan for 5,000 worshippers. Explore the spice markets and narrow lanes of Chandni Chowk on a rickshaw.","tags":["Mughal","Islam","Culture"]},
                "evening":   {"name":"Qutub Minar by Evening Light","desc":"The 73m iron pillar complex at golden hour. The 800-year-old iron pillar that has never rusted remains one of metallurgy's great mysteries.","tags":["UNESCO","Delhi Sultanate","Architecture"]}},
            2: {"title": "Mughal Tombs & New Delhi",
                "morning":   {"name":"Humayun's Tomb","desc":"The architectural forerunner of the Taj Mahal — a UNESCO World Heritage Site and a masterpiece of Persian-influenced Mughal design.","tags":["UNESCO","Mughal","Garden"]},
                "afternoon": {"name":"India Gate & Rajpath","desc":"The 42m war memorial for 90,000 soldiers. Walk down the grand ceremonial boulevard to the Presidential Palace.","tags":["Colonial","Memorial","Architecture"]},
                "evening":   {"name":"Hauz Khas Village","desc":"Medieval stepwell ruins and a 14th-century madrasa complex surrounded by cafes, galleries, and restaurants.","tags":["Medieval","Food","Nightlife"]}},
        }
    },
    "jaipur": {
        "name": "Jaipur, Rajasthan", "emoji": "🌸",
        "tagline": "The Pink City of Rajput Splendour",
        "monuments": ["Amber Fort","Hawa Mahal","City Palace","Jantar Mantar","Nahargarh Fort","Jal Mahal"],
        "days": {
            1: {"title": "Amber Fort & the Hilltops",
                "morning":   {"name":"Amber Fort at Sunrise","desc":"Arrive before the crowds to see the hilltop fort bathed in golden light. Explore the Sheesh Mahal — the Hall of Mirrors.","tags":["UNESCO","Rajput","Fort"]},
                "afternoon": {"name":"Jaigarh & Nahargarh Forts","desc":"The twin guardian forts above Amber. Jaigarh houses the world's largest cannon on wheels; Nahargarh offers panoramic city views.","tags":["Fort","Views","History"]},
                "evening":   {"name":"Jal Mahal at Dusk","desc":"The Water Palace rising from Man Sagar Lake turns a glowing amber at sunset.","tags":["Palace","Photography","Sunset"]}},
            2: {"title": "The Pink City Within",
                "morning":   {"name":"Hawa Mahal & Bazaars","desc":"The Palace of Winds — 953 jharokha windows built so royal women could watch street life unseen.","tags":["Architecture","Shopping","Photography"]},
                "afternoon": {"name":"City Palace & Jantar Mantar","desc":"The living royal palace adjacent to the world's largest stone sundial observatory — a UNESCO World Heritage Site.","tags":["UNESCO","Royalty","Science"]},
                "evening":   {"name":"Chokhi Dhani Village Dinner","desc":"An immersive Rajasthani cultural village with folk dance, puppet shows, and an authentic thali dinner under the stars.","tags":["Culture","Food","Music"]}},
        }
    },
    "hampi": {
        "name": "Hampi, Karnataka", "emoji": "HMP",
        "tagline": "The Ruined Capital of a Lost Empire",
        "monuments": ["Virupaksha Temple","Vittala Temple","Lotus Mahal","Elephant Stables","Hemakuta Hill","Matanga Hill"],
        "days": {
            1: {"title": "Temples & Sacred River",
                "morning":   {"name":"Virupaksha Temple at Dawn","desc":"The living temple at the heart of Hampi — still active after 7 centuries. Climb the gopuram for a panorama of ruins.","tags":["Temple","Living Heritage","Dravidian"]},
                "afternoon": {"name":"Vittala Temple & Stone Chariot","desc":"The pinnacle of Vijayanagara architecture — the musical pillars that ring like bells and the iconic stone chariot.","tags":["UNESCO","Vijayanagara","Architecture"]},
                "evening":   {"name":"Tungabhadra River Sunset","desc":"Coracle boat ride on the boulder-strewn river at sunset. Watch the ruins glow orange from the water.","tags":["Nature","Sunset","Adventure"]}},
            2: {"title": "Royal Enclosure & Hilltops",
                "morning":   {"name":"Matanga Hill Sunrise","desc":"Climb 600 steps before dawn for a 360 degree panorama of Hampi's ruins.","tags":["Photography","Sunrise","Trekking"]},
                "afternoon": {"name":"Royal Enclosure & Lotus Mahal","desc":"The Zenana enclosure housing the exquisite Lotus Mahal — a blend of Hindu and Islamic styles.","tags":["Royalty","Architecture","History"]},
                "evening":   {"name":"Hemakuta Hill Temples","desc":"A hilltop strewn with pre-Vijayanagara Jain and Hindu temples with a 360 degree sunset view.","tags":["Jain","Sunset","Architecture"]}},
        }
    },
    "varanasi": {
        "name": "Varanasi, Uttar Pradesh", "emoji": "VNS",
        "tagline": "The Eternal City on the Ganges",
        "monuments": ["Kashi Vishwanath Temple","Dashashwamedh Ghat","Sarnath","Ramnagar Fort","Manikarnika Ghat"],
        "days": {
            1: {"title": "The Ghats & Sacred Fire",
                "morning":   {"name":"Sunrise Boat Ride","desc":"Drift along the ghats at dawn as the city wakes — priests, pilgrims, and the morning light on the ancient riverfront.","tags":["Sacred","Photography","Culture"]},
                "afternoon": {"name":"Kashi Vishwanath Temple","desc":"One of the 12 Jyotirlingas of Shiva — the holiest temple in the Hindu world, rebuilt by Ahilya Bai Holkar in 1780.","tags":["Temple","Hindu","History"]},
                "evening":   {"name":"Ganga Aarti at Dashashwamedh","desc":"A spectacular evening fire ritual with hundreds of oil lamps, incense, and chanting priests — one of India's most mesmerising spectacles.","tags":["Ritual","Fire","Sacred"]}},
            2: {"title": "Sarnath & the Old City",
                "morning":   {"name":"Sarnath — Where Buddha Spoke","desc":"The deer park where the Buddha gave his first sermon after enlightenment. The Dhamek Stupa stands on the exact spot.","tags":["Buddhist","UNESCO","History"]},
                "afternoon": {"name":"Ramnagar Fort & Museum","desc":"The 18th-century seat of the Maharajas of Varanasi, housing a museum of royal artefacts, vintage cars, and rare manuscripts.","tags":["Fort","Royalty","Museum"]},
                "evening":   {"name":"Silk Weaving Quarter","desc":"Visit the Banarasi weavers in their workshop — a tradition of silk weaving that has endured for 2,000 years.","tags":["Craft","Culture","Shopping"]}},
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
    "Analysing destination monuments...",
    "Mapping heritage sites by era...",
    "Optimising travel distances...",
    "Crafting morning activities...",
    "Selecting afternoon experiences...",
    "Curating evening highlights...",
    "Adding local food recommendations...",
    "Finalising your itinerary...",
]

BUDGET_COST         = {"budget": "Rs.4,200",  "mid": "Rs.11,500",  "luxury": "Rs.32,000"}
BUDGET_COST_DISPLAY = {"budget": "Rs.4,200",  "mid": "Rs.11,500",  "luxury": "Rs.32,000"}

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
                 style_name, days_data, n_monuments, start_date):
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
    cost = BUDGET_COST[budget_key]
    stats = [(str(total_days),"DAYS"),(str(total_days*3),"EXPERIENCES"),
             (str(n_monuments),"MONUMENTS"),(cost,"EST./DAY")]
    sd = [[Paragraph(n, ps(f"sn{i}",14,PDF_GOLD,bold=True,align=TA_CENTER)) for n,_ in stats],
          [Paragraph(l, ps(f"sl{i}", 7,PDF_TEXT2,align=TA_CENTER)) for _,l in stats]]
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
                        style_name, days_data, n_monuments, start_date):
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

    cost = BUDGET_COST_DISPLAY[budget_key]
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

        # Store for PDF/share
        self._current = dict(city_name=city_name, tagline=tagline,
                             total_days=total_days, budget_key=budget_key,
                             style_name=style_name, days_data=days_data,
                             n_monuments=n_mons, start_date=start_date)

        container = tk.Frame(self.result_inner, bg=BG)
        container.pack(fill="both", expand=True, padx=4, pady=4)

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
                          (str(n_mons),"Monuments"),(BUDGET_COST_DISPLAY[budget_key],"Est./Day")]:
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