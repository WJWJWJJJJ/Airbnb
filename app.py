"""
AI dispute assistant for Airbnb (BC3415 group project demo)

Workflow: Intake chatbot -> RAG policy lookup -> Resolution agent -> Decision check
          -> Auto-resolve OR Human assessor -> Explanation to both sides

Student project, not affiliated with Airbnb. Bookings, people and policies are
simplified samples written for the demo, not Airbnb's actual terms.
"""
import os, re, html

import gradio as gr
from textblob import TextBlob

# ---------------- Settings ----------------
AUTO_LIMIT_USD = 500      # claims at or below this can be auto-resolved...
MIN_CONFIDENCE = 0.75     # ...if the agent's confidence is at least this
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
# Optional: add GEMINI_API_KEY under Environment in the Render dashboard.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def llm(prompt):
    """Gemini, if a key is set. Returns None otherwise so the app falls back to templates."""
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY).models.generate_content(model=LLM_MODEL, contents=prompt).text.strip()
    except Exception as e:
        print("LLM unavailable, using template:", e)
        return None


# ---------------- Sample data ----------------
BOOKINGS = {
    "HM-1001": {"guest": "Alicia", "host": "Ben", "nights": 3, "nightly_rate": 150, "policy": "Moderate"},
    "HM-2002": {"guest": "Chen", "host": "Dana", "nights": 5, "nightly_rate": 220, "policy": "Strict"},
    "HM-3003": {"guest": "Elif", "host": "Faisal", "nights": 2, "nightly_rate": 180, "policy": "Flexible"},
}

POLICIES = [
    {"title": "Essential amenity not working",
     "text": "If an essential amenity such as hot water, heating, Wi-Fi or electricity does not work, the guest must report it "
             "within 72 hours of check-in with photo or video evidence. The guest may receive a refund of up to 50% of the "
             "nightly rate for each affected night."},
    {"title": "Damage claims by hosts",
     "text": "Hosts must submit damage claims within 14 days of checkout, with photos of the damage and repair quotes or receipts. "
             "The guest is given a chance to respond. Claims above the automatic limit or disputed by the guest are reviewed "
             "by a specialist."},
    {"title": "Payments outside the platform",
     "text": "Requests to pay by bank transfer, cash or messaging apps outside the platform are a scam signal. The booking is "
             "protected only when paid through the platform. Suspected scams are escalated to the trust and safety team and "
             "the guest is refunded in full."},
    {"title": "Cancellation policies",
     "text": "Flexible: full refund up to 24 hours before check-in. Moderate: full refund up to 5 days before check-in. "
             "Strict: full refund within 48 hours of booking if check-in is at least 14 days away."},
]

CASES = {
    "No hot water (guest)": {
        "booking": "HM-1001", "raised_by": "guest",
        "message": "This is terrible. There has been no hot water since we checked in and the boiler shows an error light.",
        "evidence": ["photo of boiler error light", "video of cold tap"],
        "hours_after_checkin": 2, "days_after_checkout": None, "claimed_usd": None,
        "affected_nights": 3, "other_party": "Host confirms the boiler broke and a repair is booked.",
    },
    "Damaged TV and sofa (host)": {
        "booking": "HM-2002", "raised_by": "host",
        "message": "Guest left the TV screen cracked and wine stains on the sofa. Repair quotes attached.",
        "evidence": ["photos of cracked TV", "photos of stained sofa", "repair quotes"],
        "hours_after_checkin": None, "days_after_checkout": 3, "claimed_usd": 5000,
        "affected_nights": 0, "other_party": "Guest disputes: the TV was already cracked at check-in.",
    },
    "Host asks for bank transfer (guest)": {
        "booking": "HM-3003", "raised_by": "guest",
        "message": "The host messaged me on WhatsApp asking me to pay by bank transfer to get a discount. Is this normal?",
        "evidence": ["screenshot of WhatsApp chat"],
        "hours_after_checkin": None, "days_after_checkout": None, "claimed_usd": None,
        "affected_nights": 0, "other_party": "No response from host.",
    },
}


# ---------------- Step 1: Intake chatbot (LLM + sentiment) ----------------
ISSUE_TYPES = ["amenity", "damage", "scam", "cancellation"]


def sentiment(text):
    """TextBlob polarity with the course's 3-class rule (Week 6): > +0.2 positive, < -0.2 negative, else neutral."""
    p = round(TextBlob(text).sentiment.polarity, 2)
    label = "positive" if p > 0.2 else "negative" if p < -0.2 else "neutral"
    return p, label


def intake(case):
    label = llm("Classify this Airbnb support message into exactly one word from "
                f"{ISSUE_TYPES}. Reply with the word only.\n\nMessage: {case['message']}")
    label = (label or "").strip().lower()
    if label not in ISSUE_TYPES:
        t = case["message"].lower()
        if re.search(r"bank transfer|whatsapp|outside|cash|wire|paypal", t): label = "scam"
        elif re.search(r"damage|broke|cracked|stain|repair", t): label = "damage"
        elif re.search(r"cancel", t): label = "cancellation"
        else: label = "amenity"
    return label


# ---------------- Step 2: RAG policy lookup (TF-IDF vectorization + similarity search) ----------------
# TF-IDF (Week 7) turns each policy into a vector of word weights. It's light enough for Render's free plan.
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def _words(t):
    # lowercase, drop stop words, crude plural stemming ("rates" -> "rate")
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.findall(r"[a-z]+", t.lower()) if w not in ENGLISH_STOP_WORDS]


tfidf = TfidfVectorizer(analyzer=_words)
policy_vectors = tfidf.fit_transform([f"{p['title']}. {p['text']}" for p in POLICIES])   # the knowledge base


def policy_lookup(case, issue):
    scores = cosine_similarity(tfidf.transform([f"{issue}: {case['message']}"]), policy_vectors)[0]
    best = POLICIES[int(scores.argmax())]
    return {"title": best["title"], "text": best["text"], "score": round(float(scores.max()), 2)}


# ---------------- Step 3: Resolution agent (tools) ----------------
SCAM_PATTERNS = r"bank transfer|whatsapp|outside the (app|platform)|cash|wire|paypal|western union"


def tool_lookup_booking(case):
    return BOOKINGS[case["booking"]]


def tool_check_evidence(case, issue):
    conf, notes = 1.0, []
    if not case["evidence"]:
        conf -= 0.3; notes.append("no photo/video evidence")
    else:
        n = len(case["evidence"])
        notes.append(f"{n} piece{'s' if n > 1 else ''} of evidence")
    if issue == "amenity" and (case["hours_after_checkin"] or 0) > 72:
        conf -= 0.4; notes.append("reported after the 72-hour window")
    if issue == "damage" and (case["days_after_checkout"] or 0) > 14:
        conf -= 0.4; notes.append("claim filed after the 14-day window")
    other = case["other_party"].lower()
    if "dispute" in other:
        conf -= 0.4; notes.append("other party disputes the claim")
    elif "confirm" in other:
        notes.append("other party confirms")
    return max(round(conf, 2), 0.0), notes


def tool_scam_screen(case):
    return bool(re.search(SCAM_PATTERNS, case["message"].lower()))


def tool_calculate(case, issue, booking):
    if issue == "amenity":
        return 0.5 * booking["nightly_rate"] * case["affected_nights"], "50% of nightly rate x affected nights"
    if issue == "damage":
        return case["claimed_usd"] or 0, "host's claimed repair cost"
    if issue == "scam":
        return booking["nights"] * booking["nightly_rate"], "full booking value (refund if paid off-platform)"
    return booking["nights"] * booking["nightly_rate"], "booking value under cancellation policy"


def resolution_agent(case, issue):
    booking = tool_lookup_booking(case)
    confidence, notes = tool_check_evidence(case, issue)
    fraud = tool_scam_screen(case)
    amount, basis = tool_calculate(case, issue, booking)
    return {"booking": booking, "confidence": confidence, "notes": notes, "fraud": fraud,
            "amount": amount, "basis": basis}


# ---------------- Steps 4-6: Decision, outcome, explanation ----------------
def decide(a, issue):
    reasons = []
    if issue == "damage" and not a["amount"]:
        reasons.append("no claimed amount or repair quote provided")
    if a["amount"] > AUTO_LIMIT_USD:
        reasons.append(f"amount US${a['amount']:,.0f} is above the US${AUTO_LIMIT_USD} auto limit")
    if a["confidence"] < MIN_CONFIDENCE:
        reasons.append(f"confidence {a['confidence']:.0%} is below {MIN_CONFIDENCE:.0%}")
    if a["fraud"]:
        reasons.append("possible scam detected")
    return ("auto" if not reasons else "human"), reasons


def explain(case, issue, policy, a, route):
    b = a["booking"]
    ai = llm("Write a short, neutral message (max 70 words) to both an Airbnb guest and host explaining this outcome. "
             f"Cite the policy title. Case: {case['message']} Policy: {policy['title']}: {policy['text']} "
             f"Route: {route}. Amount: US${a['amount']:,.0f} ({a['basis']}).")
    if ai:
        return ai
    if route == "auto":
        return (f"Hi {b['guest']} and {b['host']}, under our '{policy['title']}' policy, {b['guest']} receives a refund of "
                f"US${a['amount']:,.0f} ({a['basis']}). This was resolved automatically. Either of you can appeal.") \
            if issue != "damage" else \
               (f"Hi {b['guest']} and {b['host']}, under our '{policy['title']}' policy, {b['host']} receives a payout of "
                f"US${a['amount']:,.0f} ({a['basis']}). This was resolved automatically. Either of you can appeal.")
    return (f"Hi {b['guest']} and {b['host']}, your case falls under our '{policy['title']}' policy. A specialist is "
            "now reviewing it and will contact you both. You do not need to resend your evidence.")


def human_summary(case, issue, policy, a, reasons, senti):
    ai = llm("Summarise this dispute for a human Airbnb case assessor in 4 bullet points: what happened, evidence, "
             f"applicable policy, suggested next step. Case: {case} Policy: {policy['title']}. Agent findings: {a}. "
             f"Escalated because: {reasons}.")
    return ai or (f"Issue: {issue} raised by {case['raised_by']} (sentiment: {senti[1]}). {case['message']} | "
                  f"Evidence: {', '.join(case['evidence'])} | Other party: {case['other_party']} | "
                  f"Policy: {policy['title']} | Amount at stake: US${a['amount']:,.0f} | "
                  f"Escalated because: {'; '.join(reasons)}.")


def run_workflow(case):
    issue = intake(case)
    if issue == "damage":
        case = {**case, "raised_by": "host"}   # damage claims come from hosts
    senti = sentiment(case["message"])
    policy = policy_lookup(case, issue)
    a = resolution_agent(case, issue)
    route, reasons = decide(a, issue)
    return {"case": case, "issue": issue, "sentiment": senti, "policy": policy, "agent": a, "route": route,
            "reasons": reasons, "explanation": explain(case, issue, policy, a, route),
            "summary": human_summary(case, issue, policy, a, reasons, senti) if route == "human" else None}


# ---------------- Visual: the path each case takes ----------------
def render(r):
    a, e = r["agent"], html.escape
    on, off = "#1a7f37", "#c9ccd1"

    def box(title, concept, detail, active=True, accent=None):
        c = accent or (on if active else off)
        return (f'<div style="flex:1;min-width:150px;border:2px solid {c};border-radius:10px;padding:10px;'
                f'background:{c if active else "#fff"}14;opacity:{1 if active else .45}">'
                f'<div style="font-weight:700;color:#111">{title}</div>'
                f'<div style="font-size:11px;color:{c};font-weight:600;margin:2px 0 6px">{concept}</div>'
                f'<div style="font-size:12px;color:#333">{detail}</div></div>')

    arrow = '<div style="align-self:center;font-size:20px;color:#888">&rarr;</div>'
    either = '<div style="align-self:center;font-size:12px;font-weight:700;color:#888">or</div>'
    auto, human = r["route"] == "auto", r["route"] == "human"
    pol, slabel = r["sentiment"]
    priority = " &middot; <b style='color:#b42318'>priority</b>" if slabel == "negative" else ""
    flags = " &middot; <b style='color:#b42318'>scam flag</b>" if a["fraud"] else ""
    row1 = arrow.join([
        box("1. Intake chatbot", "LLM + TextBlob sentiment",
            f"Issue type: <b>{r['issue']}</b><br>Sentiment: <b>{slabel}</b> ({pol:+.2f}){priority}<br>"
            f"Raised by {r['case']['raised_by']}"),
        box("2. RAG policy lookup", "TF-IDF + similarity search",
            f"Matched: <b>{e(r['policy']['title'])}</b><br>Similarity score: {r['policy']['score']:.2f}"),
        box("3. Resolution agent", "AI agent + tools",
            f"Amount: <b>US${a['amount']:,.0f}</b><br>Confidence: <b>{a['confidence']:.0%}</b>{flags}<br>"
            f"<span style='font-size:11px'>{e('; '.join(a['notes']))}</span>"),
        box("4. Decision check", "Business rule",
            f"&le; US${AUTO_LIMIT_USD}, &ge; {MIN_CONFIDENCE:.0%} confidence, no scam?<br><b>{'Yes' if auto else 'No'}</b>"),
    ])
    row2 = (box("5a. Auto-resolve", "Automation",
                (f"Payout of US${a['amount']:,.0f} to host issued in minutes" if r["issue"] == "damage" else f"Refund of US${a['amount']:,.0f} issued in minutes") if auto else "Skipped", auto)
            + either
            + box("5b. Human assessor", "Human-in-the-loop",
                  ("Escalated: " + e("; ".join(r["reasons"]))) if human else "Skipped", human,
                  "#b54708" if human else None)
            + arrow
            + box("6. Explanation to both sides", "GenAI + transparency", e(r["explanation"])))
    summary = (f'<div style="margin-top:10px;padding:10px;border-left:4px solid #b54708;background:#fff8f1;'
               f'font-size:12px;color:#333"><b>AI case summary for the assessor</b><br>{e(r["summary"])}</div>'
               ) if human else ""
    return (f'<div style="font-family:Arial,sans-serif;background:#fff;padding:14px;border-radius:12px">'
            f'<div style="font-size:16px;font-weight:700;margin-bottom:4px;color:#111">Case: {e(r["case"]["message"])}</div>'
            f'<div style="font-size:12px;color:#555;margin-bottom:12px">Booking {r["case"]["booking"]} &middot; '
            f'Route: <b style="color:{"#1a7f37" if auto else "#b54708"}">{"AUTO-RESOLVED" if auto else "SENT TO HUMAN"}</b></div>'
            f'<div style="display:flex;gap:6px;margin-bottom:10px">{row1}</div>'
            f'<div style="display:flex;gap:6px">{row2}</div>{summary}</div>')


# ---------------- App ----------------
def run(sample, custom_text, booking, claimed):
    case = dict(CASES[sample])
    if custom_text and custom_text.strip():
        case.update({"message": custom_text.strip(), "booking": booking, "claimed_usd": claimed or None,
                     "evidence": ["customer upload"], "other_party": "No response yet.",
                     "hours_after_checkin": 1, "days_after_checkout": 2, "affected_nights": 1})
    return render(run_workflow(case))


with gr.Blocks(title="Airbnb AI dispute assistant") as demo:
    gr.Markdown(
        "## AI dispute assistant for Airbnb\n"
        "Pick a sample case or type your own complaint, then watch the path it takes. "
        "Green = steps taken, grey = skipped.\n\n"
        "*BC3415 student project, not affiliated with Airbnb. Bookings and policies are simplified samples.*")
    with gr.Row():
        sample = gr.Dropdown(list(CASES), value=list(CASES)[0], label="Sample case")
        booking = gr.Dropdown(list(BOOKINGS), value="HM-1001", label="Booking (for your own complaint)")
        claimed = gr.Number(label="Claimed amount in US$ (for a damage claim)", value=0)
    custom = gr.Textbox(label="Or type your own complaint (leave empty to use the sample case)",
                        placeholder="e.g. The Wi-Fi hasn't worked since we arrived")
    go = gr.Button("Run the case", variant="primary")
    out = gr.HTML(run(list(CASES)[0], "", "HM-1001", 0))
    go.click(run, [sample, custom, booking, claimed], out)

if __name__ == "__main__":
    # Render tells the app which port to use through the PORT variable
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

