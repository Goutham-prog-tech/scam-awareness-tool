"""Hybrid analyzer: rule-based indicators (Level 1/3) + ML probability (Level 2)."""
import os
import re
import joblib

# (key, label, regex, explanation)
INDICATORS = [
    ("urgency", "Urgency / pressure",
     r"\b(urgent|urgently|immediately|right now|now|today|tonight|within \d+ (hours|hrs|minutes)|last chance|final warning|expires?|limited time)\b",
     "Scammers rush you so you don't have time to think or verify."),
    ("prize", "Unexpected prize / reward",
     r"\b(congratulations|you (have )?won|winner|lottery|lucky (draw|customer)|free (iphone|gift|recharge)|prize|selected for)\b",
     "You can't win a contest you never entered."),
    ("money", "Financial incentive or payment request",
     r"(₹|\brs\.?\s?\d|\binr\b|\bcashback\b|\brefund\b|\bpay\b|\btransfer\b|send (me )?(money|funds)|\bfee\b|\bupi\b|gift cards?)",
     "Unexpected money offers or requests for fees are common scam hooks."),
    ("link", "Suspicious link",
     r"(https?://|www\.|bit\.ly|tinyurl|\.xyz\b|\.top\b|\.click\b|click (here|this|the) link|click the link)",
     "Links in unsolicited messages can lead to fake websites."),
    ("sensitive", "Request for sensitive information",
     r"\b(otp|upi pin|pin|cvv|password|aadhaar|pan card|kyc|card details|bank details)\b",
     "Genuine organisations never ask for your OTP, PIN or password."),
    ("threat", "Threat or fear tactic",
     r"\b(blocked|suspended|frozen|closed|disconnected|arrest(ed)?|legal action|police|case registered|customs)\b",
     "Fear is used to make you act without checking."),
    ("impersonation", "Possible impersonation",
     r"(it'?s me|this is my new number|my new number|\bboss here\b|i lost my phone|phone (was )?(lost|broke|broken)|stuck abroad|income tax department|your bank|\bofficer\b)",
     "Scammers pretend to be friends, family, officials or banks."),
    ("new_payee", "Changed / new payment details",
     r"(new (upi|account|number)|different (upi|account)|this (new )?(upi id|account))",
     "A sudden change in who to pay is a classic warning sign."),
]

ADVICE = {
    "link": "Do not click the link. Open the official website/app by typing the address yourself.",
    "sensitive": "Never share OTP, PIN, CVV or passwords with anyone, even if they claim to be from a bank.",
    "money": "Do not send money. Verify the request independently first.",
    "impersonation": "Contact the person or organisation through a different trusted channel (call the saved number or the official helpline).",
    "new_payee": "Confirm the new payment details directly with the person before paying.",
    "urgency": "Pause. Genuine requests can wait a few minutes for verification.",
    "prize": "Ignore unexpected prize or reward claims.",
    "threat": "Authorities and banks do not threaten you by SMS or chat. Call their official number to check.",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def analyze(text: str) -> dict:
    text = (text or "").strip()
    found, not_found = [], []
    for key, label, pattern, why in INDICATORS:
        hit = re.search(pattern, text, flags=re.IGNORECASE) is not None
        (found if hit else not_found).append({"key": key, "label": label, "why": why})

    ml_score = None
    if _model is not None and text:
        ml_score = float(_model.predict_proba([text])[0][1])

    rule_score = min(len(found) / 5, 1.0)          # 5+ indicators = max
    score = rule_score if ml_score is None else 0.5 * rule_score + 0.5 * ml_score

    if not text:
        level = "No message"
    elif score >= 0.60:
        level = "Several strong scam indicators detected"
    elif score >= 0.35:
        level = "Some scam indicators detected"
    else:
        level = "Few or no scam indicators detected"

    actions = [ADVICE[f["key"]] for f in found]
    if not found:
        actions = ["No common indicators found, but stay careful. If unsure, verify with the sender through another channel."]

    return {
        "level": level,
        "score": round(score * 100),
        "ml_score": None if ml_score is None else round(ml_score * 100),
        "indicators_found": found,
        "indicators_not_found": [n["label"] for n in not_found],
        "explanation": (f"This message contains {len(found)} common scam indicator(s)."
                        if found else "This message does not match the common scam indicators we check."),
        "actions": actions,
        "disclaimer": "This is an awareness aid, not a verdict. It can make mistakes. Always verify independently.",
    }