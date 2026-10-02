"""Hybrid analyzer: weighted rules (English + Telugu) + ML as a supporting signal."""
import os
import re
import joblib

SCRIPTS = [  # (name key, unicode range)
    ("telugu", "\u0C00-\u0C7F"), ("devanagari", "\u0900-\u097F"), ("tamil", "\u0B80-\u0BFF"),
    ("kannada", "\u0C80-\u0CFF"), ("malayalam", "\u0D00-\u0D7F"), ("bengali", "\u0980-\u09FF"),
    ("gujarati", "\u0A80-\u0AFF"), ("gurmukhi", "\u0A00-\u0A7F"), ("odia", "\u0B00-\u0B7F"),
]
SCRIPT_RX = {name: re.compile(f"[{rng}]") for name, rng in SCRIPTS}
LATIN_RX = re.compile(r"[A-Za-z]")
ANY_INDIC = re.compile(r"[\u0900-\u0DFF]")
FULLY_SUPPORTED = {"latin", "telugu", "devanagari"}
THRESHOLD = 3      # total indicator weight needed before any warning is shown
MAX_WEIGHT = 7     # weight at which the rule score is 100%

# (key, weight, English regex, Telugu regex)
INDICATORS = [
    ("urgency", 1,
     r"\b(urgent(ly)?|immediately|right now|asap|act now|within \d+ ?(hours?|hrs|minutes|mins)|last chance|final warning|expires? (today|tonight|soon)|will expire|limited time)\b",
     r"అత్యవసరం|అర్జెంట్|వెంటనే|ఇప్పుడే|తక్షణమే|చివరి అవకాశం|గంటల్లో|గంటలలోపు|\b(ventane|ippude|twaraga|tvaraga)\b"),
    ("prize", 1,
     r"\b(you (have )?won|winner|lottery|lucky (draw|customer)|free (iphone|gift|recharge)|prize|selected for|congratulations[^.!?]{0,40}(won|win|prize|reward|selected|lucky))",
     r"గెలుచుకున్నారు|గెలిచారు|లాటరీ|లక్కీ డ్రా|బహుమతి"),
    ("money", 1,
     r"(₹|\brs\.?\s?\d|\binr\b|\bcashback\b|\brefund\b|\bpay\b|\btransfer\b|\bsend (me )?(money|funds)|\bfee\b|\bupi\b|gift cards?)",
     r"రూపాయ|₹|చెల్లించండి|చెల్లింపు|రీఫండ్|క్యాష్\u200c?బ్యాక్|ఫీజు|యూపీఐ|డబ్బు[^.!?]{0,20}(పంపండి|పంపించండి|పంపు)|\b(dabbulu|pampandi)\b"),
    ("link", 2,
     r"(bit\.ly|tinyurl|cutt\.ly|goo\.gl|\.xyz\b|\.top\b|\.click\b|\.icu\b|click (here|this|the) link|click on (the|this) link|https?://\S*(login|verify|update|kyc|bank|secure|claim|reward)\S*)",
     r"క్లిక్ చేయండి|లింక్[^.!?]{0,8}క్లిక్"),
    ("sensitive", 3,
     r"\b(shar(e|ing)|send(ing)?|tell|giv(e|ing)|enter|provide|verify|confirm|updat(e|ing)|submit|reveal)\b[^.!?]{0,40}\b(otp|upi pin|pin|cvv|password|aadhaar|pan|card details|bank details|card number)\b|\bkyc\b|\b(your|ur) (bank|card) details\b",
     r"(ఓటీపీ|ఒటిపి|పిన్|పాస్\u200c?వర్డ్|ఆధార్|కార్డ్ వివరాలు|బ్యాంక్ వివరాలు)[^.!?]{0,30}(చెప్పండి|పంపండి|ఇవ్వండి|షేర్ చేయండి|నమోదు చేయండి|వెరిఫై|ధృవీకరించ)|కేవైసీ"),
    ("threat", 1,
     r"\b(blocked|suspended|frozen|disconnected|arrest(ed)?|legal action|police|case registered|customs|will be closed|will be (cut|disconnected))\b",
     r"బ్లాక్|నిలిపివేయ|స్తంభింప|అరెస్ట్|అరెస్టు|పోలీస్|పోలీసు|కేసు నమోదు|కస్టమ్స్"),
    ("impersonation", 1,
     r"(it'?s me|this is my new number|my new number|\bboss here\b|i lost my phone|phone (was )?(lost|broke|broken)|stuck abroad|income tax department|\b(bank|customer care) (officer|executive|manager)\b|\bofficer\b)",
     r"నా కొత్త నంబర్|కొత్త నంబర్|ఫోన్ పోయింది|ఫోన్ పోగొట్టుకున్నా|ఫోన్ పగిలిపోయింది|విదేశాల్లో చిక్కుకు|ఆదాయపు పన్ను శాఖ|బ్యాంక్ అధికారి"),
    ("new_payee", 2,
     r"(new (upi|account|number)|different (upi|account)|this (new )?(upi id|account))",
     r"కొత్త\s*(UPI|యూపీఐ|అకౌంట్|ఖాతా)"),
    ("too_good", 2,
     r"(earn (rs\.?\s?)?₹?\s?\d[\d,]* ?(daily|per day|a day|weekly)|double your money|guaranteed (returns?|profit)|registration fee|processing fee|liking videos|without documents)",
     r"రోజుకు[^.!?]{0,15}సంపాదించ|రెట్టింపు|రిజిస్ట్రేషన్ ఫీజు|లైక్ చేసి"),
]

HINDI = {'urgency': 'तुरंत|फौरन|अभी|जल्दी|आखिरी मौका|अंतिम मौका|घंटे में|घंटों में|\\b(turant|foran|abhi|jaldi)\\b', 'prize': 'आपने[^.!?]{0,30}जीत|आप जीत गए|लॉटरी|लकी ड्रॉ|इनाम|बधाई[^.!?]{0,40}(जीत|इनाम|लॉटरी)|\\b(jeet gaye|jeet gaya|inaam|inam)\\b', 'money': 'रुपये|रुपए|₹|भुगतान|पेमेंट|रिफंड|कैशबैक|शुल्क|फीस|यूपीआई|ट्रांसफर|पैसे[^.!?]{0,20}(भेजें|भेजो|भेजिए)|\\b(bhejo|bhejiye|bhejein|rupaye|rupay)\\b', 'link': 'लिंक[^.!?]{0,8}क्लिक|क्लिक करें|\\b(click karo|click kare|click kijiye)\\b', 'sensitive': '(ओटीपी|ओ टी पी|otp|पिन|pin|cvv|सीवीवी|पासवर्ड|password|आधार|कार्ड (की )?(जानकारी|विवरण)|बैंक (की )?(जानकारी|विवरण))[^.!?]{0,30}(बताएं|बताओ|बताइए|भेजें|भेजो|भेजिए|साझा|शेयर|डालें|दर्ज|सत्यापित|वेरिफाई|दीजिए)|केवाईसी|\\b(otp|pin|cvv|password)\\b[^.!?]{0,20}\\b(batao|bataiye|bataye|bhejo|bhejiye|dijiye|share (karo|kare|kijiye))\\b', 'threat': 'ब्लॉक|निलंबित|फ्रीज|गिरफ्तार|पुलिस|केस दर्ज|कस्टम|कानूनी कार्रवाई|बंद हो जाएगा|कट जाएगा|\\b(block ho|band ho jayega|band ho jaayega|giraftar)\\b', 'impersonation': 'मेरा नया नंबर|नया नंबर|फोन खो गया|फोन गुम|फोन टूट|विदेश में फंस|आयकर विभाग|बैंक अधिकारी|\\b(naya number|mera naya number|phone kho gaya|phone gum)\\b', 'new_payee': '(नया|नए|नई)\\s*(UPI|यूपीआई|अकाउंट|खाता)|\\bnaya (upi|account)\\b', 'too_good': '(रोजाना|रोज़ाना|रोज़|रोज|हर दिन|दिन में)[^.!?]{0,20}कमा|दोगुना|दोगुने|रजिस्ट्रेशन फीस|प्रोसेसिंग फीस|\\b(roz kamao|daily kamao|paisa double)\\b'}

COMPILED = []
for key, weight, en, te in INDICATORS:
    COMPILED.append((key, weight, re.compile(f"(?:{en})|(?:{te})|(?:{HINDI[key]})", re.IGNORECASE)))

COMBOS = [("prize", "money"), ("threat", "money"), ("threat", "impersonation")]

# "Do not share your OTP" is a genuine warning, not a request for the OTP
NEGATION = re.compile(
    r"\b(do not|don'?t|never|not) (to )?shar(e|ing)\b|ఎవరితోనూ[^.!?]{0,20}(పంచుకోవద్దు|షేర్ చేయవద్దు|చెప్పవద్దు)|పంచుకోవద్దు|చెప్పవద్దు|साझा न करें|शेयर न करें|किसी को न बताएं|न बताएं",
    re.IGNORECASE)

TEXT = {
    "en": {
        "labels": {
            "urgency": "Urgency / pressure",
            "prize": "Unexpected prize / reward",
            "money": "Financial incentive or payment request",
            "link": "Suspicious link",
            "sensitive": "Request for sensitive information",
            "threat": "Threat or fear tactic",
            "impersonation": "Possible impersonation",
            "new_payee": "Changed / new payment details",
            "too_good": "Too-good-to-be-true offer",
        },
        "why": {
            "urgency": "Scammers rush you so you don't have time to think or verify.",
            "prize": "You can't win a contest you never entered.",
            "money": "Unexpected money offers or requests for fees are common scam hooks.",
            "link": "Links in unsolicited messages can lead to fake websites.",
            "sensitive": "Genuine organisations never ask for your OTP, PIN or password.",
            "threat": "Fear is used to make you act without checking.",
            "impersonation": "Scammers pretend to be friends, family, officials or banks.",
            "new_payee": "A sudden change in who to pay is a classic warning sign.",
            "too_good": "Offers of easy daily income or doubled money are almost always scams.",
        },
        "advice": {
            "link": "Do not click the link. Open the official website/app by typing the address yourself.",
            "sensitive": "Never share OTP, PIN, CVV or passwords with anyone, even if they claim to be from a bank.",
            "money": "Do not send money. Verify the request independently first.",
            "impersonation": "Contact the person or organisation through a different trusted channel (call the saved number or the official helpline).",
            "new_payee": "Confirm the new payment details directly with the person before paying.",
            "urgency": "Pause. Genuine requests can wait a few minutes for verification.",
            "prize": "Ignore unexpected prize or reward claims.",
            "threat": "Authorities and banks do not threaten you by SMS or chat. Call their official number to check.",
            "too_good": "Do not trust promises of easy profit, and never pay a fee upfront.",
        },
        "none_advice": "No common indicators found, but stay careful. If unsure, verify with the sender through another channel.",
        "levels": {
            "empty": "No message",
            "none": "No common scam indicators detected",
            "few": "Few scam indicators detected",
            "some": "Some scam indicators detected",
            "several": "Several strong scam indicators detected",
        },
        "explain_found": "This message contains {n} common scam indicator(s).",
        "explain_none": "This message does not match the common scam indicators we check.",
        "disclaimer": "This is an awareness aid, not a verdict. It can make mistakes. Always verify independently.",
    },
    "te": {
        "labels": {
            "urgency": "తొందర పెట్టడం / అత్యవసరం",
            "prize": "ఊహించని బహుమతి / లాటరీ",
            "money": "డబ్బు ఆఫర్ లేదా చెల్లింపు అభ్యర్థన",
            "link": "అనుమానాస్పద లింక్",
            "sensitive": "OTP / PIN / పాస్‌వర్డ్ వంటి గోప్య సమాచారం అడగడం",
            "threat": "భయపెట్టే బెదిరింపు",
            "impersonation": "వేరొకరిలా నటించే అవకాశం",
            "new_payee": "కొత్త చెల్లింపు వివరాలు",
            "too_good": "నమ్మశక్యం కాని ఆఫర్",
        },
        "why": {
            "urgency": "మీరు ఆలోచించి నిర్ధారించుకునే సమయం ఇవ్వకుండా మోసగాళ్లు తొందర పెడతారు.",
            "prize": "మీరు పాల్గొనని పోటీలో గెలవడం సాధ్యం కాదు.",
            "money": "ఊహించని డబ్బు ఆఫర్లు లేదా ఫీజు అడగడం మోసాలలో సాధారణం.",
            "link": "తెలియని సందేశాల్లోని లింకులు నకిలీ వెబ్‌సైట్లకు తీసుకెళ్లవచ్చు.",
            "sensitive": "నిజమైన సంస్థలు మీ OTP, PIN లేదా పాస్‌వర్డ్ ఎప్పుడూ అడగవు.",
            "threat": "తనిఖీ చేయకుండా వెంటనే స్పందించేలా భయాన్ని ఉపయోగిస్తారు.",
            "impersonation": "మోసగాళ్లు స్నేహితులు, కుటుంబ సభ్యులు, అధికారులు లేదా బ్యాంక్ వారిలా నటిస్తారు.",
            "new_payee": "ఎవరికి చెల్లించాలో అకస్మాత్తుగా మారడం ఒక ప్రసిద్ధ హెచ్చరిక సంకేతం.",
            "too_good": "రోజూ సులభంగా సంపాదన లేదా డబ్బు రెట్టింపు ఆఫర్లు దాదాపు ఎల్లప్పుడూ మోసాలే.",
        },
        "advice": {
            "link": "లింక్‌పై క్లిక్ చేయవద్దు. అధికారిక వెబ్‌సైట్ లేదా యాప్‌ను మీరే టైప్ చేసి తెరవండి.",
            "sensitive": "OTP, PIN, CVV లేదా పాస్‌వర్డ్‌లను ఎవరితోనూ పంచుకోవద్దు, బ్యాంక్ వారమని చెప్పినా సరే.",
            "money": "డబ్బు పంపవద్దు. ముందు స్వతంత్రంగా నిర్ధారించుకోండి.",
            "impersonation": "వేరే నమ్మదగిన మార్గంలో (సేవ్ చేసిన నంబర్‌కు కాల్ లేదా అధికారిక హెల్ప్‌లైన్) ఆ వ్యక్తిని లేదా సంస్థను సంప్రదించండి.",
            "new_payee": "చెల్లించే ముందు కొత్త చెల్లింపు వివరాలను ఆ వ్యక్తితో నేరుగా నిర్ధారించుకోండి.",
            "urgency": "ఆగండి. నిజమైన అభ్యర్థనలు నిర్ధారణ కోసం కొన్ని నిమిషాలు ఆగగలవు.",
            "prize": "ఊహించని బహుమతులు లేదా రివార్డుల సందేశాలను పట్టించుకోవద్దు.",
            "threat": "అధికారులు, బ్యాంకులు SMS లేదా చాట్ ద్వారా బెదిరించవు. వారి అధికారిక నంబర్‌కు కాల్ చేసి నిర్ధారించుకోండి.",
            "too_good": "సులభ లాభం హామీలను నమ్మవద్దు, ముందుగా ఎలాంటి ఫీజు చెల్లించవద్దు.",
        },
        "none_advice": "సాధారణ సూచికలు కనిపించలేదు, అయినా జాగ్రత్తగా ఉండండి. అనుమానం ఉంటే పంపినవారిని మరో మార్గంలో సంప్రదించి నిర్ధారించుకోండి.",
        "levels": {
            "empty": "సందేశం లేదు",
            "none": "సాధారణ మోసం సూచికలు కనిపించలేదు",
            "few": "కొన్ని తక్కువ స్థాయి మోసం సూచికలు గుర్తించబడ్డాయి",
            "some": "కొన్ని మోసం సూచికలు గుర్తించబడ్డాయి",
            "several": "అనేక బలమైన మోసం సూచికలు గుర్తించబడ్డాయి",
        },
        "explain_found": "ఈ సందేశంలో {n} సాధారణ మోసం సూచిక(లు) ఉన్నాయి.",
        "explain_none": "ఈ సందేశం మేము తనిఖీ చేసే సాధారణ మోసం సూచికలకు సరిపోలలేదు.",
        "disclaimer": "ఇది అవగాహన కోసం మాత్రమే, తుది నిర్ణయం కాదు. ఇందులో తప్పులు ఉండవచ్చు. ఎల్లప్పుడూ స్వతంత్రంగా నిర్ధారించుకోండి.",
    },
}


LANG_NAMES = {
    "en": {"latin": "English / Roman letters", "telugu": "Telugu", "devanagari": "Hindi / Marathi (Devanagari)",
           "tamil": "Tamil", "kannada": "Kannada", "malayalam": "Malayalam", "bengali": "Bengali",
           "gujarati": "Gujarati", "gurmukhi": "Punjabi", "odia": "Odia"},
    "te": {"latin": "ఇంగ్లీష్ / రోమన్ అక్షరాలు", "telugu": "తెలుగు", "devanagari": "హిందీ / మరాఠీ (దేవనాగరి)",
           "tamil": "తమిళం", "kannada": "కన్నడ", "malayalam": "మలయాళం", "bengali": "బెంగాలీ",
           "gujarati": "గుజరాతీ", "gurmukhi": "పంజాబీ", "odia": "ఒడియా"},
}
LANG_NOTE = {
    "en": "This message also contains {langs}. Full keyword checking is available for English, Telugu and Hindi. "
          "For other languages only language-independent signals (links, ₹ amounts, UPI, KYC, OTP) are checked, "
          "so some scams may be missed. Please be extra careful.",
    "te": "ఈ సందేశంలో {langs} భాష కూడా ఉంది. పూర్తి పదాల తనిఖీ ఇంగ్లీష్, తెలుగు, హిందీకి మాత్రమే ఉంది. "
          "ఇతర భాషలకు భాషతో సంబంధం లేని సూచికలు (లింకులు, ₹ మొత్తాలు, UPI, KYC, OTP) మాత్రమే తనిఖీ చేయబడతాయి, "
          "కాబట్టి కొన్ని మోసాలు గుర్తించబడకపోవచ్చు. దయచేసి మరింత జాగ్రత్తగా ఉండండి.",
}


def detect_scripts(text):
    found = []
    if len(LATIN_RX.findall(text)) >= 3:
        found.append("latin")
    for name, rx in SCRIPT_RX.items():
        if len(rx.findall(text)) >= 2:
            found.append(name)
    return found


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def analyze(text: str, lang: str = "en") -> dict:
    lang = lang if lang in TEXT else "en"
    t = TEXT[lang]
    text = (text or "").strip()

    hits = {key: weight for key, weight, rx in COMPILED if rx.search(text)}
    if "sensitive" in hits and NEGATION.search(text):
        del hits["sensitive"]          # e.g. "Do not share your OTP"

    total = sum(hits.values())
    # Combinations that are much more suspicious together than alone
    for a, b in COMBOS:
        if a in hits and b in hits:
            total += 1
    flagged = total >= THRESHOLD       # weak signals alone (casual chat) stay at 0
    keys = [k for k, _, _ in COMPILED if k in hits] if flagged else []

    # ML was trained on English only, and is just a supporting signal
    ml_score = None
    if flagged and _model is not None and not ANY_INDIC.search(text):
        ml_score = float(_model.predict_proba([text])[0][1])

    rule_score = min(total / MAX_WEIGHT, 1.0) if flagged else 0.0
    score = rule_score if ml_score is None else 0.7 * rule_score + 0.3 * ml_score

    if not text:
        level = t["levels"]["empty"]
    elif not flagged:
        level = t["levels"]["none"]
    elif score >= 0.60:
        level = t["levels"]["several"]
    elif score >= 0.35:
        level = t["levels"]["some"]
    else:
        level = t["levels"]["few"]

    scripts = detect_scripts(text)
    names_ = LANG_NAMES[lang]
    unsupported = [names_[x] for x in scripts if x not in FULLY_SUPPORTED]

    found = [{"key": k, "label": t["labels"][k], "why": t["why"][k]} for k in keys]
    actions = [t["advice"][k] for k in keys] or [t["none_advice"]]

    return {
        "lang": lang,
        "level": level,
        "score": round(score * 100),
        "ml_score": None if ml_score is None else round(ml_score * 100),
        "indicators_found": found,
        "explanation": t["explain_found"].format(n=len(found)) if found else t["explain_none"],
        "actions": actions,
        "disclaimer": t["disclaimer"],
        "detected_languages": ", ".join(names_[x] for x in scripts),
        "language_note": LANG_NOTE[lang].format(langs=", ".join(unsupported)) if unsupported else "",
    }
