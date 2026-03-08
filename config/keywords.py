"""
Financial event keyword lexicon for 8-K filing classification.
Each category maps to a list of trigger phrases and a weight.
"""

KEYWORD_LEXICON = {
    "acquisition": {
        "keywords": [
            "acquisition", "acquire", "merger", "merge", "takeover",
            "buyout", "purchase agreement", "definitive agreement",
            "business combination", "tender offer"
        ],
        "weight": 0.9
    },
    "divestiture": {
        "keywords": [
            "divestiture", "divest", "disposal", "sold subsidiary",
            "asset sale", "spin-off", "spinoff", "separation"
        ],
        "weight": 0.8
    },
    "bankruptcy": {
        "keywords": [
            "bankruptcy", "chapter 11", "chapter 7", "insolvency",
            "restructuring", "creditor", "debtor-in-possession"
        ],
        "weight": 1.0
    },
    "executive_change": {
        "keywords": [
            "ceo", "cfo", "coo", "chief executive", "chief financial",
            "resignation", "appointment", "terminated", "succession",
            "interim", "board of directors"
        ],
        "weight": 0.7
    },
    "earnings": {
        "keywords": [
            "earnings", "revenue", "net income", "eps",
            "guidance", "forecast", "outlook", "adjusted ebitda",
            "quarterly results", "annual results"
        ],
        "weight": 0.6
    },
    "restatement": {
        "keywords": [
            "restatement", "restate", "material weakness",
            "internal control", "audit committee", "non-reliance",
            "accounting error", "correction"
        ],
        "weight": 1.0
    },
    "litigation": {
        "keywords": [
            "litigation", "lawsuit", "settlement", "class action",
            "subpoena", "investigation", "sec enforcement", "complaint",
            "indictment", "regulatory action"
        ],
        "weight": 0.8
    },
    "capital_markets": {
        "keywords": [
            "offering", "issuance", "debt", "equity", "credit facility",
            "shelf registration", "convertible", "stock repurchase",
            "buyback", "dividend"
        ],
        "weight": 0.5
    },
}

# Urgency markers boost the signal score
URGENCY_MARKERS = [
    "material", "significant", "substantial", "immediately",
    "urgent", "critical", "extraordinary", "unprecedented"
]

# 8-K Item codes that are typically material
MATERIAL_ITEMS = [
    "1.01",  # Entry into material agreement
    "1.02",  # Termination of material agreement
    "2.01",  # Completion of acquisition/disposition
    "2.05",  # Costs of restructuring
    "2.06",  # Material impairment
    "4.01",  # Changes in auditor
    "4.02",  # Non-reliance on prior financials
    "5.02",  # Director/officer changes
]
