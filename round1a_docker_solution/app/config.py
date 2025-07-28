# Thresholds and weights
MIN_SCORE = 3.0

WEIGHTS = {
    "is_bold": 1.5,
    "font_size_rank": 1.5,
    "starts_with_numbering": 1.0,
    "is_centered": 1.0,
    "spacing_above": 1.0,
    "has_heading_keywords": 1.0,
    "ends_with_punctuation": -1.5
}

HEADING_KEYWORDS = ['introduction', 'chapter', 'overview', 'summary', 'conclusion']

# New filters
MAX_HEADING_WORDS = 12               # drop overly long “list” lines
FREQUENT_HEADING_PAGES = 3           # drop any line appearing on >3 distinct pages
HEADER_FOOTER_REL_POS = (0.05, 0.95)   # keep only lines between 10%–90% page height
