import re
from config import HEADING_KEYWORDS

def extract_features(text, font_size, font_name, bbox, page_num, next_text):
    is_bold = 'Bold' in font_name or 'bold' in font_name
    is_italic = 'Italic' in font_name or 'Oblique' in font_name
    text_case = 'ALL_CAPS' if text.isupper() else 'Title' if text.istitle() else 'lower'
    starts_with_numbering = bool(re.match(r'^\d+(\.\d+)*\s+', text))
    has_heading_keywords = any(k in text.lower() for k in HEADING_KEYWORDS)
    ends_with_punctuation = text.strip()[-1:] in '.:;!?'
    word_count = len(text.split())
    line_width = bbox[2] - bbox[0]
    is_centered = 0.4 < (bbox[0] / 595) < 0.6  # page width = 595pt

    return {
        "text": text,
        "font_size": font_size,
        "font_name": font_name,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "text_case": text_case,
        "starts_with_numbering": starts_with_numbering,
        "has_heading_keywords": has_heading_keywords,
        "ends_with_punctuation": ends_with_punctuation,
        "word_count": word_count,
        "line_width": line_width,
        "is_centered": is_centered,
        "page_number": page_num,
        "next_text": next_text
    }
