import re
from config import HEADING_KEYWORDS, MAX_HEADING_WORDS

# date‑like lines (drop them)
DATE_PAT = re.compile(r'^\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}$')

def infer_numbered_level(text):
    t = text.strip()
    if re.match(r'^\d+\.\d+\.\d+', t):
        return "H3"
    if re.match(r'^\d+\.\d+', t):
        return "H2"
    if re.match(r'^\d+\.', t):
        return "H1"
    return None

def is_valid_line(f):
    t  = f["text"].strip()
    wc = f["word_count"]
    # must have a letter
    if not any(c.isalpha() for c in t): 
        return False
    # drop date or pure digits
    if t.isdigit() or DATE_PAT.match(t): 
        return False
    # drop paragraphs (too many words)
    if wc > MAX_HEADING_WORDS: 
        return False
    # drop inline emphasis (mixed fonts)
    if not f.get("uniform_font", False): 
        return False
    return True

def assign_heading_level(features):
    out = []
    for f in features:
        if not is_valid_line(f):
            continue

        txt = f["text"].strip()
        pg  = f["page_number"]

        # 1) Numbered headings always take precedence
        lvl = infer_numbered_level(txt)
        if lvl:
            out.append({"text": txt, "page": pg, "level": lvl, "y0": f["y0"]})
            continue

        # 2) Front‑matter special case (pages 1–2)
        if pg in (1, 2):
            # H1: largest font cluster
            if f["font_size_rank"] == 0:
                lvl = "H1"
            # H2: second font cluster
            elif f["font_size_rank"] == 1:
                lvl = "H2"
            # H3: ends with colon (e.g. "Timeline:")
            elif txt.endswith(":"):
                lvl = "H3"
            # H2: any HEADING_KEYWORDS (“Summary”, etc.)
            elif any(kw in txt.lower() for kw in HEADING_KEYWORDS):
                lvl = "H2"
            else:
                continue

        # 3) Body‑matter: use global font‑rank
        else:
            if f["font_size_rank"] == 0:
                lvl = "H1"
            elif f["font_size_rank"] == 1:
                lvl = "H2"
            elif f["font_size_rank"] == 2:
                lvl = "H3"
            elif f["font_size_rank"] == 3:
                lvl = "H4"
            else:
                continue

            # allow H3 for colon lines even if rank==2 or 3
            if txt.endswith(":") and lvl not in ("H1","H2"):
                lvl = "H3"

            # keywords fallback for H2 in body
            if lvl in ("H3","H4") and any(kw in txt.lower() for kw in HEADING_KEYWORDS):
                lvl = "H2"

        out.append({"text": txt, "page": pg, "level": lvl, "y0": f["y0"]})

    return out
