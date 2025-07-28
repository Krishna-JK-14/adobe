import fitz, sys, os, json, re
from collections import defaultdict
from features import extract_features
from utils import cluster_font_sizes
from config import FREQUENT_HEADING_PAGES, HEADER_FOOTER_REL_POS
from classifier import assign_heading_level

# 0) Setup
pdf_path, output_dir = sys.argv[1], sys.argv[2]
doc = fitz.open(pdf_path)
N   = min(50, len(doc))

# Pattern to detect "Page X of Y" (case‑insensitive)
PAGE_PAT = re.compile(r'Page\s+(\d+)\s+of\s+\d+', re.IGNORECASE)
NUM_PAT  = re.compile(r'^\d+(\.\d+)*\s+')

# 1) Build printed‑page map
page_map = {}
for idx in range(N):
    for block in doc[idx].get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            txt = "".join(s["text"] for s in line["spans"]).strip()
            m   = PAGE_PAT.search(txt)
            if m and m.lastindex and m.lastindex >= 1:
                try:
                    page_map[idx] = int(m.group(1))
                except ValueError:
                    pass
                break
        if idx in page_map:
            break
# default to internal index+1
for idx in range(N):
    page_map.setdefault(idx, idx + 1)

# 2) Extract raw lines + collect font sizes
raw, sizes = [], []
for idx in range(N):
    page   = doc[idx]
    height = page.rect.height
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue
        for l in block["lines"]:
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            text      = " ".join(s["text"] for s in spans)
            bbox      = l["bbox"]
            rel_pos   = bbox[1] / height
            printed_p = page_map[idx]
            raw.append((text, spans, bbox, printed_p, rel_pos))
            sizes.extend([s["size"] for s in spans])

# 3) Cluster font sizes (4 clusters → H1…H4)
font_rank = cluster_font_sizes(sizes, n_clusters=4)

# 4) Page-frequency map for boilerplate
pf = defaultdict(set)
for text, spans, bbox, pg, rel in raw:
    pf[text.strip()].add(pg)

# 5) Build feature dicts
features = []
for text, spans, bbox, pg, rel in raw:
    fs = spans[0]["size"]
    fn = spans[0]["font"]
    f  = extract_features(text, fs, fn, bbox, pg, "")
    f["font_size_rank"] = font_rank[fs]
    f["page_freq"]      = len(pf[text.strip()])
    f["rel_pos"]        = rel
    f["y0"]             = bbox[1]
    f["uniform_font"]   = (len({(s["size"], s["font"]) for s in spans}) == 1)
    features.append(f)

# 6) Title detection: single largest-font uniform line on printed page 1
title_cands = [
    f for f in features
    if (
        f["page_number"] == 1
        and f["uniform_font"]
        and not PAGE_PAT.search(f["text"].strip())
        and HEADER_FOOTER_REL_POS[0] < f["rel_pos"] < HEADER_FOOTER_REL_POS[1]
        and f["page_freq"] <= FREQUENT_HEADING_PAGES
    )
]
if title_cands:
    max_size = max(f["font_size"] for f in title_cands)
    best     = min(
        (f for f in title_cands if f["font_size"] == max_size),
        key=lambda f: f["y0"]
    )
    title = best["text"].strip()
else:
    title = ""

# 7) Pre‑filter: only bold or numbered, drop "Page X of Y" & boilerplate
filtered = [
    f for f in features
    if (
        (f["is_bold"] or NUM_PAT.match(f["text"].strip()))
        and not PAGE_PAT.search(f["text"].strip())
        and HEADER_FOOTER_REL_POS[0] < f["rel_pos"] < HEADER_FOOTER_REL_POS[1]
        and f["page_freq"] <= FREQUENT_HEADING_PAGES
    )
]

# 8) Assign headings
outline = assign_heading_level(filtered)

# 9) Sort by printed page then vertical position
outline.sort(key=lambda h: (h["page"], h["y0"]))

# 10) Merge split headings
merged, i, threshold = [], 0, 15
while i < len(outline):
    cur = outline[i]
    txt, pg, lvl, y0 = cur["text"], cur["page"], cur["level"], cur["y0"]
    j = i + 1
    while (
        j < len(outline)
        and outline[j]["page"] == pg
        and outline[j]["level"] == lvl
        and (outline[j]["y0"] - y0) < threshold
    ):
        txt += " " + outline[j]["text"]
        y0   = outline[j]["y0"]
        j  += 1
    merged.append({"text": txt.strip(), "page": pg, "level": lvl})
    i = j

# 11) Output JSON
os.makedirs(output_dir, exist_ok=True)
out_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
with open(os.path.join(output_dir, out_name), "w") as out:
    json.dump({"title": title, "outline": merged}, out, indent=2)
