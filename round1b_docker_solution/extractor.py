import fitz  # PyMuPDF
import json
import os
import sys
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import datetime

# --- Configuration Constants ---
# Customize these for fine-tuning behavior if allowed by contest rules.
# Otherwise, use as provided.
HEADING_KEYWORDS = [
    "introduction", "summary", "abstract", "conclusion", "results",
    "methodology", "discussion", "chapter", "section", "overview",
    "background", "findings", "recommendations", "future work",
    "appendix", "references", "acknowledgements", "data", "analysis",
    "objectives", "scope", "definition", "purpose", "aims", "materials",
    "experimental setup", "literature review", "theory", "model", "implementation",
    "evaluation", "performance", "case study", "applications", "implications",
    "limitations", "challenges", "outlook", "prospects", "vision",
    "key points", "guidelines", "best practices", "faq", "glossary"
]
MAX_HEADING_WORDS = 15  # Max words a line can have to be considered a heading
MIN_SCORE = 0.3         # Minimum cosine similarity score for a section to be considered relevant
TOP_N = 5               # Number of top relevant sections to include in the final output

# --- DOCKER-SPECIFIC PATHS (DO NOT CHANGE THESE FOR CONTEST) ---
# These paths are fixed to align with the Docker volume mounts.
# Input files (input.json and PDFs) are expected in /app/input
PDF_BASE_DIR = "/app/input"
# The NLP model is expected within the container at this path
NLP_MODEL_PATH = "./models/all-MiniLM-L6-v2"

# Regex to detect common bullet points or list indicators
BULLET_PAT = re.compile(r"^\s*[\-•–—*·]\s*|^[a-zA-Z]\)\s*|^\d+\.\s*|^\([a-zA-Z0-9]+\)\s*")
# Regex to detect common page number patterns or revision footers
PAGE_NUM_PAT = re.compile(r"^\s*(Page\s+\d+of\s+\d+|\d+\s*\|\s*Page\s*\d+|[ivxIVXLCDM]+\s*|[0-9]{1,4})\s*$", re.IGNORECASE)
REV_FOOTER_PAT = re.compile(r"^(revision|document id|confidential)\s*[:\d\s\.]*$", re.IGNORECASE)

# --- Helper Functions ---

def clean_heading_text(text):
    """Removes leading numbers (like '1.1'), extra whitespace, and leading symbols from a heading."""
    text = re.sub(r"^\s*(\d+(\.\d+)*\s*|[•-]\s*)*", "", text).strip()
    return re.sub(r"\s+", " ", text).strip()

def infer_numbered_level(text):
    """Infers heading level based on numerical prefixes (e.g., 1., 1.1., 1.1.1.)."""
    match = re.match(r"^\s*(\d+(\.\d+)*)", text)
    if match:
        parts = match.group(1).split('.')
        num_parts = len(parts)
        if num_parts == 1: return "H1"
        if num_parts == 2: return "H2"
        if num_parts >= 3: return "H3"
    return None

def is_valid_line_for_heading(f):
    """
    Checks if a line's features make it a valid candidate for a heading,
    filtering out common non-heading elements like list items, page numbers, etc.
    """
    text = f["text"].strip()
    # Basic filters
    if not text: return False
    if len(text) < 3 and not f["has_heading_keywords"]: return False # Very short lines without keywords
    if f["word_count"] > MAX_HEADING_WORDS: return False
    if not any(c.isalpha() for c in text): return False # No alphabetic characters

    # Filter out common non-heading patterns
    if BULLET_PAT.match(text): return False # Bullet points
    if PAGE_NUM_PAT.match(text): return False # Page numbers
    if REV_FOOTER_PAT.match(text): return False # Revision footers

    # Filter out lines that look like a table of contents entry but might be misclassified
    # Example: ". . . . . . . . 10"
    if " . . . " in text and not f["is_bold"]: return False

    # Filter out lines that are just numbers (e.g., in a list)
    if re.fullmatch(r"^\s*\d+\s*$", text): return False

    return True

def extract_text_segments(doc):
    """
    Extracts text segments from a PDF document with detailed features.
    Includes text, font size, font name, bounding box, page number, and bold status.
    """
    segments = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block["lines"]:
                    # Heuristic to combine spans that might be broken by font changes but are part of one logical line
                    combined_text = ""
                    font_size = None
                    font_name = None
                    # Use bbox of the first span, extend it if multiple spans are merged
                    line_bbox = list(line["spans"][0]["bbox"])
                    is_bold = False

                    for span in line["spans"]:
                        combined_text += span["text"]
                        if font_size is None:
                            font_size = span["size"]
                            font_name = span["font"]
                        
                        # Update line_bbox to encompass all spans
                        line_bbox[0] = min(line_bbox[0], span["bbox"][0])
                        line_bbox[1] = min(line_bbox[1], span["bbox"][1])
                        line_bbox[2] = max(line_bbox[2], span["bbox"][2])
                        line_bbox[3] = max(line_bbox[3], span["bbox"][3])

                        if 'bold' in span["font"].lower() or '-bd' in span["font"].lower() or 'black' in span["font"].lower():
                            is_bold = True

                    # Calculate relative position on page (0 to 1)
                    page_height = page.rect.height
                    rel_pos = line_bbox[1] / page_height if page_height else 0

                    segments.append({
                        "text": combined_text.strip(),
                        "font_size": font_size,
                        "font_name": font_name,
                        "bbox": line_bbox,
                        "page_num": page_num,
                        "is_bold": is_bold,
                        "rel_pos": rel_pos
                    })
    return segments

def extract_features(text, font_size, font_name, bbox, page_num): # Removed next_text as it's not used in current logic
    """Extracts various features from a text segment."""
    features = {
        "text": text,
        "font_size": font_size,
        "font_name": font_name,
        "bbox": bbox,
        "page_num": page_num,
        "is_bold": 'bold' in font_name.lower() or '-bd' in font_name.lower() or 'black' in font_name.lower(),
        "is_italic": 'italic' in font_name.lower() or '-it' in font_name.lower(),
        "starts_with_numbering": bool(re.match(r"^\s*(\d+(\.\d+)*)\s+", text)),
        "has_heading_keywords": any(kw in text.lower() for kw in HEADING_KEYWORDS),
        "ends_with_punctuation": text.strip().endswith(('.', '!', '?', ':', ';')),
        "word_count": len(text.split()),
        "line_width": bbox[2] - bbox[0],
        "is_centered": False, # More complex to determine accurately without full page context
        "rel_pos": bbox[1] / fitz.open().pages[0].rect.height if fitz.open().page_count > 0 else 0, # Placeholder, will be accurate from segment
    }
    return features

def assign_heading_level(features_list):
    """
    Analyzes text features to identify potential headings and assign H1, H2, H3 levels.
    Employs heuristics based on font size, bolding, and content.
    """
    headings = []
    bold_sizes = defaultdict(int)
    for f in features_list:
        if f["is_bold"] and f["font_size"]:
            bold_sizes[f["font_size"]] += 1

    # Sort bold sizes by frequency and then by size (largest first)
    sorted_bold_sizes = sorted(bold_sizes.keys(), key=lambda x: (bold_sizes[x], x), reverse=True)

    # Assign rough heading levels based on prominent bold font sizes
    h_sizes = {}
    if sorted_bold_sizes:
        h_sizes["H1"] = sorted_bold_sizes[0]
        if len(sorted_bold_sizes) > 1:
            h_sizes["H2"] = sorted_bold_sizes[1]
        if len(sorted_bold_sizes) > 2:
            h_sizes["H3"] = sorted_bold_sizes[2]

    # Heuristic for detecting potential headers/footers (based on frequent lines at top/bottom)
    common_top_lines = defaultdict(int)
    common_bottom_lines = defaultdict(int)
    for f in features_list:
        if f["page_num"] < 3: # Check early pages for consistent headers/footers
            if f["rel_pos"] < 0.15: # Top 15% of the page
                common_top_lines[f["text"].strip()] += 1
            elif f["rel_pos"] > 0.85: # Bottom 15% of the page
                common_bottom_lines[f["text"].strip()] += 1
    
    # Identify lines that appear frequently across pages near top/bottom
    frequent_header_footer_candidates = set()
    for text, count in common_top_lines.items():
        if count > 1: # Appears on more than one early page
            frequent_header_footer_candidates.add(text)
    for text, count in common_bottom_lines.items():
        if count > 1:
            frequent_header_footer_candidates.add(text)


    for i, f in enumerate(features_list):
        if not is_valid_line_for_heading(f):
            continue
        
        # Skip lines that are identified as frequent headers/footers
        if f["text"].strip() in frequent_header_footer_candidates:
            continue

        text_level = None
        
        # 1. Check for numbered headings (strongest indicator)
        numbered_level = infer_numbered_level(f["text"])
        if numbered_level:
            text_level = numbered_level
        # 2. Check for bold size matches
        elif f["is_bold"]:
            if "H1" in h_sizes and abs(f["font_size"] - h_sizes["H1"]) < 1.0:
                text_level = "H1"
            elif "H2" in h_sizes and abs(f["font_size"] - h_sizes["H2"]) < 1.0:
                text_level = "H2"
            elif "H3" in h_sizes and abs(f["font_size"] - h_sizes["H3"]) < 1.0:
                text_level = "H3"
            elif f["font_size"] > 16: # Catch large bold text not in top 3 sizes
                 text_level = "H1"
            elif f["font_size"] > 12: # Catch medium bold text not in top 3 sizes
                 text_level = "H2"
        # 3. Check for heading keywords (even if not bold or numbered)
        elif f["has_heading_keywords"] and f["word_count"] <= MAX_HEADING_WORDS:
            # If it's a keyword-based heading but not bold, assign H3 or lowest if other candidates are strong
            if not text_level:
                text_level = "H3" # Default to H3 for non-bold keyword headings

        if text_level:
            headings.append({
                "text": f["text"],
                "font_size": f["font_size"],
                "level": text_level,
                "page_num": f["page_num"],
                "y0": f["bbox"][1], # Y-coordinate of the top of the bbox
                "original_index": i # Store original index for content extraction
            })
    
    # Sort headings by page_num and y0 for sequential processing
    headings.sort(key=lambda x: (x["page_num"], x["y0"]))
    return headings

def merge_adjacent_headings(headings, threshold=15):
    """
    Merges adjacent lines that are part of the same logical heading.
    """
    if not headings:
        return []

    merged_headings = []
    current_merged = None

    for i, h in enumerate(headings):
        if current_merged is None:
            current_merged = {
                "text": h["text"],
                "level": h["level"],
                "page_num": h["page_num"],
                "y0": h["y0"],
                "original_indices": [h["original_index"]] # Track original indices of all merged parts
            }
        else:
            # Check if current heading is same level, same page, and vertically close to the previous one
            if (h["level"] == current_merged["level"] and
                h["page_num"] == current_merged["page_num"] and
                (h["y0"] - current_merged["y0"] < threshold or h["y0"] - current_merged["y0"] < h["font_size"] * 1.5)): # within 1.5 lines height
                
                current_merged["text"] += " " + h["text"]
                current_merged["y0"] = h["y0"] # Update y0 to the last line's y0 for correct sorting
                current_merged["original_indices"].append(h["original_index"])
            else:
                merged_headings.append(current_merged)
                current_merged = {
                    "text": h["text"],
                    "level": h["level"],
                    "page_num": h["page_num"],
                    "y0": h["y0"],
                    "original_indices": [h["original_index"]]
                }
    if current_merged:
        merged_headings.append(current_merged)

    # Clean text of merged headings and apply final filters
    final_outline = []
    for mh in merged_headings:
        cleaned_text = clean_heading_text(mh["text"])
        if cleaned_text and len(cleaned_text.split()) >= 2: # Min 2 words for a valid heading after cleaning
             final_outline.append({
                "text": cleaned_text,
                "level": mh["level"],
                "page_num": mh["page_num"],
                "y0": mh["y0"],
                "original_index": min(mh["original_indices"]) # Use the index of the first part of the merged heading
            })
    return final_outline

def extract_sections(segments, outline_items):
    """
    Extracts content sections based on identified headings.
    Ensures that content belongs to the correct heading by using original segment indices.
    """
    sections = []
    
    # Store tuples of (original_index, segment_data) for precise mapping
    # The 'segments' list from extract_text_segments is already in order.
    
    # Ensure outline_items are sorted correctly by page and then by original_index
    outline_items.sort(key=lambda x: (x["page_num"], x["original_index"]))

    # Prepare start indices for each section based on where its heading begins in the full segment list
    section_start_indices = []
    for item in outline_items:
        section_start_indices.append(item["original_index"])
    section_start_indices.append(len(segments)) # Sentinel for the end of the last section

    for i, heading in enumerate(outline_items):
        start_idx = heading["original_index"]
        end_idx = section_start_indices[i + 1] # Content goes until the next heading or end of document

        content_lines = []
        for j in range(start_idx, end_idx):
            segment = segments[j]
            # Skip the heading itself, and potential header/footer lines within content if they are highly repetitive
            if segments[j]["text"].strip() == heading["text"].strip() and \
               segments[j]["page_num"] == heading["page_num"] and \
               abs(segments[j]["bbox"][1] - heading["y0"]) < 5: # Close match to heading position
               continue

            # Heuristic to filter out likely headers/footers (top/bottom of pages, very short, repetitive)
            if (segments[j]["rel_pos"] < 0.1 or segments[j]["rel_pos"] > 0.9) and \
               len(segments[j]["text"].split()) < 6 and \
               (PAGE_NUM_PAT.match(segments[j]["text"]) or REV_FOOTER_PAT.match(segments[j]["text"])):
               continue

            content_lines.append(segment["text"])
        
        full_content = "\n".join(content_lines).strip()

        sections.append({
            "document": heading["document_filename"],
            "section_title": heading["text"],
            "page_number": heading["page_num"],
            "level": heading["level"],
            "full_content": full_content
        })
    
    return sections


# --- Main Processing Logic ---

def process_document(pdf_path, nlp_model, persona_job_embedding):
    """Processes a single PDF document to extract and rank sections."""
    doc_filename = os.path.basename(pdf_path)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    all_segments = extract_text_segments(doc)

    # Feature engineering for all segments
    features_list = []
    for i, segment in enumerate(all_segments):
        features_list.append(extract_features( # Removed next_text here
            segment["text"], segment["font_size"], segment["font_name"], 
            segment["bbox"], segment["page_num"] # Removed next_text here
        ))

    # Identify and merge headings
    identified_headings = assign_heading_level(features_list)
    merged_outline = merge_adjacent_headings(identified_headings)
    
    # Add document filename to outline items for later use
    for item in merged_outline:
        item["document_filename"] = doc_filename

    # Extract content sections based on the merged outline
    all_sections_with_content = extract_sections(all_segments, merged_outline)
    
    # Filter and rank sections based on relevance
    ranked_sections = []
    if all_sections_with_content:
        # Prepare text for embedding and filter out empty content
        sections_to_embed = []
        valid_sections = []
        for section in all_sections_with_content:
            text_to_embed = f"{section['section_title']}. {section['full_content']}"
            if len(text_to_embed) > 50: # Ensure content is substantial enough to embed meaningfully
                sections_to_embed.append(text_to_embed)
                valid_sections.append(section)
        
        if sections_to_embed:
            section_embeddings = nlp_model.encode(sections_to_embed, convert_to_tensor=True)
            
            # Calculate cosine similarity with the persona_job_embedding
            # Ensure persona_job_embedding is 2D if it's not already (e.g., [1, embedding_dim])
            if persona_job_embedding.ndim == 1:
                persona_job_embedding = persona_job_embedding.unsqueeze(0)

            similarities = util.cos_sim(persona_job_embedding, section_embeddings)[0] # Get the 1D tensor of similarities

            for i, section in enumerate(valid_sections):
                section["score"] = similarities[i].item() # Convert tensor scalar to Python float
            
            # Filter by MIN_SCORE and then sort
            ranked_sections = [s for s in valid_sections if s["score"] >= MIN_SCORE]
            ranked_sections.sort(key=lambda x: x["score"], reverse=True)
            
            # Add rank to the top_n sections
            for i, section in enumerate(ranked_sections[:TOP_N]):
                section["importance_rank"] = i + 1

    doc.close()
    return ranked_sections[:TOP_N] # Return only the top N sections

def main():
    # In this Dockerized setup, input.json is hardcoded to be within the /app/input volume.
    # The script no longer takes a command-line argument for the input JSON path.
    input_json_path = os.path.join(PDF_BASE_DIR, "input.json")

    if not os.path.exists(input_json_path):
        print(f"Error: Input JSON file not found at {input_json_path}")
        print("Please ensure your 'input.json' is placed in the root of the input directory mounted to /app/input.")
        sys.exit(1)

    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Load NLP model
    try:
        nlp_model = SentenceTransformer(NLP_MODEL_PATH)
        print(f"Loaded NLP model from: {NLP_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading NLP model from {NLP_MODEL_PATH}. Please ensure it's downloaded and correctly placed. Error: {e}")
        sys.exit(1)

    # Prepare persona and job to be done for embedding
    persona_text = input_data.get("persona", {}).get("role", "")
    job_text = input_data.get("job_to_be_done", {}).get("task", "")
    persona_job_text = f"{persona_text} {job_text}".strip()

    if not persona_job_text:
        print("Warning: No persona or job_to_be_done specified in input.json. Relevance ranking may be less effective.")
        persona_job_embedding = nlp_model.encode("general information extraction", convert_to_tensor=True)
    else:
        persona_job_embedding = nlp_model.encode(persona_job_text, convert_to_tensor=True)

    all_extracted_sections = []
    input_document_names = []

    for doc_info in input_data.get("documents", []):
        filename = doc_info.get("filename")
        if not filename:
            print("Warning: Document entry without 'filename' found in input.json. Skipping.")
            continue
        
        pdf_path = os.path.join(PDF_BASE_DIR, filename)
        input_document_names.append(filename)

        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}. Skipping.")
            continue

        print(f"Processing {filename}...")
        sections_from_doc = process_document(pdf_path, nlp_model, persona_job_embedding)
        all_extracted_sections.extend(sections_from_doc)

    # Final sorting by relevance rank across all documents
    all_extracted_sections.sort(key=lambda x: x["score"], reverse=True)

    # Select the overall top_n sections
    final_top_sections = all_extracted_sections[:TOP_N]

    # Prepare output structure
    extracted_sections_summary = []
    subsection_analysis_details = []

    for i, section in enumerate(final_top_sections):
        extracted_sections_summary.append({
            "document": section["document"],
            "page_number": section["page_num"],
            "section_title": section["section_title"],
            "importance_rank": i + 1 # Re-rank for the final overall top sections
        })
        subsection_analysis_details.append({
            "document": section["document"],
            "page_number": section["page_num"],
            "refined_text": section["full_content"]
        })

    final_output = {
        "metadata": {
            "input_documents": input_document_names,
            "persona": persona_text,
            "job_to_be_done": job_text,
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "extracted_sections": extracted_sections_summary,
        "subsection_analysis": subsection_analysis_details
    }

    # Output path modification: Write output to the /app/output volume
    # The output file will always be named 'input_output.json' for consistency.
    output_filename = "input_output.json"
    output_path = os.path.join("/app/output", output_filename) # Assumes /app/output is mounted

    # Ensure the output directory exists
    os.makedirs("/app/output", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(final_output, out, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing complete. Final output saved to: {output_path}")

if __name__ == "__main__":
    main()