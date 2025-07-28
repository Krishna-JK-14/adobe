import fitz  # PyMuPDF
import sys
import os
import json
import re
from collections import defaultdict, Counter
import numpy as np

def extract_text_segments(doc, num_pages_to_process=None):
    """Extracts text segments with detailed font info, including relative position."""
    segments = []
    num_pages = num_pages_to_process if num_pages_to_process is not None else len(doc)
    num_pages = min(num_pages, len(doc))

    for idx in range(num_pages):
        page = doc[idx]
        page_height = page.rect.height
        
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue

                # Ensure all spans in the line are uniform for core features
                first_span = spans[0]
                text = " ".join(s["text"] for s in spans)
                
                # Check for uniformity
                uniform_font = True
                for s in spans:
                    if s["size"] != first_span["size"] or s["font"] != first_span["font"]:
                        uniform_font = False
                        break

                segments.append({
                    "text": text,
                    "font_size": first_span["size"],
                    "font_name": first_span["font"],
                    "bbox": line["bbox"],
                    "page_number": idx + 1,  # 1-indexed page number
                    "rel_pos": line["bbox"][1] / page_height, # Relative Y position (top of line)
                    "is_bold": 'bold' in first_span["font"].lower() or 'black' in first_span["font"].lower() or '-bd' in first_span["font"].lower(),
                    "uniform_font": uniform_font
                })
    return segments

def analyze_document_fonts(segments):
    """Determines dominant font sizes for potential headings."""
    bold_sizes = defaultdict(int)
    all_sizes = defaultdict(int)

    for seg in segments:
        all_sizes[seg["font_size"]] += 1
        if seg["is_bold"]:
            bold_sizes[seg["font_size"]] += 1
    
    # Get the top N most frequent bold font sizes
    sorted_bold_sizes = sorted(bold_sizes.items(), key=lambda item: item[1], reverse=True)
    
    # Try to identify H1, H2, H3 sizes
    h_sizes = {}
    if sorted_bold_sizes:
        h_sizes['H1'] = sorted_bold_sizes[0][0] # Largest/most frequent bold is H1 candidate

        # Find next distinct bold size for H2
        for size, _ in sorted_bold_sizes:
            if size < h_sizes['H1'] - 1.0: # Check for distinctness (more than 1pt difference)
                h_sizes['H2'] = size
                break
        
        # Find next distinct bold size for H3
        if 'H2' in h_sizes:
            for size, _ in sorted_bold_sizes:
                if size < h_sizes['H2'] - 1.0:
                    h_sizes['H3'] = size
                    break
    
    # Fallback: if not enough distinct bold sizes, use general large fonts
    if 'H2' not in h_sizes and sorted_bold_sizes and len(sorted_bold_sizes) > 1:
        h_sizes['H2'] = sorted_bold_sizes[1][0]
    if 'H3' not in h_sizes and sorted_bold_sizes and len(sorted_bold_sizes) > 2:
        h_sizes['H3'] = sorted_bold_sizes[2][0]

    return h_sizes, dict(all_sizes)

def clean_heading_text(text):
    """Removes numbering and excess spaces from heading text."""
    # Remove common numbering patterns (e.g., "1. ", "1.1 ", "1.1.1 ")
    cleaned_text = re.sub(r'^\d+(\.\d+)*\s*', '', text).strip()
    # Remove trailing page numbers if they exist
    cleaned_text = re.sub(r'\s+\d+$', '', cleaned_text).strip()
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text

def infer_heading_level(segment, h_sizes, page_freq_map, title_text=""):
    """Assigns heading level based on rules."""
    text = segment["text"].strip()
    font_size = segment["font_size"]
    is_bold = segment["is_bold"]
    page_number = segment["page_number"]
    rel_pos = segment["rel_pos"]

    # --- Rule 0: Filter out known non-headings early ---
    # Common boilerplate/page numbers (adjust thresholds as needed)
    HEADER_FOOTER_REL_POS = (0.05, 0.95) # Lines outside this are likely headers/footers
    FREQUENT_HEADING_PAGES = 5 # Appears on more than this many pages, likely boilerplate
    MAX_HEADING_WORDS = 15 # Too many words, likely not a heading

    if not segment["uniform_font"]: # Mixed fonts in one line usually not a heading
        return None
    if not any(c.isalpha() for c in text): # Must contain letters
        return None
    if text.strip().isdigit(): # Pure numbers
        return None
    if len(text.split()) > MAX_HEADING_WORDS: # Too long for a heading
        return None
    if page_freq_map.get(text, 0) > FREQUENT_HEADING_PAGES: # Appears on too many pages
        return None
    if not (HEADER_FOOTER_REL_POS[0] < rel_pos < HEADER_FOOTER_REL_POS[1]): # Outside main body area
        return None
    if text == title_text: # Don't include the main title in the outline
        return None

    # --- Rule 1: Explicit Numbering (Highest Priority) ---
    numbered_level = None
    if re.match(r'^\d+\.\d+\.\d+(\.\d+)*\s+', text): # e.g., 1.1.1, 1.1.1.1
        numbered_level = "H3"
    elif re.match(r'^\d+\.\d+\s+', text): # e.g., 1.1, 2.3
        numbered_level = "H2"
    elif re.match(r'^\d+\.\s+', text): # e.g., 1., 2.
        numbered_level = "H1"
    
    if numbered_level:
        return numbered_level

    # --- Rule 2: Font Size Hierarchy (Primary for non-numbered) ---
    # Compare with a tolerance as font sizes can vary slightly
    font_tolerance = 1.0 

    if 'H1' in h_sizes and abs(font_size - h_sizes['H1']) < font_tolerance and is_bold:
        # Additional checks for H1 candidates not caught by numbering:
        # Must be relatively large and bold.
        # This can catch "Chapter X" or main section titles.
        return "H1"
    
    if 'H2' in h_sizes and abs(font_size - h_sizes['H2']) < font_tolerance and is_bold:
        # Additional checks for H2 candidates
        return "H2"
    
    if 'H3' in h_sizes and abs(font_size - h_sizes['H3']) < font_tolerance and is_bold:
        return "H3"
    
    # --- Rule 3: Keyword-based (Fallback/Confirmation) ---
    HEADING_KEYWORDS = ['introduction', 'chapter', 'overview', 'summary', 'conclusion', 'references', 'appendix', 'preface', 'acknowledgements', 'glossary']
    if any(kw in text.lower() for kw in HEADING_KEYWORDS):
        # Assign level based on likely significance of keyword
        if 'H1' in h_sizes and abs(font_size - h_sizes['H1']) < font_tolerance * 2: # Looser font check for keywords
            return "H1"
        elif 'H2' in h_sizes and abs(font_size - h_sizes['H2']) < font_tolerance * 2:
            return "H2"
        else: # Default keyword-based if font doesn't fit a higher defined level
            return "H2" # Keywords usually indicate at least H2 or H1

    # --- Rule 4: Other Heuristics (Lower Priority) ---
    # "Table of Contents" on early pages should generally be ignored for outline but might be picked up by font size
    if "table of contents" in text.lower() or "revision history" in text.lower():
        # Special handling for TOC entries: only consider if it's explicitly part of a main section,
        # otherwise it's usually just a list of contents.
        if page_number <= 5: # Assuming TOC is typically in first few pages
            return None # Filter out TOC listings themselves

    return None # Not a recognized heading


def get_outline(segments, h_sizes, page_freq_map, title_text):
    """Generates the outline."""
    outline = []
    seen_headings = set() # To avoid duplicates
    
    for segment in segments:
        level = infer_heading_level(segment, h_sizes, page_freq_map, title_text)
        if level:
            cleaned_text = clean_heading_text(segment["text"])
            if not cleaned_text: # Don't add empty headings
                continue

            heading_item = {
                "text": cleaned_text,
                "page": segment["page_number"],
                "level": level,
                "y0": segment["bbox"][1] # Keep y0 for sorting
            }

            # Add to outline, handling potential duplicates (e.g., TOC vs actual heading)
            # Prioritize later occurrences or higher levels if text is same
            key = (cleaned_text.lower(), level)
            if key in seen_headings:
                # If we've seen this text and level, check if current is better (e.g., on a later page)
                # This is a simplification; more complex logic might be needed for perfect deduplication
                pass # For this problem, a simple set-based check after sorting should be sufficient
            else:
                outline.append(heading_item)
                seen_headings.add(key)
    
    # Sort by page number, then vertical position (y0)
    outline.sort(key=lambda x: (x["page"], x["y0"]))

    # Final cleanup and merging for consecutive lines that should be one heading
    final_merged_outline = []
    if not outline:
        return []

    current_heading = None
    merge_threshold_y = 15 # Max vertical distance for lines to be merged

    for item in outline:
        if current_heading is None:
            current_heading = item
        else:
            # Check if current item can be merged with the previous one
            # Same page, same level, and vertically close
            if (item["page"] == current_heading["page"] and
                item["level"] == current_heading["level"] and
                (item["y0"] - current_heading["y0"]) < merge_threshold_y):
                
                current_heading["text"] += " " + item["text"]
                current_heading["y0"] = item["y0"] # Update y0 to the last part for consistent sorting
            else:
                final_merged_outline.append(current_heading)
                current_heading = item
    
    if current_heading: # Add the last heading
        final_merged_outline.append(current_heading)

    # Remove y0, it's no longer needed for output
    for item in final_merged_outline:
        if "y0" in item:
            del item["y0"]

    return final_merged_outline


# --- Main Execution ---
def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf_outline_extractor.py <pdf_path> <output_dir>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    # Process all pages for comprehensive font analysis and page frequency
    all_segments = extract_text_segments(doc, num_pages_to_process=len(doc)) 
    doc.close()

    if not all_segments:
        print("No text segments found in the PDF. Outputting empty outline.")
        output_data = {"title": "", "outline": []}
    else:
        h_sizes, all_font_counts = analyze_document_fonts(all_segments)
        
        # Calculate page frequency map for all text to identify boilerplate
        page_freq_map = defaultdict(int)
        for seg in all_segments:
            page_freq_map[seg["text"].strip()] += 1 # Count appearances of each unique text string


        # Title detection: Robustly find the title based on H1 font size and top of page
        title_candidates = [
            s for s in all_segments
            if s["page_number"] == 1 and s["is_bold"] and s["uniform_font"]
            and 'H1' in h_sizes and abs(s["font_size"] - h_sizes['H1']) < 1.0 # Must be H1-like font
            and s["rel_pos"] < 0.3 # Must be in upper part of page 1
            and not re.search(r'\b(copyright|table of contents|revision history|page)\b', s["text"].lower()) # Avoid common non-titles
        ]
        title = ""
        if title_candidates:
            # Sort by Y position (top-most)
            title_candidates.sort(key=lambda x: x["bbox"][1])
            title = clean_heading_text(title_candidates[0]["text"])

        # Generate the outline using the refined logic
        outline = get_outline(all_segments, h_sizes, page_freq_map, title)
        
        output_data = {"title": title, "outline": outline}

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    output_filepath = os.path.join(output_dir, out_name)

    with open(output_filepath, "w", encoding="utf-8") as out_file:
        json.dump(output_data, out_file, indent=2, ensure_ascii=False)
    
    print(f"Extraction complete. Output saved to: {output_filepath}")

if __name__ == "__main__":
    main()