# Adobe India Hackathon - Round 1A: Structured Outline Extraction

This repository contains the solution for Round 1A of the Adobe India Hackathon, focusing on extracting a structured hierarchical outline (Title, H1, H2, H3, H4) from PDF documents.

## Objective

The goal of this component is to process a given PDF file (up to 50 pages) and output a clean, hierarchical outline in a specified JSON format, including the document title, heading text, and its corresponding page number and level.

## Solution Approach

This solution employs a robust, multi-layered heuristic approach designed for accuracy, efficiency, and strict adherence to the hackathon's offline and CPU-only constraints.

### Key Methodologies:

1.  *Robust PDF Parsing & Data Extraction:*
    * Utilizes PyMuPDF for high-speed extraction of text blocks, bounding boxes, font sizes, font names, and other crucial metadata.
    * Handles "Page X of Y" footers to map internal PDF page indices to actual printed page numbers, ensuring correct page references in the output.
    * Filters out non-textual blocks (e.g., images) to prevent processing errors.

2.  *Comprehensive Feature Engineering:*
    * For each text line, a rich set of features is extracted, including:
        * *Textual:* Word count, character count, capitalization, presence of numbering patterns (e.g., "1.", "1.1"), heading keywords (e.g., "Introduction", "Appendix"), and punctuation at the end.
        * *Visual/Font:* Absolute font size, inferred bold/italic status, and crucially, a uniform_font flag (True if all text spans in a line share the same font properties, effectively filtering out inline emphasis).
        * *Positional/Layout:* X/Y coordinates, relative vertical position (rel_pos) for header/footer detection, and horizontal centering.
        * *Document-wide:* Font size ranks (0 for largest, 1 for next, etc., derived from KMeans clustering of all font sizes), and page frequency (page_freq) to identify repetitive boilerplate text.

3.  *Intelligent Title Detection:*
    * The document title is identified by searching for the single line (or merged lines if split) on the first printed page that exhibits the absolute largest font size and has a uniform font, while also passing boilerplate filters. This ensures the most prominent and true title is captured.

4.  *Aggressive Pre-filtering for Headings:*
    * Before detailed classification, text lines are rigorously filtered. Only lines that are uniform_font AND (either is_bold OR starts_with_numbering OR contain HEADING_KEYWORDS) are considered as potential headings.
    * Additional filters remove "Page X of Y" lines, content in header/footer regions (HEADER_FOOTER_REL_POS), and text appearing too frequently (FREQUENT_HEADING_PAGES).

5.  *Hierarchical Heading Classification:*
    * *Numbered Headings (Highest Priority):* Lines matching strict numbering patterns (e.g., "1.", "2.1", "3.1.1") are directly assigned H1, H2, or H3 levels.
    * *Non-Numbered Structural Headings:* For lines without numbering, levels are assigned based on a hierarchy of visual prominence and contextual cues:
        * *H1:* Lines with the top font size rank (rank == 0), that are bold and/or centered.
        * *H2:* Lines with the second font size rank (rank == 1), or those matching significant HEADING_KEYWORDS.
        * *H3/H4:* Lines with lower font size ranks, or those ending with a colon (:) (e.g., "Timeline:"), are considered for H3.
    * *Strict Validation:* The is_valid_line_candidate function acts as a final gate, rejecting lines that are too long (paragraphs), too short, pure digits, dates, or single words (unless numbered).

6.  *Output Refinement:*
    * Headings are sorted first by printed page number, then by vertical position (y0).
    * A merging pass combines text from consecutive lines that are part of the same split heading (e.g., a long heading wrapped onto a new line).

### Models and Libraries Used

* **PyMuPDF (fitz):** For efficient PDF parsing and text extraction.
* **scikit-learn:** Specifically KMeans for clustering font sizes.
* **numpy:** For numerical operations, especially with scikit-learn.
* *Standard Python Libraries:* re (regular expressions), os, sys, json, collections.defaultdict.

No external APIs or pre-trained large language models are used, ensuring full offline compatibility and adherence to the model size constraint (as scikit-learn models are very small).

## How to Build and Run Your Solution (Docker)

This section details the exact commands required to build and run your solution, as specified by the hackathon organizers.

### Prerequisites

* *Docker Desktop* (or Docker Engine) installed and running on your system.
* Your input PDF files (e.g., sample.pdf, file02.pdf, file03.pdf, etc.) placed in the input/ directory of this project.

### Steps

1.  *Navigate to the Project Root:*
    Open your terminal or command prompt and navigate to the round1a_submission/ directory (the one containing Dockerfile, requirements.txt, and run.sh).

    bash
    # Example for Windows Command Prompt
    cd C:\Users\YourUser\Documents\round1a_submission
    
    # Example for PowerShell
    Set-Location C:\Users\YourUser\Documents\round1a_submission
    

2.  *Build the Docker Image:*
    This command builds the Docker image named mysolutionname (you can replace mysolutionname:somerandomidentifier with r1a_extractor:latest or any name you prefer for local testing). The --platform linux/amd64 flag is crucial for compliance with the hackathon's required architecture.

    bash
    # For Command Prompt (use ^ for line continuation)
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    
    # For PowerShell (use ` for line continuation)
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    
    # Or simply on one line for either:
    docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
    
    * *Expected Output:* You will see Docker downloading base images, installing Python packages, and copying your application files. This process takes a few minutes depending on your internet speed and system. You might see a warning about FromPlatformFlagConstDisallowed, which is expected and acceptable for this hackathon's requirements.

3.  *Run the Docker Container:*
    This command runs your solution inside the Docker container.
    * --rm: Automatically removes the container once it exits.
    * -v "$(pwd)/input:/app/input": Mounts your local input/ directory (containing your PDFs) to the container's /app/input.
    * -v "$(pwd)/output:/app/output": Mounts your local output/ directory (where JSONs will be written) to the container's /app/output.
    * --network none: *Crucial.* This isolates the container from the internet, strictly enforcing the offline requirement.

    bash
    # For Command Prompt (use ^ for line continuation, %cd% for current directory)
    docker run --rm ^
      -v "%cd%\input:/app/input" ^
      -v "%cd%\output:/app/output" ^
      --network none ^
      mysolutionname:somerandomidentifier
    
    # For PowerShell (use ` for line continuation, $(Get-Location) for current directory)
    docker run --rm `
      -v "$(Get-Location)/input:/app/input" `
      -v "$(Get-Location)/output:/app/output" `
      --network none `
      mysolutionname:somerandomidentifier
    
    # Or simply on one line for either:
    docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier
    
    * *Expected Output:* The run.sh script will execute. You will see messages like "Processing PDF: /app/input/sample.pdf" and "Round 1A: All PDF Outline Extractions Complete."

### Verification

* After the Docker container finishes, check the output/ directory (within your round1a_submission folder) on your local machine.
* You should find generated JSON files (e.g., sample.json, file02.json, file03.json, etc.) corresponding to your input PDFs. Compare these against your expected outputs to verify accuracy.

```json
# Example of expected output JSON format
{
  "title": "Your Document Title",
  "outline": [
    { "level": "H1", "text": "1. Section One", "page": 1 },
    { "level": "H2", "text": "1.1 Sub-section", "page": 2 },
    { "level": "H3", "text": "1.1.1 Detail", "page": 3 }
  ]
}
