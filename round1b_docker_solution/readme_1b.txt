# PDF Information Extraction Service

## 1. Overview

This project delivers a robust, Dockerized service designed to extract and rank relevant information from PDF documents. It addresses the challenge of identifying key sections within unstructured documents based on a user-defined intent (expressed through a "persona" and "job to be done").

The service's primary functions include:
* Accurate extraction of text and its associated layout features from PDF files.
* Intelligent detection and structuring of document headings (H1, H2, H3).
* Segmentation of document content based on identified headings.
* Semantic relevance ranking of extracted sections using a pre-trained Sentence Transformer model.
* Generation of a standardized JSON output detailing the most relevant sections.

The solution is packaged as a Docker image, ensuring a consistent and isolated execution environment, which operates entirely offline once built.

## 2. Core Logic and Technical Approach

The `extractor.py` script orchestrates the information extraction pipeline:

### 2.1. PDF Text Segmentation and Feature Extraction
* Utilizes `PyMuPDF` (`fitz`) to parse PDF pages into text segments.
* Each segment is enriched with layout features such as font size, font family, bounding box coordinates, page number, bold status, and relative vertical position.

### 2.2. Intelligent Heading Detection
* Employs a heuristic-based approach to identify document headings (H1, H2, H3 levels). Key heuristics include:
  * **Numerical Prefixes:** Strong indicators for structured headings (e.g., "1. Introduction", "2.1 Methodology").
  * **Font Characteristics:** Larger font sizes and bold text are prioritized. The system dynamically identifies prominent bold font sizes to distinguish hierarchy.
  * **Keyword Matching:** Common heading keywords (e.g., "Abstract", "Conclusion", "Results") are used to identify less formal section titles.
  * **Positional Filtering:** Lines frequently appearing in header/footer regions across multiple pages are filtered out.
  * **Content Filtering:** Lines resembling bullet points, page numbers, or short numeric lists are excluded to avoid misclassification.

### 2.3. Heading Merging
* Adjacent lines that logically form a single heading (e.g., a title spanning two lines) are merged based on vertical proximity, consistent styling, and inferred hierarchy level.

### 2.4. Content Sectioning
* Identified headings act as delimiters to segment the document's full text into logical content sections.
* Each section's content is the text immediately following its heading up to the beginning of the next heading or the end of the document.
* Further filtering within content sections removes residual noise like isolated page numbers or repeated headers/footers.

### 2.5. Relevance Ranking with Natural Language Processing (NLP)
* A pre-trained `all-MiniLM-L6-v2` Sentence Transformer model is used for semantic similarity assessment.
* The "persona" and "job to be done" strings from the `input.json` are combined and embedded into a vector representation.
* Each extracted section's title and its full content are also embedded.
* Cosine similarity is computed between the combined persona/job embedding and each section's embedding.
* Sections with a similarity score below a defined threshold (`MIN_SCORE`) are filtered out.
* Remaining sections are ranked by their similarity score, and the top `TOP_N` sections are selected for the final output.

## 3. Project Structure

The project repository's structure is as follows:

```

round1b/
├── Dockerfile                  # Defines the Docker image build process
├── extractor.py                # The main Python application script
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Specifies files/directories to be ignored by Git
├── models/                     # Contains the pre-downloaded Sentence Transformer model
│   └── all-MiniLM-L6-v2/       # Directory containing all model files
│       ├── config.json
│       ├── pytorch\_model.bin
│       ├── README.md
│       ├── special\_tokens\_map.json
│       ├── tokenizer\_config.json
│       ├── tokenizer.json
│       └── vocab.json

````

## 4. Docker Usage Instructions

This project is designed for execution within a Docker container.

### 4.1. Building the Docker Image

To build the Docker image, navigate to the root directory of this project (`round1b/`) and execute the following command:

```bash
docker build --platform linux/amd64 -t <your_image_name> .
````

Replace `<your_image_name>` with a suitable tag (e.g., `my-pdf-processor:latest`).

### 4.2. Running the Docker Container

The container operates by mounting input and output directories as volumes. The `extractor.py` script within the container expects input at `/app/input` and writes output to `/app/output`. The container requires no network access during execution.

Example `docker run` command:

```bash
docker run --rm \
  -v /path/to/host/input_data_directory:/app/input:ro \
  -v /path/to/host/output_results_directory:/app/output \
  --network none \
  <your_image_name>
```

* `/path/to/host/input_data_directory`: Absolute path on the host containing `input.json` and all PDFs.
* `/path/to/host/output_results_directory`: Absolute path on the host where the result `input_output.json` will be saved.
* `<your_image_name>`: The tag name used when building the Docker image.

## 5. Input Format (`input.json`)

The service expects a single `input.json` file in the root of the mounted input directory (`/app/input` inside the container).

**Example:**

```json
{
  "documents": [
    { "filename": "document_A.pdf" },
    { "filename": "report_B.pdf" },
    { "filename": "article_C.pdf" }
  ],
  "persona": {
    "role": "Environmental Scientist"
  },
  "job_to_be_done": {
    "task": "Extract information on climate change impacts on biodiversity from scientific papers."
  }
}
```

## 6. Output Format (`input_output.json`)

The service generates a single JSON file named `input_output.json` in the root of the mounted output directory (`/app/output` inside the container).

**Example:**

```json
{
  "metadata": {
    "input_documents": [
      "document_A.pdf",
      "report_B.pdf",
      "article_C.pdf"
    ],
    "persona": "Environmental Scientist",
    "job_to_be_done": "Extract information on climate change impacts on biodiversity from scientific papers.",
    "processing_timestamp": "2025-07-27T23:52:38+05:30"
  },
  "extracted_sections": [
    {
      "document": "article_C.pdf",
      "page_number": 7,
      "section_title": "Impact on Arctic Ecosystems",
      "importance_rank": 1
    },
    {
      "document": "document_A.pdf",
      "page_number": 23,
      "section_title": "Adaptation Strategies for Marine Life",
      "importance_rank": 2
    },
    {
      "document": "report_B.pdf",
      "page_number": 12,
      "section_title": "Biodiversity Loss in Tropical Rainforests",
      "importance_rank": 3
    },
    {
      "document": "article_C.pdf",
      "page_number": 15,
      "section_title": "Policy Recommendations for Conservation",
      "importance_rank": 4
    },
    {
      "document": "document_A.pdf",
      "page_number": 8,
      "section_title": "Global Temperature Trends",
      "importance_rank": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "article_C.pdf",
      "page_number": 7,
      "refined_text": "This section details the significant changes observed in Arctic ecosystems, particularly concerning the decline in polar bear populations due to melting sea ice and its cascading effects on the food web..."
    },
    {
      "document": "document_A.pdf",
      "page_number": 23,
      "refined_text": "Examining various proposed adaptation strategies, this part focuses on how marine species are responding to ocean acidification and rising sea temperatures, including genetic shifts and migration patterns..."
    },
    {
      "document": "report_B.pdf",
      "page_number": 12,
      "refined_text": "Discussing the accelerated rate of biodiversity loss within tropical rainforests, this analysis highlights the impact of deforestation and altered precipitation patterns on endemic species..."
    },
    {
      "document": "article_C.pdf",
      "page_number": 15,
      "refined_text": "Key policy recommendations include strengthening international agreements on carbon emissions, funding for protected areas, and supporting local communities in sustainable land management..."
    },
    {
      "document": "document_A.pdf",
      "page_number": 8,
      "refined_text": "An overview of global temperature trends over the past century, presenting data from various meteorological stations and satellite observations, indicating a consistent warming pattern..."
    }
  ]
}
```

## 7. Important Notes

* **Offline Execution:** The Docker container is configured to run without any external network access. All dependencies, including the NLP model, are bundled during the image build.
* **Model Specification:** Uses the pre-downloaded `all-MiniLM-L6-v2` Sentence Transformer for semantic relevance scoring.
* **PDF Quality:** The system performs best on digital (non-scanned) PDFs with logical layout and selectable text.
* **Heuristic Approach:** While the heading detection logic is robust and tuned for general documents, unusual formatting or OCR-scanned PDFs may yield imperfect results.

```