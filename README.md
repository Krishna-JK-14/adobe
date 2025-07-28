# Adobe India Hackathon - Connecting the Dots: Intelligent PDF Information Extractor

This project delivers a comprehensive Python solution for the Adobe India Hackathon challenge, transforming static PDFs into intelligent, interactive experiences. It's built to operate entirely offline and on CPU-only systems, adhering to all hackathon constraints.

## Solution Overview

The core functionality is split into two integrated parts:

### Round 1A: Structured Outline Extraction

This component precisely extracts hierarchical outlines (Title, H1-H4) from PDFs. It uses `PyMuPDF` for robust parsing, extensive feature engineering (including font analysis and layout cues), and aggressive filtering to ensure accurate heading detection while eliminating boilerplate and non-structural text. It intelligently handles multi-line titles and merges split headings.

### Round 1B: Persona-Driven Document Intelligence

Building on Round 1A's output, this part analyzes document collections to identify and rank sections most relevant to a specific persona and "job-to-be-done". It leverages a lightweight, on-device NLP embedding model for semantic matching and performs extractive summarization of key subsections.

## Getting Started

The entire solution is designed for Docker deployment. Build and run the single Docker image from the project root to process PDFs and generate results automatically.
