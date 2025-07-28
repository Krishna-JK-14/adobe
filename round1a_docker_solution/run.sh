#!/bin/bash

# Ensure output directory exists inside the container
mkdir -p /app/output

echo "Starting Round 1A: PDF Outline Extraction..."

# Loop through each PDF file in the /app/input directory
for pdf_file in /app/input/*.pdf; do
    if [ -f "$pdf_file" ]; then # Check if it's a regular file
        echo "Processing PDF: $pdf_file"
        # Call the main Python script (renamed from pdf_extractor[1].py)
        # Pass the input PDF path and the output directory
        python /app/pdf_outline_extractor.py "$pdf_file" "/app/output"
        if [ $? -ne 0 ]; then
            echo "Error: PDF processing failed for $pdf_file"
            exit 1 # Exit with an error code if the script fails
        fi
    fi
done

echo "Round 1A: Outline Extraction Complete."