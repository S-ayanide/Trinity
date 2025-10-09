#!/usr/bin/env python3
"""
Script to generate submission CSV from classification results.
Format: First line is userid, subsequent lines are filename,captcha (sorted 0-9a-f)
"""

import sys
import os

def generate_submission_csv(input_file, output_file, userid):
    """
    Generate a submission CSV file from classification results.
    
    Args:
        input_file: Path to the input file (e.g., stuff.txt)
        output_file: Path to the output CSV file
        userid: TCD short userid (e.g., mondalsa)
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    # Read and parse the input file
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the line (format: "filename, CAPTCHA" or "filename,CAPTCHA")
            parts = line.split(',')
            if len(parts) >= 2:
                filename = parts[0].strip()
                captcha = parts[1].strip()
                
                # Only include .png files
                if filename.endswith('.png'):
                    entries.append((filename, captcha))
    
    # Sort entries by filename (0-9a-f order - natural hex sort)
    entries.sort(key=lambda x: x[0].lower())
    
    # Write to CSV
    with open(output_file, 'w') as f:
        # First line: userid
        f.write(f"{userid}\n")
        
        # Subsequent lines: filename,captcha (no space after comma)
        for filename, captcha in entries:
            f.write(f"{filename},{captcha}\n")
    
    print(f"✓ Generated submission file: {output_file}")
    print(f"✓ Total entries: {len(entries)}")
    print(f"✓ Format: First line = {userid}, sorted 0-9a-f")

if __name__ == "__main__":
    # Configuration
    USERID = "mondalsa"
    INPUT_FILE = "stuff.txt"
    OUTPUT_FILE = "submission.csv"
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]
    if len(sys.argv) > 3:
        USERID = sys.argv[3]
    
    generate_submission_csv(INPUT_FILE, OUTPUT_FILE, USERID)


