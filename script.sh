#!/bin/bash

# Iterate through all files in the "dna" directory
for file in dna/*; do
    # Get the name of the file without the file extension
    filename=$(basename "$file" .txt)
    # Check if the file named "0.txt" exists in the same directory as the current file
    if [ -f dna/"$filename"/0.txt ]; then
        # Check if the directory already exists, if not create it
        if [ ! -d dna/"$filename" ]; then
            mkdir dna/"$filename"
        fi
        # Move the current file to the directory with the same name as the file containing "0.txt"
        mv "$file" dna/"$filename"/
    fi
done