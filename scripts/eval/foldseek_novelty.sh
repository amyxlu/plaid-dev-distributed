#!/bin/bash

# Define paths
STRUCTURE_FOLDER=$1         # Path to folder containing your .pdb files
PDB_DATABASE=$2             # Path to the Foldseek PDB database
OUTPUT_DIR=./tmp_out               # Output directory for results
TMP_DIR=./tmp                  # Temporary directory for intermediate files
CORES=8                    # Number of cores for parallelization

# Create output directories if not exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

# Define function for Foldseek search (runs on each PDB file)
run_foldseek() {
    pdb_file=$1
    PDB_DATABASE=$2
    TMP_DIR=$3
    OUTPUT_DIR=$4

    # Extract the filename without extension
    filename=$(basename -- "$pdb_file")
    filename="${filename%.*}"

    # Run Foldseek search for each file
    foldseek easy-search "$pdb_file" "$PDB_DATABASE" "$TMP_DIR/$filename.m8" "$TMP_DIR/$filename.aln" --format-output target,pident,TMscore

    # Extract top result (closest neighbor) based on TM-score
    closest_hit=$(sort -k3,3gr "$TMP_DIR/$filename.m8" | head -n 1)
    
    # Output the result
    echo "Protein: $filename" >> "$OUTPUT_DIR/results.txt"
    echo "Closest neighbor: $closest_hit" >> "$OUTPUT_DIR/results.txt"
    echo "" >> "$OUTPUT_DIR/results.txt"
}

export -f run_foldseek  # Export function for parallel execution

# Find all .pdb files and run them in parallel
find "$STRUCTURE_FOLDER" -name '*.pdb' | parallel -j "$CORES" run_foldseek {} "$PDB_DATABASE" "$TMP_DIR" "$OUTPUT_DIR"
