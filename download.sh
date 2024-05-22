#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p "embeddings"

ENCODERS=("bert" "all-mpnet-base-v2" "simcse" "angle-bert" "angle-llama" "llama-7B" "llama2-7B" "text-embedding-3-small")

# Loop through no split datasets
DATASETSNOSPLIT=("mr" "cr" "subj" "mpqa") 
for DATASET in "${DATASETSNOSPLIT[@]}"; do
    # Loop through encoders
    for ENCODER in "${ENCODERS[@]}"; do
	# Extract filename from URL
    	FILENAME="$ENCODER""_""$DATASET""_embeddings.csv"
        DOMAIN="https://jlu.myweb.cs.uwindsor.ca/embeddings/$DATASET/$FILENAME"
	
        OUTPUT_DIR="embeddings/$DATASET"
        mkdir -p $OUTPUT_DIR

        # Check if the file already exists
        if [ -e "$OUTPUT_DIR/$FILENAME" ]; then
            echo "File $FILENAME already exists. Skipping download."
            continue
        fi
        # Download file
        echo "Downloading $FILENAME from $DOMAIN..."
        wget "$DOMAIN" -P "$OUTPUT_DIR"
        
        # Check if download was successful
        if [ $? -eq 0 ]; then
		    echo "Download of $FILENAME successful."
        else
		    echo "Failed to download $FILENAME from $DOMAIN."
        fi
    done
done

# Loop through train/test splitted datasets
# test part
DATASETSTESTTRAIN=("sstf" "trec")
for DATASET in "${DATASETSTESTTRAIN[@]}"; do
    # Loop through encoders
    for ENCODER in "${ENCODERS[@]}"; do
        FILENAME="$ENCODER""_""$DATASET""_test_embeddings.csv"
        DOMAIN="https://jlu.myweb.cs.uwindsor.ca/embeddings/$DATASET/$FILENAME"
            
        OUTPUT_DIR="embeddings/$DATASET"
        mkdir -p $OUTPUT_DIR

        # Check if the file already exists
        if [ -e "$OUTPUT_DIR/$FILENAME" ]; then
            echo "File $FILENAME already exists. Skipping download."
            continue
        fi
        # Download file
        echo "Downloading $FILENAME from $DOMAIN..."
        wget "$DOMAIN" -P "$OUTPUT_DIR"
            
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "Download of $FILENAME successful."
        else
            echo "Failed to download $FILENAME from $DOMAIN."
        fi
    done
done
# train part
for DATASET in "${DATASETSTESTTRAIN[@]}"; do
    # Loop through encoders
    for ENCODER in "${ENCODERS[@]}"; do
        FILENAME="$ENCODER""_""$DATASET""_train_embeddings.csv"
        DOMAIN="https://jlu.myweb.cs.uwindsor.ca/embeddings/$DATASET/$FILENAME"
            
        OUTPUT_DIR="embeddings/$DATASET"
        mkdir -p $OUTPUT_DIR

        # Check if the file already exists
        if [ -e "$OUTPUT_DIR/$FILENAME" ]; then
            echo "File $FILENAME already exists. Skipping download."
            continue
        fi
        # Download file
        echo "Downloading $FILENAME from $DOMAIN..."
        wget "$DOMAIN" -P "$OUTPUT_DIR"
            
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "Download of $FILENAME successful."
        else
            echo "Failed to download $FILENAME from $DOMAIN."
        fi
    done
done

# Loop through train/test splitted datasets
# test part
DATASETSTESTTRAIN=("sts1" "sts2")
for DATASET in "${DATASETSTESTTRAIN[@]}"; do
    # Loop through encoders
    for ENCODER in "${ENCODERS[@]}"; do
        FILENAME="$ENCODER""_""$DATASET""_test_embeddings.csv"
        DOMAIN="https://jlu.myweb.cs.uwindsor.ca/embeddings/sts/$FILENAME"
            
        OUTPUT_DIR="embeddings/sts"
        mkdir -p $OUTPUT_DIR
        # Check if the file already exists
        if [ -e "$OUTPUT_DIR/$FILENAME" ]; then
            echo "File $FILENAME already exists. Skipping download."
            continue
        fi
        # Download file
        echo "Downloading $FILENAME from $DOMAIN..."
        wget "$DOMAIN" -P "$OUTPUT_DIR"
            
        # Check if download was successful
        if [ $? -eq 0 ]; then
            echo "Download of $FILENAME successful."
        else
            echo "Failed to download $FILENAME from $DOMAIN."
        fi
    done
done