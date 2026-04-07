## Dataset Building Guide

**Overview**  
This document describes how to build the core X-POSE dataset from the X TSV checkpoint and extract all required note–tweet data used by the later stages.

### Steps to Build the Dataset

1. **Download the TSV checkpoint**
   - Obtain the TSV checkpoint file from X public [files](https://x.com/i/communitynotes/download-data)

2. **Create note classes** (`build_note_classes.py`)
   - Run `build_note_classes.py` to:
     - Process the TSV checkpoint
     - Build the two classes
     - Save the resulting class files

3. **Extract tweets and images** (`tweet_extraction.py`)
   - Configure and run `tweet_extraction.py` separately for each note class to:
     - Fetch the corresponding tweets, using [twikit](https://github.com/d60/twikit) library
     - Download and save the associated images (when available)
   - Ensure the output paths are correctly set so they can be consumed by the next step

4. **Combine notes and tweets into the final dataset** (`combine_notes_tweets.py`)
   - Run `combine_notes_tweets.py` to:
     - Load the processed note classes and extracted tweet data
     - Match notes with their tweets
     - Filter out entries where tweet extraction failed
     - Save the final combined dataset (CSV) that downstream modules (`feature_extraction.py`, `claim_extraction.py`, `query_generation.py` and evidence extraction scripts) will consume
