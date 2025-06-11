#!/bin/bash

# credit to https://github.com/orgs/community/discussions/155944#discussioncomment-12923686
NB_PATH="supervised_ML_identify_author.ipynb"

cp "$NB_PATH" "$NB_PATH.bak"


# Use jq to add empty state key and save to a temporary file
jq '.metadata.widgets."application/vnd.jupyter.widget-state+json" += {"state": {}}' "$NB_PATH" > "$NB_PATH.tmp"

# Replace the original file with the fixed one
mv "$NB_PATH.tmp" "$NB_PATH" 
