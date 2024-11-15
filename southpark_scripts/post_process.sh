#!/bin/bash

# The scipts always start with two lines of episode name and blank line also end with two lines of blank line and episode name
# This script will remove the first two lines and last two lines of each episode
for file in all_scripts/*; do
  sed '1,2d;$d' "$file" > tempfile && mv tempfile "$file"
done
