#!/bin/bash

# move all files from the subdirectories to the `all_scripts` directory
for dir in */; do
    if [ "$dir" != "all_scripts/" ]; then
        cp $dir/* all_scripts/
    fi
done
