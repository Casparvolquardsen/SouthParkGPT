#!/bin/bash

total_scripts=0
total_lines=0
total_words=0
total_characters=0

for file in all_scripts/*; do
	# calculate the number of scripts, lines, and words in each script
	echo "Processing $file"

	file_lines=$(wc -l <$file)
	file_words=$(wc -w <$file)
	file_characters=$(wc -m <$file)

	((total_scripts += 1))
	((total_lines += file_lines))
	((total_words += file_words))
	((total_characters += file_characters))

done

echo "total scripts: $total_scripts"
echo "total_lines: $total_lines"
echo "total_words: $total_words"
echo "total_characters: $total_characters"
