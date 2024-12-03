#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_file> <string1> <string2>"
    exit 1
fi

input_file=$1
output_file=$2
string1=$3
string2=$4

# Count the lines containing each string
count1=$(grep -c "$string1" "$input_file")
count2=$(grep -c "$string2" "$input_file")

echo "Before equalization:"
echo "Lines containing '$string1': $count1"
echo "Lines containing '$string2': $count2"

# Determine the number of lines to delete
if [ "$count1" -gt "$count2" ]; then
    delete_count=$((count1 - count2))
    grep -v "$string1" "$input_file" > temp_file
    grep "$string1" "$input_file" | shuf | tail -n +$((delete_count + 1)) >> temp_file
else
    delete_count=$((count2 - count1))
    grep -v "$string2" "$input_file" > temp_file
    grep "$string2" "$input_file" | shuf | tail -n +$((delete_count + 1)) >> temp_file
fi

# Shuffle the lines randomly and save to output file
shuf temp_file > "$output_file"
rm temp_file

# Count the lines containing each string after equalization
new_count1=$(grep -c "$string1" "$output_file")
new_count2=$(grep -c "$string2" "$output_file")

echo "After equalization:"
echo "Lines containing '$string1': $new_count1"
echo "Lines containing '$string2': $new_count2"