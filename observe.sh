#!/bin/bash

# File: observe.sh
# Usage: ./observe.sh /path/to/directory [interval]

TARGET_DIR="$1"
INTERVAL="${2:-5}"  # Default to 5 seconds

if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory> [check_interval_seconds]"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Monitoring: $TARGET_DIR (recursively)"
echo "Check interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo "----------------------------------------"

# Get initial file list
previous_files=$(find "$TARGET_DIR" -type f 2>/dev/null | sort)

while true; do
    # Get current file list
    current_files=$(find "$TARGET_DIR" -type f 2>/dev/null | sort)
    
    # Find new files
    new_files=$(comm -13 <(echo "$previous_files") <(echo "$current_files"))
    
    if [ -n "$new_files" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] New files detected:"
        while IFS= read -r file; do
            [ -n "$file" ] && echo "  $file"
        done <<< "$new_files"
        echo "----------------------------------------"
    fi
    
    previous_files="$current_files"
    sleep "$INTERVAL"
done