#!/bin/bash
set -e

# Checkout master or main branch
git checkout -b main 2>/dev/null || git checkout main 2>/dev/null || git checkout -b master 2>/dev/null || git checkout master

# List all untracked and modified files, ignoring gitignored files
# git ls-files --others --exclude-standard --modified
git add -A -N # add intent-to-add for all files so they show up in status

git status --porcelain | awk '{print $2}' | while read -r file; do
    if [ -f "$file" ]; then
        echo "Adding and committing: $file"
        git add "$file"
        git commit -m "Add $file"
    fi
done

echo "Pushing to GitHub..."
git push -u origin HEAD
