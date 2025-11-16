#!/bin/bash
# Interactive script to push to GitHub

cd /home/horacio/bindcraft

echo "=== Push to GitHub ==="
echo ""
echo "This script will ask for your GitHub credentials"
echo ""

# Check that we are in the correct directory
if [ ! -f "bindcraft_cli.py" ]; then
    echo "Error: You are not in the correct directory"
    exit 1
fi

# Check status
echo "Repository status:"
git status --short | head -5
echo ""

# Request credentials
read -p "GitHub Username (hperez-ucam): " GIT_USER
GIT_USER=${GIT_USER:-hperez-ucam}

echo ""
echo "For the password, you need a Personal Access Token."
echo "If you don't have one, create it at: https://github.com/settings/tokens"
echo ""
read -s -p "GitHub Personal Access Token: " GIT_TOKEN
echo ""

if [ -z "$GIT_TOKEN" ]; then
    echo "Error: Token required"
    exit 1
fi

# Configure URL with credentials
GIT_URL="https://${GIT_USER}:${GIT_TOKEN}@github.com/hperez-ucam/bindcraft_cli.git"

echo ""
echo "Pushing..."
git push $GIT_URL main 2>&1

# Clean credentials from remote URL
git remote set-url origin https://github.com/hperez-ucam/bindcraft_cli.git

echo ""
echo "✅ Push completed (or check errors above)"
