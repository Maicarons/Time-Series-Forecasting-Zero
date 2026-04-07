#!/bin/bash
# Release helper script for Time-Series-Forecasting-Zero
# Usage: ./scripts/release.sh [version]

set -e

VERSION=${1:-""}

if [ -z "$VERSION" ]; then
    echo "Error: Version number required"
    echo "Usage: ./scripts/release.sh 0.1.0"
    exit 1
fi

echo "=========================================="
echo "Release Helper - Version $VERSION"
echo "=========================================="

# Step 1: Check current branch
echo ""
echo "[1/7] Checking current branch..."
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    echo "Warning: You are on branch '$CURRENT_BRANCH'"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Run tests
echo ""
echo "[2/7] Running tests..."
python verify_installation.py
pytest tests/ -v || { echo "Tests failed!"; exit 1; }

# Step 3: Update version in setup.py
echo ""
echo "[3/7] Updating version in setup.py..."
sed -i.bak "s/version=\"[^\"]*\"/version=\"$VERSION\"/" setup.py
rm setup.py.bak

# Step 4: Update version in __init__.py
echo ""
echo "[4/7] Updating version in __init__.py..."
sed -i.bak "s/__version__ = \"[^\"]*\"/__version__ = \"$VERSION\"/" src/time_series_forecasting_zero/__init__.py
rm src/time_series_forecasting_zero/__init__.py.bak

# Step 5: Commit changes
echo ""
echo "[5/7] Committing version changes..."
git add setup.py src/time_series_forecasting_zero/__init__.py
git commit -m "Bump version to $VERSION"

# Step 6: Create tag
echo ""
echo "[6/7] Creating git tag v$VERSION..."
git tag -a "v$VERSION" -m "Release version $VERSION"

# Step 7: Push
echo ""
echo "[7/7] Pushing to remote..."
echo "This will trigger CI/CD pipeline to:"
echo "  - Run tests on multiple platforms"
echo "  - Build distribution packages"
echo "  - Publish to PyPI"
echo "  - Create GitHub Release"
echo ""
read -p "Continue? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin HEAD
    git push origin "v$VERSION"
    echo ""
    echo "✅ Release process started!"
    echo "Check progress at: https://github.com/$(git config --get remote.origin.url | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/actions"
else
    echo "Cancelled. Tag and commit are local only."
    echo "To push manually:"
    echo "  git push origin HEAD"
    echo "  git push origin v$VERSION"
fi

echo ""
echo "=========================================="
echo "Release preparation complete!"
echo "=========================================="
