#!/bin/bash

# Step 0: Clean up
rm -rf dist

# Step 1: Change the package name to "llava-torch"
sed -i 's/name = "llava"/name = "llava-torch"/' pyproject.toml

# Step 2: Build the package
python -m build

# Step 3: Revert the changes in pyproject.toml to the original
sed -i 's/name = "llava-torch"/name = "llava"/' pyproject.toml

# Step 4: Upload to PyPI
python -m twine upload dist/*
