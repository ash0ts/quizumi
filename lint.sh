#!/bin/bash

echo "🔍 Running Python linters and formatters..."

# Run Ruff for fast linting and fixing
echo "\n📝 Running Ruff..."
ruff check . --fix

# Run Ruff format (replacement for Black)
echo "\n✨ Running Ruff formatter..."
ruff format .

# Run mypy for type checking
echo "\n🔎 Running mypy..."
mypy .

# Run isort for import sorting (if you prefer it over ruff's import sorting)
# echo "\n📋 Running isort..."
# isort .

# Run pylint for additional checks
echo "\n🔍 Running pylint..."
pylint **/*.py

# Exit with status code
if [ $? -eq 0 ]; then
    echo "\n✅ All checks passed!"
    exit 0
else
    echo "\n❌ Some checks failed. Please fix the issues above."
    exit 1
fi