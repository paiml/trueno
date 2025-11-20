#!/bin/bash
# Comprehensive benchmark runner for Trueno vs NumPy vs PyTorch
#
# This script:
# 1. Runs Rust benchmarks with Criterion
# 2. Runs Python benchmarks (NumPy/PyTorch)
# 3. Compares results and generates report
#
# Usage:
#   ./benchmarks/run_all.sh

set -e

echo "================================================================================"
echo "Trueno Comprehensive Benchmark Suite"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import numpy, torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Python dependencies missing. Installing...${NC}"
    pip3 install numpy torch --quiet
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Python dependencies already installed${NC}"
fi
echo ""

# Step 1: Run Rust benchmarks
echo "================================================================================"
echo "Step 1/3: Running Trueno (Rust) Benchmarks"
echo "================================================================================"
echo ""
echo "This will take 5-10 minutes..."
echo ""

cargo bench --all-features --no-fail-fast

echo ""
echo -e "${GREEN}✅ Trueno benchmarks complete${NC}"
echo ""

# Step 2: Run Python benchmarks
echo "================================================================================"
echo "Step 2/3: Running Python (NumPy/PyTorch) Benchmarks"
echo "================================================================================"
echo ""
echo "This will take 2-3 minutes..."
echo ""

python3 benchmarks/python_comparison.py

echo ""
echo -e "${GREEN}✅ Python benchmarks complete${NC}"
echo ""

# Step 3: Compare results
echo "================================================================================"
echo "Step 3/3: Comparing Results"
echo "================================================================================"
echo ""

python3 benchmarks/compare_results.py

echo ""
echo "================================================================================"
echo "🎉 Benchmark Suite Complete!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Markdown report: benchmarks/comparison_report.md"
echo "  - JSON data:       benchmarks/comparison_summary.json"
echo "  - Python results:  benchmarks/python_results.json"
echo "  - Rust results:    target/criterion/"
echo ""
echo "View the report with:"
echo "  cat benchmarks/comparison_report.md"
echo ""
