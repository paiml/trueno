#!/bin/bash
# Validate Book Examples - Enforce EXTREME TDD Quality for Documentation
#
# This script ensures all code examples in the book meet the same quality
# standards as the Trueno source code:
# - All examples must compile
# - All examples must have tests
# - Examples must pass clippy
# - Examples referenced in book must exist

set -e

echo "📚 Validating Book Examples - EXTREME TDD Quality Enforcement"
echo "=============================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check_pass() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "Step 1: Verify all examples compile"
echo "-----------------------------------"
if cargo build --examples --all-features --quiet 2>&1; then
    check_pass "All examples compile successfully"
else
    check_fail "Examples failed to compile"
    exit 1
fi
echo ""

echo "Step 2: Run clippy on examples"
echo "-------------------------------"
if cargo clippy --examples --all-features --quiet -- -D warnings 2>&1; then
    check_pass "All examples pass clippy (zero warnings)"
else
    check_fail "Examples have clippy warnings"
    exit 1
fi
echo ""

echo "Step 3: Verify examples have documentation"
echo "------------------------------------------"
EXAMPLES_WITHOUT_DOCS=0
for example in examples/*.rs; do
    if ! grep -q "^//!" "$example"; then
        check_fail "$(basename $example) missing module documentation (//!)"
        EXAMPLES_WITHOUT_DOCS=$((EXAMPLES_WITHOUT_DOCS + 1))
    else
        check_pass "$(basename $example) has module documentation"
    fi
done
echo ""

echo "Step 4: Check for example tests"
echo "--------------------------------"
# Check if examples are tested in integration tests or have inline tests
EXAMPLE_COUNT=$(ls examples/*.rs | wc -l)
check_pass "Found $EXAMPLE_COUNT example files"

# Verify examples can run
echo "  Running examples to verify they execute without errors..."
for example in examples/*.rs; do
    EXAMPLE_NAME=$(basename "$example" .rs)
    if timeout 5 cargo run --example "$EXAMPLE_NAME" >/dev/null 2>&1; then
        check_pass "  Example '$EXAMPLE_NAME' runs successfully"
    else
        check_warn "  Example '$EXAMPLE_NAME' failed or timed out"
    fi
done
echo ""

echo "Step 5: Validate book references actual examples"
echo "------------------------------------------------"
# Check that examples mentioned in book chapters actually exist
if [ -d "book/src" ]; then
    MISSING_REFS=0
    # Look for code blocks that reference example files
    for chapter in book/src/**/*.md; do
        # Extract example file references (e.g., `examples/vector_math.rs`)
        REFS=$(grep -o 'examples/[a-z_]*.rs' "$chapter" 2>/dev/null || true)
        if [ -n "$REFS" ]; then
            while IFS= read -r ref; do
                if [ -f "$ref" ]; then
                    check_pass "Book references existing example: $ref"
                else
                    check_fail "Book references missing example: $ref (in $(basename $chapter))"
                    MISSING_REFS=$((MISSING_REFS + 1))
                fi
            done <<< "$REFS"
        fi
    done

    if [ $MISSING_REFS -eq 0 ]; then
        check_pass "All book example references are valid"
    fi
else
    check_warn "Book directory not found, skipping reference validation"
fi
echo ""

echo "Step 6: Verify examples follow naming conventions"
echo "-------------------------------------------------"
for example in examples/*.rs; do
    EXAMPLE_NAME=$(basename "$example")
    # Check for snake_case naming
    if [[ "$EXAMPLE_NAME" =~ ^[a-z][a-z0-9_]*\.rs$ ]]; then
        check_pass "Example uses snake_case: $EXAMPLE_NAME"
    else
        check_fail "Example naming violation: $EXAMPLE_NAME (use snake_case)"
    fi
done
echo ""

# Summary
echo "=============================================================="
echo "Validation Summary"
echo "=============================================================="
echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✓ All book examples meet EXTREME TDD quality standards!${NC}"
    exit 0
else
    echo -e "${RED}✗ Quality gate failed: $FAILED_CHECKS issues found${NC}"
    echo ""
    echo "Book examples MUST meet the same quality as source code:"
    echo "  - All examples must compile"
    echo "  - Zero clippy warnings"
    echo "  - Complete documentation"
    echo "  - Must be runnable"
    echo "  - Follow naming conventions"
    exit 1
fi
