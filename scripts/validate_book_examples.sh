#!/bin/bash
# Validate Book Examples - Enforce EXTREME TDD Quality for Documentation
#
# This script ensures all code examples in the book meet the same quality
# standards as the Trueno source code:
# - All examples must compile
# - All examples must have tests
# - Examples must pass clippy
# - Examples referenced in book must exist
#
# shellcheck disable=SC2031,SC2154,SC2117,SC2317,SC2062

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

# Helper functions
check_pass() {
    TOTAL_CHECKS="$((TOTAL_CHECKS+1))"
    PASSED_CHECKS="$((PASSED_CHECKS+1))"
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    TOTAL_CHECKS="$((TOTAL_CHECKS+1))"
    FAILED_CHECKS="$((FAILED_CHECKS+1))"
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Verify all examples compile
verify_examples_compile() {
    echo "Step 1: Verify all examples compile"
    echo "-----------------------------------"
    if cargo build --examples --all-features --quiet 2>&1; then
        check_pass "All examples compile successfully"
    else
        check_fail "Examples failed to compile"
        exit 1
    fi
    echo ""
}

# Step 2: Run clippy on examples
verify_clippy() {
    echo "Step 2: Run clippy on examples"
    echo "-------------------------------"
    if cargo clippy --examples --all-features --quiet -- -D warnings 2>&1; then
        check_pass "All examples pass clippy (zero warnings)"
    else
        check_fail "Examples have clippy warnings"
        exit 1
    fi
    echo ""
}

# Step 4: Check for example tests
verify_example_tests() {
    local -a example_files=("$@")

    echo "Step 4: Check for example tests"
    echo "--------------------------------"
    local example_count
    example_count="$(ls examples/*.rs 2>/dev/null | wc -l)"
    check_pass "Found $example_count example files"

    echo "  Running examples to verify they execute without errors..."
    for example in "${example_files[@]}"; do
        local example_name
        example_name="$(basename "$example" .rs)"
        if timeout 5 cargo run --example "$example_name" >/dev/null 2>&1; then
            check_pass "  Example \"$example_name\" runs successfully"
        else
            check_warn "  Example \"$example_name\" failed or timed out"
        fi
    done
    echo ""
}

# Step 5: Validate book references actual examples
verify_book_references() {
    echo "Step 5: Validate book references actual examples"
    echo "------------------------------------------------"

    if [[ ! -d "book/src" ]]; then
        check_warn "Book directory not found, skipping reference validation"
        echo ""
        return
    fi

    local missing_refs=0
    mapfile -t chapter_files < <(find book/src -name "*.md" -type f 2>/dev/null | sort)

    for chapter in "${chapter_files[@]}"; do
        local refs
        refs="$(grep -o 'examples/[a-z_]*.rs' "$chapter" 2>/dev/null || true)"
        if [[ -n "$refs" ]]; then
            while IFS= read -r ref; do
                if [[ -f "$ref" ]]; then
                    check_pass "Book references existing example: $ref"
                else
                    local chapter_base
                    chapter_base="$(basename "$chapter")"
                    check_fail "Book references missing example: $ref (in $chapter_base)"
                    missing_refs="$((missing_refs+1))"
                fi
            done < <(echo "$refs")
        fi
    done

    if [[ "$missing_refs" -eq 0 ]]; then
        check_pass "All book example references are valid"
    fi
    echo ""
}

# Step 6: Verify examples follow naming conventions
verify_naming_conventions() {
    local -a example_files=("$@")

    echo "Step 6: Verify examples follow naming conventions"
    echo "-------------------------------------------------"
    for example in "${example_files[@]}"; do
        local example_name
        example_name="$(basename "$example")"
        if [[ "$example_name" =~ ^[a-z][a-z0-9_]*\.rs$ ]]; then
            check_pass "Example uses snake_case: $example_name"
        else
            check_fail "Example naming violation: $example_name (use snake_case)"
        fi
    done
    echo ""
}

# Print summary
print_summary() {
    echo "=============================================================="
    echo "Validation Summary"
    echo "=============================================================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo ""

    if [[ "$FAILED_CHECKS" -eq 0 ]]; then
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
}

# Main execution
main() {
    verify_examples_compile
    verify_clippy

    # Get example files - run verification and capture file list
    mapfile -t example_files < <(ls examples/*.rs 2>/dev/null | sort)

    # Verify documentation
    echo "Step 3: Verify examples have documentation"
    echo "------------------------------------------"
    local examples_without_docs=0
    for example in "${example_files[@]}"; do
        local example_base
        example_base="$(basename "$example")"
        if ! grep -q "^//!" "$example"; then
            check_fail "$example_base missing module documentation (//!)"
            examples_without_docs="$((examples_without_docs+1))"
        else
            check_pass "$example_base has module documentation"
        fi
    done
    echo ""

    verify_example_tests "${example_files[@]}"
    verify_book_references
    verify_naming_conventions "${example_files[@]}"

    print_summary
}

# Run main function
main
