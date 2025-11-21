#!/usr/bin/env python3
"""
Pre-commit check for SIMD intrinsics without #[target_feature] attribute.

This prevents commits that add SIMD code without the required compiler attribute,
which causes severe performance degradation (5.9x slower to missing 21x speedup).

Bug instances found: 12 (6 sqrt/recip + 6 logarithms)
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

# SIMD intrinsic patterns by backend
INTRINSIC_PATTERNS = {
    'SSE2': (r'_mm_\w+', 'sse2'),
    'AVX2': (r'_mm256_\w+', 'avx2'),
    'AVX512': (r'_mm512_\w+', 'avx512f'),
    'NEON': (r'v\w+q_f32', 'neon'),
}

def check_file(filepath: Path, backend: str) -> List[Tuple[int, str, str]]:
    """
    Check a backend file for SIMD intrinsics without #[target_feature].

    Returns list of (line_number, function_name, required_attribute) for violations.
    """
    if backend not in INTRINSIC_PATTERNS:
        return []

    pattern, required_feature = INTRINSIC_PATTERNS[backend]
    intrinsic_re = re.compile(pattern)
    unsafe_fn_re = re.compile(r'^\s*unsafe\s+fn\s+(\w+)')
    target_feature_re = re.compile(r'#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]')

    violations = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for unsafe fn declaration
        match = unsafe_fn_re.match(line)
        if match:
            fn_name = match.group(1)
            fn_line = i + 1  # 1-indexed for display

            # Check if previous line has #[target_feature]
            has_target_feature = False
            if i > 0:
                prev_line = lines[i-1]
                if target_feature_re.search(prev_line):
                    has_target_feature = True

            # Scan function body for SIMD intrinsics
            # Simple scan: read until we find a closing brace at the same indentation
            brace_count = 0
            found_intrinsic = False
            j = i

            while j < len(lines):
                fn_line_text = lines[j]

                # Count braces
                brace_count += fn_line_text.count('{')
                brace_count -= fn_line_text.count('}')

                # Check for SIMD intrinsics
                if intrinsic_re.search(fn_line_text):
                    found_intrinsic = True

                # If we've closed all braces, we're done with this function
                if brace_count == 0 and j > i:
                    break

                j += 1

            # If function uses SIMD but missing attribute, report violation
            if found_intrinsic and not has_target_feature:
                violations.append((fn_line, fn_name, required_feature))

            i = j  # Skip to end of function

        i += 1

    return violations

def main():
    print(f"{BLUE}🔍 Checking for SIMD intrinsics without #[target_feature]...{NC}")

    backend_files = {
        'SSE2': Path('src/backends/sse2.rs'),
        'AVX2': Path('src/backends/avx2.rs'),
        'AVX512': Path('src/backends/avx512.rs'),
        'NEON': Path('src/backends/neon.rs'),
    }

    all_violations = []

    for backend, filepath in backend_files.items():
        violations = check_file(filepath, backend)
        for line_num, fn_name, feature in violations:
            all_violations.append((filepath, line_num, fn_name, feature, backend))

    if not all_violations:
        print(f"\n{GREEN}✅ PASS: No SIMD intrinsics without #[target_feature] detected{NC}\n")
        return 0

    # Report violations
    print(f"\n{RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
    print(f"{RED}❌ COMMIT BLOCKED{NC}")
    print(f"{RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}\n")
    print(f"{YELLOW}Found {len(all_violations)} function(s) using SIMD intrinsics without #[target_feature]{NC}\n")

    print(f"{BLUE}Violations:{NC}")
    for filepath, line_num, fn_name, feature, backend in all_violations:
        print(f"  {RED}❌{NC} {filepath}:{line_num} - {YELLOW}{fn_name}{NC}")
        print(f"     Missing: {YELLOW}#[target_feature(enable = \"{feature}\")]{NC}\n")

    print(f"{BLUE}Why this matters:{NC}")
    print("  Without #[target_feature], the Rust compiler CANNOT emit SIMD instructions,")
    print("  even though the code compiles. This causes severe performance degradation:")
    print(f"    • 5.9x slower (recip bug)")
    print(f"    • Missing 21x speedup potential (log10)\n")

    print(f"{BLUE}How to fix:{NC}")
    print("  Add the #[target_feature] attribute ABOVE each unsafe function:\n")
    print(f"  {GREEN}#[target_feature(enable = \"avx2\")]  // ← ADD THIS{NC}")
    print(f"  {GREEN}unsafe fn my_function(a: &[f32]) {{{NC}")
    print(f"  {GREEN}    let vec = _mm256_loadu_ps(...);{NC}")
    print(f"  {GREEN}}}{NC}\n")

    return 1

if __name__ == '__main__':
    sys.exit(main())
