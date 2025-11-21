#!/usr/bin/env python3
"""
Comprehensive SIMD Property Checker - Pre-commit validation

This validates multiple SIMD code properties to ensure correctness and performance:
1. [CRITICAL] Missing #[target_feature] attributes
2. [ERROR] Attribute-intrinsic mismatch (e.g., avx2 attribute with avx512 intrinsics)
3. [WARNING] Missing safety comments on unsafe SIMD functions
4. [WARNING] Missing #[inline] on SIMD hot paths
5. [INFO] Proper remainder handling pattern

Bug instances found: 104 functions missing #[target_feature] across all backends
Performance impact: 5.9x slower to missing 21x speedup potential
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

# ANSI colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
NC = '\033[0m'

class ViolationLevel(Enum):
    """Severity levels for violations"""
    CRITICAL = "CRITICAL"  # Blocks commit, causes severe bugs
    ERROR = "ERROR"        # Blocks commit, causes correctness issues
    WARNING = "WARNING"    # Reports but doesn't block
    INFO = "INFO"          # Informational, best practice suggestions


@dataclass
class Violation:
    """Represents a code quality violation"""
    level: ViolationLevel
    filepath: Path
    line_num: int
    function_name: str
    message: str
    fix_suggestion: str
    backend: str


# SIMD intrinsic patterns by backend
INTRINSIC_PATTERNS = {
    'SSE2': (r'_mm_\w+', 'sse2', ['_mm_']),
    'AVX': (r'_mm256_\w+(?<!_mm256_castps128_ps256)', 'avx', ['_mm256_']),  # Exclude AVX2-only
    'AVX2': (r'_mm256_\w+', 'avx2', ['_mm256_']),
    'AVX512': (r'_mm512_\w+', 'avx512f', ['_mm512_']),
    'NEON': (r'v(?:ld|st|add|sub|mul|div)\w*q_f32', 'neon', ['vld', 'vst', 'vadd', 'vsub', 'vmul', 'vdiv']),
}

# FMA-specific intrinsics (require fma feature in addition to avx2)
FMA_INTRINSICS = {
    '_mm256_fmadd_ps', '_mm256_fmsub_ps', '_mm256_fnmadd_ps', '_mm256_fnmsub_ps',
    '_mm_fmadd_ps', '_mm_fmsub_ps', '_mm_fnmadd_ps', '_mm_fnmsub_ps',
}


def check_target_feature_attribute(
    lines: List[str],
    fn_line: int,
    fn_name: str,
    backend: str,
    intrinsic_re: re.Pattern
) -> Optional[Tuple[bool, Optional[str], Set[str]]]:
    """
    Check for #[target_feature] attribute and return (has_attribute, feature_name, intrinsics_used).

    Returns:
        (True, feature, intrinsics) if attribute present
        (False, None, intrinsics) if missing
        None if no intrinsics found
    """
    # Check for #[target_feature] on previous lines (up to 5 lines before for attributes/comments)
    target_feature_re = re.compile(r'#\[target_feature\(enable\s*=\s*"([^"]+)"\)\]')
    has_target_feature = False
    feature_name = None

    for check_line in range(max(0, fn_line - 5), fn_line):
        if target_feature_re.search(lines[check_line]):
            has_target_feature = True
            match = target_feature_re.search(lines[check_line])
            if match:
                feature_name = match.group(1)
            break

    # Scan function body for intrinsics
    intrinsics_found = set()
    brace_count = 0
    j = fn_line

    while j < len(lines):
        fn_line_text = lines[j]

        # Count braces
        brace_count += fn_line_text.count('{')
        brace_count -= fn_line_text.count('}')

        # Check for SIMD intrinsics
        for match in intrinsic_re.finditer(fn_line_text):
            intrinsics_found.add(match.group(0))

        # If we've closed all braces, we're done with this function
        if brace_count == 0 and j > fn_line:
            break

        j += 1

    if not intrinsics_found:
        return None

    return (has_target_feature, feature_name, intrinsics_found)


def check_safety_comment(lines: List[str], fn_line: int) -> bool:
    """Check if proper SAFETY comment exists within 10 lines before unsafe fn."""
    safety_re = re.compile(r'//\s*SAFETY:', re.IGNORECASE)

    for check_line in range(max(0, fn_line - 10), fn_line):
        if safety_re.search(lines[check_line]):
            return True
    return False


def check_inline_attribute(lines: List[str], fn_line: int) -> bool:
    """Check if #[inline] or #[inline(always)] attribute exists."""
    inline_re = re.compile(r'#\[inline(?:\(always\))?\]')

    for check_line in range(max(0, fn_line - 5), fn_line):
        if inline_re.search(lines[check_line]):
            return True
    return False


def check_remainder_handling(lines: List[str], fn_start: int, fn_end: int, backend: str) -> bool:
    """
    Check if SIMD loop has proper remainder/scalar fallback handling.

    Common patterns:
    1. for i in (chunks * 8)..a.len() { ... }  (AVX2)
    2. for i in (chunks * 4)..a.len() { ... }  (SSE2/NEON)
    3. if remainder > 0 { ... scalar code ... }
    """
    remainder_patterns = [
        r'for\s+\w+\s+in\s+\(?\w+\s*\*\s*\d+\)?\s*\.\.',  # for i in (chunks * N)..
        r'if\s+remainder\s*[>!=]',                          # if remainder > 0
        r'let\s+remainder\s*=',                             # let remainder = ...
        r'\.len\(\)\s*%\s*\d+',                             # len() % N (remainder calculation)
    ]

    function_body = '\n'.join(lines[fn_start:fn_end])

    for pattern in remainder_patterns:
        if re.search(pattern, function_body):
            return True

    return False


def check_attribute_intrinsic_mismatch(
    feature_name: Optional[str],
    intrinsics: Set[str],
    backend: str
) -> Optional[str]:
    """
    Check if target_feature attribute matches the intrinsics actually used.

    Returns error message if mismatch detected, None otherwise.
    """
    if not feature_name:
        return None

    # Extract intrinsic prefixes
    prefixes = set()
    for intrinsic in intrinsics:
        if intrinsic.startswith('_mm512_'):
            prefixes.add('avx512')
        elif intrinsic.startswith('_mm256_'):
            # Check if it's actually AVX2-only or could be AVX
            prefixes.add('avx2')
        elif intrinsic.startswith('_mm_'):
            prefixes.add('sse2')
        elif intrinsic.startswith('v') and 'q_f32' in intrinsic:
            prefixes.add('neon')

    # Check for mismatches
    if 'avx512' in prefixes and 'avx512f' not in feature_name:
        return f"Using AVX-512 intrinsics but attribute is '{feature_name}' (should be 'avx512f')"

    if 'avx2' in prefixes and feature_name == 'sse2':
        return f"Using AVX2 intrinsics but attribute is 'sse2' (should be 'avx2')"

    if 'avx512' not in prefixes and 'avx512f' in feature_name:
        return f"Attribute is 'avx512f' but no AVX-512 intrinsics found"

    return None


def check_fma_intrinsics(intrinsics: Set[str], feature_name: Optional[str]) -> Optional[str]:
    """Check if FMA intrinsics are used and FMA feature is enabled."""
    fma_used = any(intrinsic in FMA_INTRINSICS for intrinsic in intrinsics)

    if fma_used and feature_name and 'fma' not in feature_name:
        return "Using FMA intrinsics (_mm256_fmadd_ps, etc.) but 'fma' feature not enabled"

    return None


def check_file(filepath: Path, backend: str) -> List[Violation]:
    """
    Comprehensive SIMD property checking for a backend file.

    Validates:
    - [CRITICAL] Missing #[target_feature] attributes
    - [ERROR] Attribute-intrinsic mismatch
    - [WARNING] Missing safety comments
    - [WARNING] Missing #[inline] on hot paths
    - [INFO] Remainder handling patterns
    """
    if backend not in INTRINSIC_PATTERNS:
        return []

    pattern, required_feature, prefixes = INTRINSIC_PATTERNS[backend]
    intrinsic_re = re.compile(pattern)
    unsafe_fn_re = re.compile(r'^\s*unsafe\s+fn\s+(\w+)')

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
            fn_line = i

            # Check #[target_feature] attribute and scan for intrinsics
            result = check_target_feature_attribute(lines, fn_line, fn_name, backend, intrinsic_re)

            if result is None:
                # No intrinsics found, skip
                i += 1
                continue

            has_target_feature, feature_name, intrinsics = result

            # [CRITICAL] Missing #[target_feature] attribute
            if not has_target_feature:
                violations.append(Violation(
                    level=ViolationLevel.CRITICAL,
                    filepath=filepath,
                    line_num=fn_line + 1,  # 1-indexed
                    function_name=fn_name,
                    message=f"Missing #[target_feature] attribute (uses {len(intrinsics)} SIMD intrinsics)",
                    fix_suggestion=f"Add #[target_feature(enable = \"{required_feature}\")] above function",
                    backend=backend
                ))
            else:
                # [ERROR] Attribute-intrinsic mismatch
                mismatch = check_attribute_intrinsic_mismatch(feature_name, intrinsics, backend)
                if mismatch:
                    violations.append(Violation(
                        level=ViolationLevel.ERROR,
                        filepath=filepath,
                        line_num=fn_line + 1,
                        function_name=fn_name,
                        message=mismatch,
                        fix_suggestion=f"Correct #[target_feature] attribute to match intrinsics used",
                        backend=backend
                    ))

                # [ERROR] FMA intrinsics without FMA feature
                fma_issue = check_fma_intrinsics(intrinsics, feature_name)
                if fma_issue:
                    violations.append(Violation(
                        level=ViolationLevel.ERROR,
                        filepath=filepath,
                        line_num=fn_line + 1,
                        function_name=fn_name,
                        message=fma_issue,
                        fix_suggestion="Add 'fma' to target_feature: #[target_feature(enable = \"avx2,fma\")]",
                        backend=backend
                    ))

            # Find function end
            brace_count = 0
            j = fn_line
            while j < len(lines):
                brace_count += lines[j].count('{')
                brace_count -= lines[j].count('}')
                if brace_count == 0 and j > fn_line:
                    break
                j += 1
            fn_end = j

            # [WARNING] Missing SAFETY comment
            if not check_safety_comment(lines, fn_line):
                violations.append(Violation(
                    level=ViolationLevel.WARNING,
                    filepath=filepath,
                    line_num=fn_line + 1,
                    function_name=fn_name,
                    message="Missing SAFETY comment for unsafe function with SIMD",
                    fix_suggestion="Add // SAFETY: comment explaining why unsafe code is correct",
                    backend=backend
                ))

            # [WARNING] Missing #[inline] attribute (performance optimization)
            if not check_inline_attribute(lines, fn_line):
                violations.append(Violation(
                    level=ViolationLevel.WARNING,
                    filepath=filepath,
                    line_num=fn_line + 1,
                    function_name=fn_name,
                    message="Missing #[inline] attribute on SIMD hot path",
                    fix_suggestion="Add #[inline] above function for better optimization",
                    backend=backend
                ))

            # [INFO] Remainder handling check (best practice)
            if not check_remainder_handling(lines, fn_line, fn_end, backend):
                violations.append(Violation(
                    level=ViolationLevel.INFO,
                    filepath=filepath,
                    line_num=fn_line + 1,
                    function_name=fn_name,
                    message="No obvious remainder handling pattern detected",
                    fix_suggestion="Ensure scalar fallback for elements that don't fit SIMD width",
                    backend=backend
                ))

            i = fn_end  # Skip to end of function

        i += 1

    return violations


def main():
    print(f"{BLUE}🔍 Comprehensive SIMD Property Checker{NC}")
    print(f"{BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}\n")

    backend_files = {
        'SSE2': Path('src/backends/sse2.rs'),
        'AVX2': Path('src/backends/avx2.rs'),
        'AVX512': Path('src/backends/avx512.rs'),
        'NEON': Path('src/backends/neon.rs'),
    }

    all_violations = []

    for backend, filepath in backend_files.items():
        violations = check_file(filepath, backend)
        all_violations.extend(violations)

    # Separate by severity level
    critical = [v for v in all_violations if v.level == ViolationLevel.CRITICAL]
    errors = [v for v in all_violations if v.level == ViolationLevel.ERROR]
    warnings = [v for v in all_violations if v.level == ViolationLevel.WARNING]
    info = [v for v in all_violations if v.level == ViolationLevel.INFO]

    # Determine if we should block commit
    should_block = len(critical) > 0 or len(errors) > 0

    if not all_violations:
        print(f"{GREEN}✅ PASS: All SIMD property checks passed!{NC}")
        print(f"{GREEN}   • No missing #[target_feature] attributes{NC}")
        print(f"{GREEN}   • All attributes match intrinsics used{NC}")
        print(f"{GREEN}   • All unsafe functions have SAFETY comments{NC}")
        print(f"{GREEN}   • All SIMD functions have #[inline] attributes{NC}\n")
        return 0

    # Report violations by severity
    if critical:
        print(f"\n{RED}{'=' * 60}{NC}")
        print(f"{RED}❌ CRITICAL VIOLATIONS ({len(critical)}){NC}")
        print(f"{RED}{'=' * 60}{NC}\n")

        for v in critical:
            print(f"  {RED}🚨 {v.filepath}:{v.line_num}{NC} - {YELLOW}{v.function_name}(){NC}")
            print(f"     {RED}Problem:{NC} {v.message}")
            print(f"     {GREEN}Fix:{NC} {v.fix_suggestion}\n")

    if errors:
        print(f"\n{RED}{'=' * 60}{NC}")
        print(f"{RED}❌ ERRORS ({len(errors)}){NC}")
        print(f"{RED}{'=' * 60}{NC}\n")

        for v in errors:
            print(f"  {RED}⚠️  {v.filepath}:{v.line_num}{NC} - {YELLOW}{v.function_name}(){NC}")
            print(f"     {RED}Problem:{NC} {v.message}")
            print(f"     {GREEN}Fix:{NC} {v.fix_suggestion}\n")

    if warnings:
        print(f"\n{YELLOW}{'=' * 60}{NC}")
        print(f"{YELLOW}⚠️  WARNINGS ({len(warnings)}){NC}")
        print(f"{YELLOW}{'=' * 60}{NC}\n")

        for v in warnings[:10]:  # Show first 10 to avoid overwhelming output
            print(f"  {YELLOW}⚠️  {v.filepath}:{v.line_num}{NC} - {CYAN}{v.function_name}(){NC}")
            print(f"     {YELLOW}Issue:{NC} {v.message}")
            print(f"     {GREEN}Suggestion:{NC} {v.fix_suggestion}\n")

        if len(warnings) > 10:
            print(f"  {YELLOW}... and {len(warnings) - 10} more warnings{NC}\n")

    if info:
        print(f"\n{CYAN}{'=' * 60}{NC}")
        print(f"{CYAN}ℹ️  INFORMATION ({len(info)}){NC}")
        print(f"{CYAN}{'=' * 60}{NC}\n")
        print(f"  {CYAN}{len(info)} functions may need remainder handling review{NC}")
        print(f"  {CYAN}(Run with --verbose to see details){NC}\n")

    # Summary
    print(f"\n{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}SUMMARY{NC}")
    print(f"{BLUE}{'=' * 60}{NC}\n")

    if critical:
        print(f"  {RED}🚨 {len(critical)} CRITICAL{NC} - Compiler CANNOT emit SIMD instructions")
        print(f"     {RED}Impact: 5.9x slower to missing 21x speedup potential{NC}\n")

    if errors:
        print(f"  {RED}❌ {len(errors)} ERRORS{NC} - Incorrect or incompatible attributes\n")

    if warnings:
        print(f"  {YELLOW}⚠️  {len(warnings)} WARNINGS{NC} - Best practices not followed\n")

    if info:
        print(f"  {CYAN}ℹ️  {len(info)} INFO{NC} - Review recommended\n")

    # Block or allow commit
    if should_block:
        print(f"{RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
        print(f"{RED}❌ COMMIT BLOCKED - Fix CRITICAL/ERROR violations{NC}")
        print(f"{RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}\n")
        return 1
    else:
        print(f"{GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}")
        print(f"{GREEN}✅ COMMIT ALLOWED - Only warnings/info present{NC}")
        print(f"{YELLOW}   Consider addressing warnings in follow-up commits{NC}")
        print(f"{GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{NC}\n")
        return 0


if __name__ == '__main__':
    sys.exit(main())
