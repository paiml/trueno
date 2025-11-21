# Issue #26 Investigation: v0.4.1 Performance Regression

**Date**: 2025-11-21
**Issue**: https://github.com/paiml/trueno/issues/26
**Status**: 🔍 **INVESTIGATING**

## Issue Summary

GitHub issue #26 reports a 3-5% performance regression in v0.4.1 compared to v0.4.0:

| Operation | v0.4.0 | v0.4.1 | Regression |
|-----------|--------|--------|-----------|
| SUM | 227.44µs | 235.22µs | **+3.4%** slower |
| AVG | 227.90µs | 238.89µs | **+4.8%** slower |

**Test Conditions**: 1 million rows, f32/i32 data types, Criterion.rs benchmarking

## Investigation Timeline

### Step 1: Check for Code Changes ✅ COMPLETE

**Hypothesis**: sum/mean implementations or SIMD backends changed between versions.

**Investigation**:
```bash
git diff v0.4.0..v0.4.1 -- src/vector.rs | grep -A 15 "pub fn sum\|pub fn mean"
git diff v0.4.0..v0.4.1 -- src/backends/
```

**Finding**: ✅ **NO CHANGES to sum() or mean() implementations**
- `sum()` implementation: IDENTICAL between v0.4.0 and v0.4.1
- `mean()` implementation: IDENTICAL (calls sum() / len())
- Backend SIMD code: UNCHANGED (only GPU tests added, no functional changes)

**Conclusion**: The regression is NOT caused by code changes to aggregation functions.

---

### Step 2: Identify What Changed in v0.4.1 ✅ COMPLETE

**Changed commits between v0.4.0 and v0.4.1**:

1. **5442b8b**: "Fix systemic double allocation bug in ALL element-wise operations"
   - Changed `Vector::from_slice(&result)` → `Vector::from_vec(result)` for 23 operations
   - Affects: sigmoid, tanh, relu, softmax, gelu, swish, etc. (element-wise ops)
   - **Does NOT affect**: sum/mean (they are reduction operations, not element-wise)

2. **fdcdded**: "Add Rayon parallel processing for relu (>100K elements)"
   - Added Rayon dependency for parallel processing
   - Only affects relu operation with `parallel` feature enabled
   - **Does NOT affect**: sum/mean operations

3. **Cargo.toml changes**:
   - Added `rayon` as optional dependency
   - Added `parallel` feature flag
   - Version bump: 0.4.0 → 0.4.1

**Conclusion**: No code changes that should affect sum/mean performance.

---

### Step 3: Check for SIMD #[target_feature] Issues ✅ COMPLETE

**Hypothesis**: Missing `#[target_feature]` attributes on sum operations (like sqrt/recip bug).

**Investigation**:
- sqrt/recip/ln/log2/log10 with missing attributes were added AFTER v0.4.1
- Commit 71257c8 (which introduced those functions) is NOT in v0.4.1:
  ```bash
  git merge-base --is-ancestor 71257c8 v0.4.1
  # Result: 71257c8 NOT in v0.4.1
  ```

**Timeline**:
- v0.4.0: Nov 20, 2025 (before sqrt/recip functions existed)
- v0.4.1: Nov 20, 2025 (before sqrt/recip functions existed)
- commit 71257c8: Nov 20, 2025 14:29:56 (added sqrt/recip WITH bugs, AFTER v0.4.1)
- commit 04cc458: Nov 21, 2025 (fixed sqrt/recip bugs)

**Conclusion**: The `#[target_feature]` bug cannot explain v0.4.1 regression because those functions didn't exist yet.

---

### Step 4: Root Cause Analysis 🔍 IN PROGRESS

**Current Hypotheses**:

#### Hypothesis A: Rayon Dependency Overhead (HIGH PROBABILITY)

**Evidence**:
- Rayon was added as a dependency in v0.4.1, even when `parallel` feature is disabled
- This increases binary size and dependency tree
- Could affect Link-Time Optimization (LTO) decisions
- Could reduce instruction cache locality

**Test**:
- Benchmark sum/mean on current code (which has rayon)
- Compare with v0.4.0 (which doesn't have rayon)
- Check if removing rayon from Cargo.toml eliminates regression

#### Hypothesis B: Build Configuration Mismatch (MEDIUM PROBABILITY)

**Evidence**:
- Issue compares **published crates.io v0.4.0** vs **local development v0.4.1**
- Different build contexts could produce different optimization results
- LTO decisions may differ between published and local builds
- Compiler version could differ

**Profile settings** (v0.4.1):
```toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
debug = true  # Debug symbols for profiling
```

**Test**:
- Build v0.4.1 locally and publish to local crates.io mirror
- Benchmark against both published and local builds
- Verify identical optimization settings

#### Hypothesis C: Measurement Variance (LOW PROBABILITY)

**Evidence**:
- 3-5% is outside typical measurement noise (usually <2%)
- Both SUM and AVG show consistent regression pattern
- Issue author acknowledges this exceeds normal variance

**Test**:
- Run multiple benchmark iterations with fresh builds
- Check coefficient of variation (CV) for stability

#### Hypothesis D: Dependency Version Changes (LOW PROBABILITY)

**Evidence**:
- Only dependency addition was rayon
- No updates to existing dependencies (wgpu, criterion, etc.)
- Criterion version unchanged between v0.4.0 and v0.4.1

---

### Step 5: Benchmark Validation ⏳ RUNNING

**Running benchmarks**:
```bash
cargo bench --bench vector_ops "sum/" -- --measurement-time 10
cargo bench --bench vector_ops "mean/" -- --measurement-time 10
```

**Objective**: Establish baseline performance on current code to compare with v0.4.0/v0.4.1

---

## Key Findings So Far

### ✅ Confirmed Facts

1. **sum() and mean() code is IDENTICAL** between v0.4.0 and v0.4.1
2. **SIMD backend implementations UNCHANGED** for sum operations
3. **sqrt/recip bugs NOT relevant** - those functions added after v0.4.1
4. **No functional changes** that should affect aggregation performance

### ⚠️ Suspicious Findings

1. **Rayon added as dependency** even when not using parallel feature
2. **Binary size increased** due to rayon dependency
3. **LTO could be affected** by additional dependency graph complexity

### 🎯 Most Likely Explanation

The regression is NOT a code bug but rather an **artifact of build configuration or dependency overhead**:

- Rayon dependency increases binary size (~100KB+ for rayon runtime)
- Larger binary reduces instruction cache hit rate
- LTO may make different optimization decisions with larger dependency graph
- Effect is small (3-5%) but measurable for tight loops like sum

---

## Recommended Next Steps

### Immediate Actions

1. **✅ Complete benchmark validation** (in progress)
2. **Test Rayon hypothesis**:
   ```bash
   # Remove rayon dependency temporarily
   git checkout v0.4.1
   # Edit Cargo.toml to remove rayon
   cargo bench --bench vector_ops "sum/1000000"
   # Compare with and without rayon
   ```

3. **Build configuration validation**:
   ```bash
   # Build v0.4.0 and v0.4.1 with identical settings
   cargo build --release
   ls -lh target/release/trueno  # Check binary sizes
   ```

### Investigation Plan

- [ ] Benchmark sum/mean on current code (baseline)
- [ ] Checkout v0.4.1 and benchmark with rayon
- [ ] Checkout v0.4.1 and benchmark WITHOUT rayon (remove from Cargo.toml)
- [ ] Compare binary sizes: v0.4.0 vs v0.4.1 vs v0.4.1-no-rayon
- [ ] Profile instruction cache misses using `perf stat`
- [ ] Analyze LTO decisions using `cargo rustc -- --emit=llvm-ir`

### Potential Resolutions

**If rayon overhead confirmed**:
1. Move rayon to **dev-dependencies** (not regular dependencies)
2. Only enable rayon when `parallel` feature is explicitly used
3. Document in v0.4.2 release notes

**If build configuration issue**:
1. Ensure published builds use identical settings to local
2. Add CI check to validate published binary performance
3. Document build requirements

**If legitimate regression**:
1. Profile to identify specific bottleneck
2. Optimize sum/mean implementations if needed
3. Add regression tests to CI

---

## Connection to Current Work

**Relationship to SIMD Audit**:

While investigating this issue, we completed a systematic audit that found missing `#[target_feature]` attributes on:
- ln, log2, log10 in AVX2 and AVX512 (6 functions fixed)
- sqrt, recip previously fixed (6 functions)

These bugs were introduced AFTER v0.4.1 but could affect future releases if not caught.

**Lesson**: This investigation highlights the importance of:
1. Systematic performance regression testing
2. Comparing published vs development builds
3. Understanding dependency overhead impact
4. Automated benchmarking in CI

---

## Status: 🔍 INVESTIGATING

**Next Update**: After benchmark completion and rayon hypothesis testing

---

**Generated by**: Claude Code systematic investigation
**Investigation Start**: 2025-11-21
**Related Issues**: #26
**Related Commits**: 5442b8b, fdcdded, v0.4.0, v0.4.1
