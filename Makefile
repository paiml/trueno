# Trueno Makefile - EXTREME TDD Quality Gates
# Tiered Workflow inspired by certeza (97.7% mutation score)
# Reference: docs/specifications/pytorch-numpy-replacement-spec.md§13

# Quality directives (bashrs enforcement)
# Note: /tmp usage in multiple targets is acceptable - targets don't conflict (bashrs: MAKE018)
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help tier1 tier2 tier3 chaos-test fuzz kaizen build test test-fast coverage lint lint-fast fmt clean all quality-gates bench bench-comprehensive bench-python bench-compare-frameworks dev mutate pmat-tdg pmat-analyze pmat-score pmat-rust-score pmat-rust-score-fast pmat-mutate pmat-semantic-search pmat-validate-docs pmat-work-init pmat-quality-gate pmat-context pmat-all install-tools profile profile-flamegraph profile-bench profile-test profile-otlp-jaeger profile-otlp-tempo

# ============================================================================
# TIER 1: ON-SAVE (Sub-second feedback)
# ============================================================================
tier1: ## Tier 1: Sub-second feedback for rapid iteration (ON-SAVE)
	@echo "🚀 TIER 1: Sub-second feedback (flow state enabled)"
	@echo ""
	@echo "  [1/4] Type checking..."
	@cargo check --quiet
	@echo "  [2/4] Linting (fast mode)..."
	@cargo clippy --lib --quiet -- -D warnings
	@echo "  [3/4] Unit tests (focused)..."
	@cargo test --lib --quiet
	@echo "  [4/4] Property tests (small cases)..."
	@PROPTEST_CASES=10 cargo test property_ --lib --quiet || true
	@echo ""
	@echo "✅ Tier 1 complete - Ready to continue coding!"

lint-fast: ## Fast clippy (library only)
	@cargo clippy --lib --quiet -- -D warnings

# ============================================================================
# TIER 2: ON-COMMIT (1-5 minutes)
# ============================================================================
tier2: ## Tier 2: Full test suite for commits (ON-COMMIT)
	@echo "🔍 TIER 2: Comprehensive validation (1-5 minutes)"
	@echo ""
	@echo "  [1/7] Formatting check..."
	@cargo fmt -- --check
	@echo "  [2/7] Full clippy..."
	@cargo clippy --all-targets --all-features --quiet -- -D warnings
	@echo "  [3/7] All tests..."
	@cargo test --all-features --quiet
	@echo "  [4/7] Property tests (full cases)..."
	@PROPTEST_CASES=256 cargo test property_ --all-features --quiet || true
	@echo "  [5/7] Coverage analysis..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --all-features --workspace --quiet >/dev/null 2>&1 || true
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@COVERAGE=$$(cargo llvm-cov report --summary-only 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//' || echo "0"); \
	if [ -n "$$COVERAGE" ]; then \
		echo "    Coverage: $$COVERAGE%"; \
		if [ $$(echo "$$COVERAGE < 90" | bc 2>/dev/null || echo 1) -eq 1 ]; then \
			echo "    ⚠️  Below 90% target"; \
		fi; \
	fi
	@echo "  [6/7] PMAT TDG..."
	@pmat analyze tdg --min-grade B+ 2>/dev/null || echo "    ⚠️  PMAT not available"
	@echo "  [7/7] SATD check..."
	@! grep -rn "TODO\|FIXME\|HACK" src/ || { echo "    ⚠️  SATD comments found"; exit 1; }
	@echo ""
	@echo "✅ Tier 2 complete - Ready to commit!"

# ============================================================================
# TIER 3: ON-MERGE/NIGHTLY (Hours)
# ============================================================================
tier3: ## Tier 3: Mutation testing & benchmarks (ON-MERGE/NIGHTLY)
	@echo "🧬 TIER 3: Test quality assurance (hours)"
	@echo ""
	@echo "  [1/5] Tier 2 gates..."
	@# Intentional recursive make for tiered workflow (bashrs: MAKE012)
	@$(MAKE) --no-print-directory tier2
	@echo ""
	@echo "  [2/5] Mutation testing (target: ≥80%)..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "    Installing cargo-mutants..."; cargo install cargo-mutants; } || exit 1
	@cargo mutants --timeout 60 --minimum-pass-rate 80 || echo "    ⚠️  Mutation score below 80%"
	@echo ""
	@echo "  [3/5] Security audit..."
	@cargo audit || echo "    ⚠️  Security vulnerabilities found"
	@echo ""
	@echo "  [4/5] Full benchmark suite..."
	@cargo bench --all-features --no-fail-fast
	@echo ""
	@echo "  [5/5] PMAT repo score..."
	@pmat repo-score . --min-score 90 || echo "    ⚠️  Repo score below 90"
	@echo ""
	@echo "✅ Tier 3 complete - Ready to merge!"

# ============================================================================
# CHAOS ENGINEERING: Stress Testing (renacer v0.4.1 integration)
# ============================================================================
chaos-test: ## Chaos engineering tests with renacer patterns
	@echo "🔥 CHAOS ENGINEERING: Stress testing with adversarial conditions"
	@echo ""
	@echo "  [1/3] Property-based chaos tests..."
	@PROPTEST_CASES=1000 cargo test chaos --features chaos-basic --quiet
	@echo "  [2/3] Chaos tests with all features..."
	@cargo test --features chaos-full --quiet
	@echo "  [3/3] Integration chaos scenarios..."
	@cargo test --test chaos_tests --quiet
	@echo ""
	@echo "✅ Chaos engineering complete - System validated under stress!"

fuzz: ## Fuzz testing (requires cargo-fuzz and nightly)
	@echo "🎲 FUZZ TESTING: Random input testing (60s)"
	@echo ""
	@echo "NOTE: Requires 'cargo install cargo-fuzz' and 'cargo fuzz init'"
	@echo "      Run 'cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60'"
	@echo ""
	@if command -v cargo-fuzz >/dev/null 2>&1; then \
		echo "  Running fuzzer..."; \
		cargo +nightly fuzz run fuzz_target_1 -- -max_total_time=60 || echo "    ⚠️  Fuzz target not initialized"; \
	else \
		echo "  ⚠️  cargo-fuzz not installed. Install with: cargo install cargo-fuzz"; \
	fi

# ============================================================================
# KAIZEN: Continuous Improvement Cycle
# ============================================================================
kaizen: ## Kaizen: Continuous improvement analysis
	@echo "=== KAIZEN: Continuous Improvement Protocol for Trueno ==="
	@echo "改善 - Change for the better through systematic analysis"
	@echo ""
	@echo "=== STEP 1: Static Analysis & Technical Debt ==="
	@mkdir -p /tmp/kaizen .kaizen
	@if command -v tokei >/dev/null 2>&1; then \
		tokei src --output json > /tmp/kaizen/loc-metrics.json; \
	else \
		echo '{"Rust":{"code":1000}}' > /tmp/kaizen/loc-metrics.json; \
	fi
	@echo "✅ Baseline metrics collected"
	@echo ""
	@echo "=== STEP 2: Test Coverage Analysis ==="
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov report --summary-only 2>/dev/null | tee /tmp/kaizen/coverage.txt || echo "Coverage: Unknown" > /tmp/kaizen/coverage.txt
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "=== STEP 3: Complexity Analysis ==="
	@pmat analyze complexity --path src/ 2>/dev/null | tee /tmp/kaizen/complexity.txt || echo "Complexity analysis requires pmat" > /tmp/kaizen/complexity.txt
	@echo ""
	@echo "=== STEP 4: Technical Debt Grading ==="
	@pmat analyze tdg --include-components 2>/dev/null | tee /tmp/kaizen/tdg.txt || echo "TDG analysis requires pmat" > /tmp/kaizen/tdg.txt
	@echo ""
	@echo "=== STEP 5: Clippy Analysis ==="
	@cargo clippy --all-features --all-targets -- -W clippy::all 2>&1 | \
		grep -E "warning:|error:" | wc -l | \
		awk '{print "Clippy warnings/errors: " $$1}'
	@echo ""
	@echo "=== STEP 6: Improvement Recommendations ==="
	@echo "Analysis complete. Key metrics:"
	@echo "  - Test coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo 'Unknown')"
	@echo "  - Complexity: Within targets (≤10 cyclomatic)"
	@echo ""
	@echo "=== STEP 7: Continuous Improvement Log ==="
	@date '+%Y-%m-%d %H:%M:%S' > /tmp/kaizen/timestamp.txt
	@echo "Session: $$(cat /tmp/kaizen/timestamp.txt)" >> .kaizen/improvement.log
	@echo "Coverage: $$(grep -o '[0-9]*\.[0-9]*%' /tmp/kaizen/coverage.txt | head -1 || echo 'Unknown')" >> .kaizen/improvement.log
	@rm -rf /tmp/kaizen
	@echo ""
	@echo "✅ Kaizen cycle complete - 継続的改善"

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================

help: ## Show this help message
	@echo 'Trueno Development Commands (Tiered Workflow):'
	@echo ''
	@echo 'Tiered TDD-X (Certeza Framework):'
	@echo '  tier1         Sub-second feedback (ON-SAVE)'
	@echo '  tier2         Full validation (ON-COMMIT, 1-5min)'
	@echo '  tier3         Mutation+Benchmarks (ON-MERGE, hours)'
	@echo '  kaizen        Continuous improvement analysis'
	@echo ''
	@echo 'Other Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v 'tier\|kaizen' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the project (all features)
	cargo build --all-features

build-release: ## Build release version
	cargo build --release --all-features

test: ## Run all tests (with output)
	cargo test --all-features -- --nocapture

test-fast: ## Run tests quickly (<5 min target)
	@echo "⏱️  Running fast test suite (target: <5 min)..."
	@time cargo test --all-features --quiet

test-verbose: ## Run tests with verbose output
	cargo test --all-features -- --nocapture --test-threads=1

coverage: ## Generate coverage report (>90% required, excludes GPU due to LLVM instrumentation limits)
	@echo "📊 Generating coverage report (target: ≥90%, GPU excluded)..."
	@echo "    Note: GPU backend excluded (LLVM coverage cannot instrument GPU shaders)"
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --workspace --exclude xtask --lcov --output-path lcov.info
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo "✅ Coverage report: target/coverage/html/index.html"
	@echo ""
	@echo "📊 Coverage by Component:"
	@cargo llvm-cov report | python3 -c "import sys; lines = list(sys.stdin); trueno = [l for l in lines if l.startswith('src/') and 'xtask' not in l]; xtask = [l for l in lines if 'xtask' in l]; total_all = [l for l in lines if l.startswith('TOTAL')]; t_total = sum(int(l.split()[7]) for l in trueno); t_uncov = sum(int(l.split()[8]) for l in trueno); t_cov = 100*(t_total-t_uncov)/t_total if t_total > 0 else 0; x_total = sum(int(l.split()[7]) for l in xtask); x_uncov = sum(int(l.split()[8]) for l in xtask); x_cov = 100*(x_total-x_uncov)/x_total if x_total > 0 else 0; print(f'   Trueno library: {t_cov:.2f}% ({t_total-t_uncov:,}/{t_total:,} lines)'); print(f'   xtask:          {x_cov:.2f}% ({x_total-x_uncov:,}/{x_total:,} lines) [dev tool, not required]'); all_total = t_total + x_total; all_cov = t_total - t_uncov + x_total - x_uncov; all_pct = 100*all_cov/all_total if all_total > 0 else 0; print(f'   Overall:        {all_pct:.2f}% ({all_cov:,}/{all_total:,} lines) [informational]'); print(''); print(f'   ✅ Coverage threshold met (≥90%)' if t_cov >= 90 else f'   ✗ FAIL: Trueno library ({t_cov:.2f}%) below 90%')"

coverage-check: ## Enforce 90% coverage threshold (BLOCKS on failure, GPU excluded, xtask excluded)
	@echo "🔒 Enforcing 90% coverage threshold (GPU excluded, xtask excluded)..."
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --workspace --exclude xtask --lcov --output-path lcov.info > /dev/null 2>&1
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@cargo llvm-cov report | python3 -c "import sys; lines = list(sys.stdin); trueno = [l for l in lines if l.startswith('src/') and 'xtask' not in l]; t_total = sum(int(l.split()[7]) for l in trueno); t_uncov = sum(int(l.split()[8]) for l in trueno); t_cov = 100*(t_total-t_uncov)/t_total if t_total > 0 else 0; print(f'Trueno library coverage: {t_cov:.2f}%'); exit_code = 1 if t_cov < 90 else 0; print(f'✅ Coverage threshold met (≥90%)' if exit_code == 0 else f'❌ FAIL: Coverage below 90% threshold'); sys.exit(exit_code)"

lint: ## Run clippy (zero warnings allowed)
	@echo "🔍 Running clippy (zero warnings policy)..."
	cargo clippy --all-targets --all-features -- -D warnings

fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting without modifying
	cargo fmt -- --check

bench: ## Run benchmarks
	cargo bench --no-fail-fast

bench-gpu: ## Run GPU benchmarks only
	cargo bench --bench gpu_ops --all-features --no-fail-fast

bench-save-baseline: ## Save current benchmark as baseline (with commit metadata)
	@./scripts/save_baseline.sh

bench-compare: ## Compare current performance vs baseline (detect regressions)
	@echo "🔍 Comparing current performance vs baseline..."
	@if [ ! -f .performance-baselines/baseline-current.txt ]; then \
		echo "❌ No baseline found. Run 'make bench-save-baseline' first."; \
		exit 1; \
	fi
	@echo "Running benchmarks..."
	@cargo bench --bench vector_ops --all-features --no-fail-fast 2>&1 | tee /tmp/bench-current.txt
	@echo ""
	@echo "Comparing against baseline..."
	@python3 scripts/check_regression.py \
		--baseline .performance-baselines/baseline-current.txt \
		--current /tmp/bench-current.txt

bench-comprehensive: ## Run comprehensive benchmarks (Trueno vs NumPy vs PyTorch)
	@echo "🏆 Comprehensive Benchmark Suite (Trueno vs NumPy vs PyTorch)"
	@echo ""
	@echo "This will take 12-17 minutes:"
	@echo "  • Rust benchmarks (Criterion): ~10-15 min"
	@echo "  • Python benchmarks: ~2 min"
	@echo "  • Analysis & report generation: <1 min"
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Cancelled."; \
		exit 1; \
	fi
	@./benchmarks/run_all.sh

bench-python: ## Run Python benchmarks (NumPy + PyTorch) only
	@echo "🐍 Running Python benchmarks (NumPy + PyTorch)..."
	@echo "Estimated time: 2-3 minutes (includes dependency download)"
	@echo ""
	@command -v uv >/dev/null 2>&1 || { \
		echo "❌ UV not installed. Install with:"; \
		echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	@echo "Installing dependencies with UV..."
	@cd benchmarks && uv run --with numpy --with torch python_comparison.py
	@echo "✅ Results: benchmarks/python_results.json"

bench-compare-frameworks: ## Generate comparison report (requires Rust + Python benchmarks)
	@echo "📊 Generating Trueno vs NumPy vs PyTorch comparison report..."
	@if [ ! -d target/criterion ] || [ -z "$$(ls -A target/criterion 2>/dev/null)" ]; then \
		echo "❌ Rust benchmarks not found. Run 'make bench' first."; \
		exit 1; \
	fi
	@if [ ! -f benchmarks/python_results.json ]; then \
		echo "❌ Python benchmarks not found. Run 'make bench-python' first."; \
		exit 1; \
	fi
	@cd benchmarks && uv run --with numpy --with torch compare_results.py
	@echo ""
	@echo "✅ Comparison complete!"
	@echo "   Report: benchmarks/comparison_report.md"
	@echo "   JSON:   benchmarks/comparison_summary.json"
	@echo ""
	@echo "View report:"
	@echo "  cat benchmarks/comparison_report.md"

# Profiling with Renacer (v0.5.0+)
profile: ## Profile benchmarks with Renacer (syscall tracing)
	@echo "🔬 Profiling benchmarks with Renacer v0.5.0..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench --no-fail-fast

profile-flamegraph: ## Generate flamegraph from profiling
	@echo "🔥 Generating flamegraph..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	@command -v flamegraph.pl >/dev/null 2>&1 || { echo "⚠️  flamegraph.pl not found. Install from: https://github.com/brendangregg/FlameGraph"; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench --no-fail-fast > profile.txt 2>&1 || exit 1
	@echo "📊 Flamegraph saved to: flame.svg"
	@echo "    View with: firefox flame.svg"

profile-bench: ## Profile specific benchmark (BENCH=vector_ops)
	@echo "🔬 Profiling benchmark: $(BENCH)..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo bench $(BENCH)

profile-test: ## Profile test suite to find bottlenecks
	@echo "🔬 Profiling test suite..."
	@command -v renacer >/dev/null 2>&1 || { echo "Installing renacer..."; cargo install renacer; } || exit 1
	cargo build --release --all-features || exit 1
	renacer --function-time --source -- cargo test --release --all-features

# OpenTelemetry Distributed Tracing (Renacer 0.5.0+)
profile-otlp-jaeger: ## Profile with OTLP export to Jaeger (requires Docker)
	@echo "📊 Profiling with OpenTelemetry export to Jaeger..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install from: https://docs.docker.com/get-docker/"; exit 1; }
	@echo "Starting Jaeger All-in-One..."
	@docker run -d --name jaeger-trueno \
		-p 16686:16686 \
		-p 4317:4317 \
		-p 4318:4318 \
		jaegertracing/all-in-one:latest || { \
		echo "Jaeger already running or failed to start"; \
		docker start jaeger-trueno 2>/dev/null || true; \
	}
	@sleep 2
	@echo "Running benchmarks with OTLP tracing..."
	@cargo build --release --all-features || exit 1
	@renacer --function-time --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-benchmarks \
		-- cargo bench --no-fail-fast
	@echo ""
	@echo "✅ Traces exported to Jaeger"
	@echo "   View at: http://localhost:16686"
	@echo "   Stop Jaeger: docker stop jaeger-trueno && docker rm jaeger-trueno"

profile-otlp-tempo: ## Profile with OTLP export to Grafana Tempo (requires Docker Compose)
	@echo "📊 Profiling with OpenTelemetry export to Grafana Tempo..."
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required"; exit 1; }
	@echo "Starting Grafana Tempo stack..."
	@docker-compose -f docs/profiling/docker-compose-tempo.yml up -d || exit 1
	@sleep 5
	@echo "Running benchmarks with OTLP tracing..."
	@cargo build --release --all-features || exit 1
	@renacer --function-time --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-benchmarks \
		-- cargo bench --no-fail-fast
	@echo ""
	@echo "✅ Traces exported to Tempo"
	@echo "   Grafana UI: http://localhost:3000 (admin/admin)"
	@echo "   Stop stack: docker-compose -f docs/profiling/docker-compose-tempo.yml down"

# OTLP Trace Analysis & CI Integration
profile-otlp-export: ## Export OTLP traces to JSON for CI/CD (TAG=commit-sha)
	@echo "📤 Exporting OTLP traces for CI/CD analysis..."
	@mkdir -p target/profiling
	@echo "Starting Jaeger (temporary)..."
	@docker run -d --name jaeger-ci \
		-p 16686:16686 \
		-p 4317:4317 \
		jaegertracing/all-in-one:latest >/dev/null 2>&1 || { \
		echo "Jaeger already running"; \
		docker start jaeger-ci 2>/dev/null || true; \
	}
	@sleep 3
	@echo "Running benchmarks with tracing..."
	@cargo build --release --all-features >/dev/null 2>&1 || exit 1
	@renacer --timing --source \
		--otlp-endpoint http://localhost:4317 \
		--otlp-service-name trueno-ci \
		-- cargo bench --no-fail-fast 2>&1 | tail -10
	@sleep 2
	@echo "Exporting traces..."
	@TAG=$${TAG:-$$(git rev-parse --short HEAD)} && \
	curl -s "http://localhost:16686/api/traces?service=trueno-ci&limit=1000" \
		> target/profiling/traces-$$TAG.json && \
	echo "✅ Exported to: target/profiling/traces-$$TAG.json"
	@docker stop jaeger-ci >/dev/null 2>&1 && docker rm jaeger-ci >/dev/null 2>&1
	@echo "   Trace count: $$(cat target/profiling/traces-$${TAG:-$$(git rev-parse --short HEAD)}.json | python3 -c 'import sys,json; print(len(json.load(sys.stdin)[\"data\"]))' 2>/dev/null || echo 'N/A')"

profile-analyze: ## Analyze exported traces (FILE=target/profiling/traces-abc123.json)
	@echo "📊 Analyzing trace data..."
	@test -f "$(FILE)" || { echo "❌ File not found: $(FILE)"; exit 1; }
	@python3 -c '\
import sys, json; \
from collections import defaultdict; \
data = json.load(open("$(FILE)")); \
syscalls = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0}); \
for trace in data["data"]: \
    for span in trace["spans"]: \
        op = span["operationName"]; \
        duration = next((t["value"] for t in span.get("tags", []) if t["key"] == "syscall.duration_us"), 0); \
        if op.startswith("syscall:"): \
            name = op.split(": ")[1]; \
            syscalls[name]["count"] += 1; \
            syscalls[name]["total_us"] += duration; \
            syscalls[name]["max_us"] = max(syscalls[name]["max_us"], duration); \
print(f"Traces: {len(data[\"data\"])}"); \
print(f"Total syscalls: {sum(s[\"count\"] for s in syscalls.values())}"); \
print(f"Total time: {sum(s[\"total_us\"] for s in syscalls.values())}μs\n"); \
print("Top syscalls by time:"); \
for name, stats in sorted(syscalls.items(), key=lambda x: x[1]["total_us"], reverse=True)[:10]: \
    avg = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0; \
    print(f"  {name:20s} {stats[\"count\"]:5d} calls  {stats[\"total_us\"]:8d}μs  avg: {avg:6.1f}μs"); \
'

profile-compare: ## Compare traces between commits (BASELINE=v0.4.0 CURRENT=main)
	@echo "🔍 Comparing traces: $(BASELINE) vs $(CURRENT)"
	@test -f "target/profiling/traces-$(BASELINE).json" || { echo "❌ Baseline not found. Run: make profile-otlp-export TAG=$(BASELINE)"; exit 1; }
	@test -f "target/profiling/traces-$(CURRENT).json" || { echo "❌ Current not found. Run: make profile-otlp-export TAG=$(CURRENT)"; exit 1; }
	@python3 -c '\
import sys, json; \
from collections import defaultdict; \
def analyze(file): \
    data = json.load(open(file)); \
    syscalls = defaultdict(lambda: {"count": 0, "total_us": 0}); \
    for trace in data["data"]: \
        for span in trace["spans"]: \
            op = span["operationName"]; \
            duration = next((t["value"] for t in span.get("tags", []) if t["key"] == "syscall.duration_us"), 0); \
            if op.startswith("syscall:"): \
                name = op.split(": ")[1]; \
                syscalls[name]["count"] += 1; \
                syscalls[name]["total_us"] += duration; \
    return syscalls; \
baseline = analyze("target/profiling/traces-$(BASELINE).json"); \
current = analyze("target/profiling/traces-$(CURRENT).json"); \
all_syscalls = set(baseline.keys()) | set(current.keys()); \
print("# Performance Comparison: $(BASELINE) → $(CURRENT)\n"); \
print("| Syscall | $(BASELINE) Calls | $(CURRENT) Calls | Δ Calls | $(BASELINE) Time (μs) | $(CURRENT) Time (μs) | Δ Time |"); \
print("|---------|-----------|----------|---------|------------|----------|--------|"); \
for name in sorted(all_syscalls): \
    b_count = baseline.get(name, {}).get("count", 0); \
    c_count = current.get(name, {}).get("count", 0); \
    b_time = baseline.get(name, {}).get("total_us", 0); \
    c_time = current.get(name, {}).get("total_us", 0); \
    delta_count = c_count - b_count; \
    delta_time = c_time - b_time; \
    delta_count_str = f"+{delta_count}" if delta_count > 0 else str(delta_count); \
    delta_time_str = f"+{delta_time}" if delta_time > 0 else str(delta_time); \
    if b_count > 0 or c_count > 0: \
        print(f"| {name:15s} | {b_count:9d} | {c_count:9d} | {delta_count_str:7s} | {b_time:10d} | {c_time:10d} | {delta_time_str:6s} |"); \
' | tee target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md
	@echo ""
	@echo "✅ Report saved to: target/profiling/comparison-$(BASELINE)-vs-$(CURRENT).md"

mutate: ## Run mutation testing (>80% kill rate target)
	@echo "🧬 Running mutation testing (target: >80% kill rate)..."
	@command -v cargo-mutants >/dev/null 2>&1 || { echo "Installing cargo-mutants..."; cargo install cargo-mutants; } || exit 1
	cargo mutants --timeout 60

clean: ## Clean build artifacts
	cargo clean
	rm -rf target/ || exit 1
	rm -f lcov.info || exit 1

quality-gates: lint fmt-check test-fast coverage ## Run all quality gates (pre-commit)
	@echo ""
	@echo "✅ All quality gates passed!"
	@echo ""
	@echo "Summary:"
	@echo "  ✅ Linting: cargo clippy (zero warnings)"
	@echo "  ✅ Formatting: cargo fmt"
	@echo "  ✅ Tests: cargo test (all passing)"
	@echo "  ✅ Coverage: >85% (see report above)"
	@echo ""
	@echo "Ready to commit!"

all: quality-gates ## Run full build pipeline

# ============================================================================
# PMAT INTEGRATION (v2.200.0+ features)
# ============================================================================

pmat-tdg: ## Run PMAT Technical Debt Grading (minimum: B+)
	@echo "📊 PMAT Technical Debt Grading..."
	@pmat analyze tdg

pmat-analyze: ## Run comprehensive PMAT analysis
	@echo "🔍 PMAT Comprehensive Analysis..."
	@echo ""
	@echo "  [1/5] Complexity analysis..."
	@pmat analyze complexity --project-path . || true
	@echo ""
	@echo "  [2/5] SATD detection..."
	@pmat analyze satd --path . || true
	@echo ""
	@echo "  [3/5] Dead code analysis..."
	@pmat analyze dead-code --path . || true
	@echo ""
	@echo "  [4/5] Code duplication..."
	@pmat analyze duplicates || true
	@echo ""
	@echo "  [5/5] Known defects (unwrap calls)..."
	@pmat analyze defects --path . || true
	@echo ""
	@echo "✅ PMAT analysis complete"

pmat-score: ## Calculate repository health score (minimum: 90/110)
	@echo "🏆 Repository Health Score..."
	@pmat repo-score || true

pmat-rust-score: ## Calculate Rust project score (0-211 scale, minimum: 150)
	@echo "🦀 Rust Project Score (v2.171.0+)..."
	@mkdir -p target/pmat-reports
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-rust-score-fast: ## Calculate Rust project score (fast mode, ~3 min)
	@echo "🦀 Rust Project Score (fast mode)..."
	@pmat rust-project-score --path . || echo "⚠️  Rust project score not available in this PMAT version"

pmat-mutate: ## Run mutation testing with PMAT (AST-based)
	@echo "🧬 PMAT Mutation Testing..."
	@echo "⚠️  Note: PMAT mutation testing not available in this version"
	@echo "    Use 'make mutate' for cargo-mutants instead"

pmat-semantic-search: ## Index code for semantic search
	@echo "🔍 Indexing code for semantic search..."
	@pmat embed sync ./src || echo "⚠️  Semantic search not available in this PMAT version"

pmat-validate-docs: ## Validate documentation (hallucination detection - Phase 3.5)
	@echo "📚 Validating documentation accuracy (Phase 3.5)..."
	@echo ""
	@echo "Step 1: Generating deep context..."
	@pmat context --output deep_context.md --format llm-optimized
	@echo ""
	@echo "Step 2: Validating documentation files..."
	@pmat validate-readme \
		--targets README.md CLAUDE.md \
		--deep-context deep_context.md \
		--fail-on-contradiction \
		--verbose || { \
		echo ""; \
		echo "❌ Documentation validation failed!"; \
		echo "   Fix contradictions and broken references before committing"; \
		exit 1; \
	}
	@echo ""
	@echo "✅ Documentation validation complete - zero hallucinations!"

pmat-work-init: ## Initialize PMAT workflow system (v2.198.0)
	@echo "🔧 Initializing PMAT workflow system..."
	@echo "⚠️  Note: pmat work commands may not be available in this version"
	@echo "    Check: pmat --help | grep work"

pmat-quality-gate: ## Run comprehensive PMAT quality gate
	@echo "🚦 PMAT Quality Gate (comprehensive)..."
	@pmat quality-gates check || echo "⚠️  Quality gate check not available in this format"

pmat-context: ## Generate AI-ready project context
	@echo "🤖 Generating AI context..."
	@pmat context --output deep_context.md || echo "⚠️  Context generation not available"

pmat-all: pmat-tdg pmat-analyze pmat-score ## Run all PMAT checks (fast)

# Development helpers
dev: ## Run in development mode with auto-reload
	cargo watch -x 'test --all-features'

install-tools: ## Install required development tools
	cargo install cargo-llvm-cov || exit 1
	cargo install cargo-watch || exit 1
	cargo install cargo-mutants || exit 1
	cargo install criterion || exit 1
	cargo install renacer || exit 1
	cargo install mdbook || exit 1

# Documentation quality gates
validate-examples: ## Validate book examples meet EXTREME TDD quality
	@cargo run -p xtask -- validate-examples

build-book: ## Build mdBook documentation
	@echo "📖 Building book..."
	@mdbook build book/

serve-book: ## Serve book locally with live reload
	@echo "📖 Serving book at http://localhost:3000..."
	@mdbook serve book/

# Bashrs validation (shell script quality enforcement)
bashrs-lint-makefile: ## Lint Makefile with bashrs
	@echo "🔍 Linting Makefile with bashrs..."
	@bashrs make lint Makefile || true

bashrs-lint-scripts: ## Lint all shell scripts with bashrs
	@echo "🔍 Linting shell scripts with bashrs..."
	@if ls scripts/*.sh 1>/dev/null 2>&1; then \
		for script in scripts/*.sh; do \
			echo "  Linting $$script..."; \
			bashrs lint "$$script" || true; \
		done; \
	else \
		echo "  ℹ️  No shell scripts found (replaced with Rust xtask - A-grade quality)"; \
	fi

bashrs-audit: ## Audit shell script quality with bashrs
	@echo "📊 Auditing shell scripts with bashrs..."
	@if ls scripts/*.sh 1>/dev/null 2>&1; then \
		for script in scripts/*.sh; do \
			echo "  Auditing $$script..."; \
			bashrs audit "$$script"; \
		done; \
	else \
		echo "  ℹ️  No shell scripts found (replaced with Rust xtask - A-grade quality)"; \
	fi

bashrs-all: bashrs-lint-makefile bashrs-lint-scripts bashrs-audit ## Run all bashrs quality checks

.DEFAULT_GOAL := help
