# trueno-gpu Makefile - Pure Rust PTX Generation
# Follows bashrs/trueno idiomatic patterns (mold hack, nextest, two-phase coverage)

.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help build test test-fast test-quick test-cuda test-viz coverage coverage-ci coverage-check coverage-clean clean-coverage bench bench-ptx lint fmt clean all

# ============================================================================
# TESTING
# ============================================================================

test: ## Run all tests
	cargo test -p trueno-gpu --all-features -- --nocapture

test-fast: ## Run tests quickly (<2 min target)
	@echo "⚡ Running fast tests for trueno-gpu (target: <2 min)..."
	@if command -v cargo-nextest >/dev/null 2>&1; then \
		PROPTEST_CASES=50 RUST_TEST_THREADS=$$(nproc) cargo nextest run \
			-p trueno-gpu \
			--status-level skip \
			--failure-output immediate; \
	else \
		PROPTEST_CASES=50 cargo test -p trueno-gpu; \
	fi

test-quick: test-fast ## Alias for test-fast (bashrs pattern)
	@echo "✅ Quick tests completed!"

test-cuda: ## Run tests with CUDA feature (requires NVIDIA driver)
	@echo "🖥️ Running CUDA tests (requires NVIDIA driver)..."
	cargo test -p trueno-gpu --features cuda -- --nocapture

test-viz: ## Run tests with visual testing feature
	@echo "🎨 Running visual tests..."
	cargo test -p trueno-gpu --features viz -- --nocapture

test-stress: ## Run stress tests with TUI monitor
	@echo "🔥 Running stress tests..."
	cargo test -p trueno-gpu --features stress-test -- stress --nocapture

# ============================================================================
# COVERAGE (Two-Phase Pattern with Mold Hack)
# ============================================================================

coverage: ## Generate coverage report (≥95% required)
	@echo "📊 Running coverage analysis for trueno-gpu (target: <5 min)..."
	@echo "🔍 Checking for cargo-llvm-cov and cargo-nextest..."
	@which cargo-llvm-cov > /dev/null 2>&1 || (echo "📦 Installing cargo-llvm-cov..." && cargo install cargo-llvm-cov --locked)
	@which cargo-nextest > /dev/null 2>&1 || (echo "📦 Installing cargo-nextest..." && cargo install cargo-nextest --locked)
	@echo "🧹 Cleaning old coverage data..."
	@cargo llvm-cov clean --workspace
	@mkdir -p target/coverage
	@echo "⚙️  Temporarily disabling global cargo config (mold breaks coverage)..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@echo "🧪 Phase 1: Running tests with instrumentation (no report)..."
	@env PROPTEST_CASES=100 cargo llvm-cov --no-report --ignore-filename-regex '(benches/|demos/|examples/|tests/|pkg/|test_output/|docs/)' nextest --no-tests=warn -p trueno-gpu --all-features
	@echo "📊 Phase 2: Generating coverage reports..."
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@cargo llvm-cov report --lcov --output-path target/coverage/lcov.info
	@echo "⚙️  Restoring global cargo config..."
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo ""
	@echo "📊 Coverage Summary:"
	@echo "=================="
	@cargo llvm-cov report --summary-only
	@echo ""
	@echo "💡 COVERAGE INSIGHTS:"
	@echo "- HTML report: target/coverage/html/index.html"
	@echo "- LCOV file: target/coverage/lcov.info"
	@echo "- Open HTML: make coverage-open"
	@echo ""

coverage-ci: ## Generate LCOV report for CI/CD (≥95% required)
	@echo "=== Code Coverage for CI/CD (≥95% required) ==="
	@cargo llvm-cov clean --workspace
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@env PROPTEST_CASES=100 cargo llvm-cov --no-report --ignore-filename-regex '(benches/|demos/|examples/|tests/|pkg/|test_output/|docs/)' nextest --no-tests=warn -p trueno-gpu --all-features
	@cargo llvm-cov report --lcov --output-path lcov.info
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo "✓ Coverage report generated: lcov.info"

coverage-check: ## Enforce 95% coverage threshold (BLOCKS on failure)
	@echo "🔒 Enforcing 95% coverage threshold for trueno-gpu..."
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov -p trueno-gpu --all-features --ignore-filename-regex '(benches/|demos/|examples/|tests/|pkg/|test_output/|docs/)' --lcov --output-path lcov.info > /dev/null 2>&1
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@COVERAGE=$$(cargo llvm-cov report --summary-only 2>/dev/null | grep "TOTAL" | awk '{print $$NF}' | sed 's/%//'); \
	echo "trueno-gpu coverage: $${COVERAGE}%"; \
	if [ $$(echo "$$COVERAGE < 95" | bc -l 2>/dev/null || echo 1) -eq 1 ]; then \
		echo "❌ FAIL: Coverage below 95% threshold"; \
		exit 1; \
	else \
		echo "✅ Coverage threshold met (≥95%)"; \
	fi

coverage-clean: ## Clean coverage artifacts
	@cargo llvm-cov clean --workspace
	@rm -f lcov.info target/coverage/lcov.info
	@rm -rf target/llvm-cov target/coverage
	@find . -name "*.profraw" -delete
	@echo "✓ Coverage artifacts cleaned"

clean-coverage: coverage-clean ## Alias for coverage-clean (bashrs pattern)
	@echo "✓ Fresh coverage ready"

# ============================================================================
# BENCHMARKS
# ============================================================================

bench: ## Run all benchmarks
	cargo bench -p trueno-gpu --no-fail-fast

bench-ptx: ## Run PTX generation benchmarks
	@echo "🚀 Running PTX generation benchmarks..."
	cargo bench -p trueno-gpu --bench ptx_gen --no-fail-fast

# ============================================================================
# BUILD & LINT
# ============================================================================

build: ## Build the crate
	cargo build -p trueno-gpu

build-release: ## Build release version
	cargo build -p trueno-gpu --release

build-wasm: ## Build for WebAssembly
	@echo "🌐 Building for WASM..."
	cargo build -p trueno-gpu --target wasm32-unknown-unknown --features wasm

lint: ## Run clippy (zero warnings allowed)
	@echo "🔍 Running clippy for trueno-gpu..."
	cargo clippy -p trueno-gpu --all-features -- -D warnings

fmt: ## Format code
	cargo fmt -p trueno-gpu

fmt-check: ## Check formatting
	cargo fmt -p trueno-gpu -- --check

# ============================================================================
# UTILITIES
# ============================================================================

clean: ## Clean build artifacts
	cargo clean -p trueno-gpu
	rm -f lcov.info
	find . -name "*.profraw" -delete

all: lint test-fast ## Run lint and fast tests
	@echo "✅ All checks passed!"

help: ## Show this help message
	@echo 'trueno-gpu Development Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
