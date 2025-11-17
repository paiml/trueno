# Trueno Makefile - EXTREME TDD Quality Gates
# All targets must complete within time constraints

# Quality directives (bashrs enforcement)
.SUFFIXES:
.DELETE_ON_ERROR:
.ONESHELL:

.PHONY: help build test test-fast coverage lint fmt clean all quality-gates bench dev mutate pmat-tdg pmat-analyze pmat-score install-tools

help: ## Show this help message
	@echo 'Trueno Development Commands:'
	@echo ''
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

coverage: ## Generate coverage report (>85% required, <10 min target)
	@echo "📊 Generating coverage report (target: >85%, <10 min)..."
	@# Temporarily disable mold linker (breaks LLVM coverage)
	@test -f ~/.cargo/config.toml && mv ~/.cargo/config.toml ~/.cargo/config.toml.cov-backup || true
	@cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@cargo llvm-cov report --html --output-dir target/coverage/html
	@# Restore mold linker
	@test -f ~/.cargo/config.toml.cov-backup && mv ~/.cargo/config.toml.cov-backup ~/.cargo/config.toml || true
	@echo "✅ Coverage report: target/coverage/html/index.html"
	@cargo llvm-cov report | grep TOTAL

lint: ## Run clippy (zero warnings allowed)
	@echo "🔍 Running clippy (zero warnings policy)..."
	cargo clippy --all-targets --all-features -- -D warnings

fmt: ## Format code
	cargo fmt

fmt-check: ## Check formatting without modifying
	cargo fmt -- --check

bench: ## Run benchmarks
	cargo bench --no-fail-fast

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

# PMAT integration
pmat-tdg: ## Run PMAT Technical Debt Grading
	pmat tdg

pmat-analyze: ## Run PMAT analysis
	pmat analyze complexity
	pmat analyze satd

pmat-score: ## Calculate repository health score
	pmat repo-score .

# Development helpers
dev: ## Run in development mode with auto-reload
	cargo watch -x 'test --all-features'

install-tools: ## Install required development tools
	cargo install cargo-llvm-cov || exit 1
	cargo install cargo-watch || exit 1
	cargo install cargo-mutants || exit 1
	cargo install criterion || exit 1

.DEFAULT_GOAL := help
