.PHONY: all build test bench bench-regress bench-json fuzz clean format check-format asan scalar release

BUILD_DIR ?= build

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j

release:
	cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
	cmake --build build-release -j

asan:
	cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCOLUMNSTORE_ENABLE_ASAN=ON
	cmake --build build-asan -j

scalar:
	cmake -S . -B build-scalar -DCMAKE_BUILD_TYPE=Release -DCOLUMNSTORE_FORCE_SCALAR=ON
	cmake --build build-scalar -j

fuzz:
	cmake -S . -B build-fuzz -DCMAKE_BUILD_TYPE=Debug -DCOLUMNSTORE_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
	cmake --build build-fuzz -j

test: build
	cd $(BUILD_DIR) && ctest --output-on-failure

bench: build
	$(BUILD_DIR)/scan_bench
	$(BUILD_DIR)/filter_bench
	$(BUILD_DIR)/aggregate_bench

# Emit a JSON-lines candidate file by running every bench across 1M, 10M, 100M
# rows. Set COLUMNSTORE_BENCH_JSON_OUT and COLUMNSTORE_BENCH_LABEL externally
# to control the output file.
bench-json: build
	@echo "writing JSON to $${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl}"
	@rm -f $${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl}
	@for n in 1000000 10000000 100000000; do \
		COLUMNSTORE_BENCH_ROWS=$$n COLUMNSTORE_BENCH_ITERS=5 \
		COLUMNSTORE_BENCH_JSON_OUT=$${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl} \
		$(BUILD_DIR)/filter_bench > /dev/null; \
		COLUMNSTORE_BENCH_ROWS=$$n COLUMNSTORE_BENCH_ITERS=5 \
		COLUMNSTORE_BENCH_JSON_OUT=$${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl} \
		$(BUILD_DIR)/aggregate_bench > /dev/null; \
		COLUMNSTORE_BENCH_ROWS=$$n COLUMNSTORE_BENCH_ITERS=5 \
		COLUMNSTORE_BENCH_JSON_OUT=$${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl} \
		$(BUILD_DIR)/scan_bench > /dev/null; \
	done
	@wc -l $${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl}

# Compare a freshly-generated candidate to the committed baseline. Fails if
# any operator regresses throughput or p99 latency by more than 30%.
bench-regress: bench-json
	@python3 bench/regress.py \
		--baseline bench/results/baseline.jsonl \
		--candidate $${COLUMNSTORE_BENCH_JSON_OUT:-/tmp/bench_candidate.jsonl} \
		--max-drift 0.30

clean:
	rm -rf build build-*

format:
	find src tests bench -name '*.cpp' -o -name '*.h' | xargs clang-format -i

check-format:
	find src tests bench -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror
