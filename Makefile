.PHONY: all build test bench fuzz clean format check-format asan scalar release

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

clean:
	rm -rf build build-*

format:
	find src tests bench -name '*.cpp' -o -name '*.h' | xargs clang-format -i

check-format:
	find src tests bench -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror
