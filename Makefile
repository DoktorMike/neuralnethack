BUILD_DIR := build
COV_DIR := build-coverage
JOBS := $(shell nproc)

.PHONY: all test clean format coverage

all:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) -j$(JOBS)

test: all
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

coverage:
	@cmake -B $(COV_DIR) -DNNH_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
	@cmake --build $(COV_DIR) -j$(JOBS)
	@ctest --test-dir $(COV_DIR) --output-on-failure
	@echo "Coverage data written to $(COV_DIR)/"
	@echo "Run 'lcov -c -d $(COV_DIR) -o coverage.info' to collect, or upload .gcda files to Codecov"

clean:
	@rm -rf $(BUILD_DIR) $(COV_DIR)

format:
	@find neuralnethack src test -name '*.cc' -o -name '*.hh' | xargs clang-format -i
	@echo "Formatted all source files"
