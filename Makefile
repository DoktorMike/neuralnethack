BUILD_DIR := build
COV_DIR := build-coverage
JOBS := $(shell nproc)

.PHONY: all test examples clean format coverage

all:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) -j$(JOBS)

test: all
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

examples:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) -j$(JOBS) --target nnh_examples

coverage:
	@cmake -B $(COV_DIR) -DNNH_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
	@cmake --build $(COV_DIR) -j$(JOBS)
	@lcov -c -i -d $(COV_DIR) -o coverage-base.info --ignore-errors mismatch,unused
	@ctest --test-dir $(COV_DIR) --output-on-failure
	@lcov -c -d $(COV_DIR) -o coverage-test.info --ignore-errors mismatch,unused
	@lcov -a coverage-base.info -a coverage-test.info -o coverage.info --ignore-errors mismatch,unused
	@lcov -r coverage.info '/usr/*' '*/neuralnethack/test/*' '*/neuralnethack/src/*' '*/neuralnethack/examples/*' -o coverage.info --ignore-errors mismatch,unused
	@genhtml coverage.info -o coverage-report --ignore-errors mismatch,unused
	@echo "Coverage report: coverage-report/index.html"

clean:
	@rm -rf $(BUILD_DIR) $(COV_DIR)

format:
	@find neuralnethack src test -name '*.cc' -o -name '*.hh' | xargs clang-format -i
	@echo "Formatted all source files"
