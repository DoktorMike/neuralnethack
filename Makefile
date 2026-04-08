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
	@lcov -c -i -d $(COV_DIR) -o coverage-base.info --ignore-errors mismatch
	@ctest --test-dir $(COV_DIR) --output-on-failure
	@lcov -c -d $(COV_DIR) -o coverage-test.info --ignore-errors mismatch
	@lcov -a coverage-base.info -a coverage-test.info -o coverage.info --ignore-errors mismatch
	@lcov -r coverage.info '/usr/*' '*/test/*' -o coverage.info --ignore-errors mismatch
	@genhtml coverage.info -o coverage-report --ignore-errors mismatch
	@echo "Coverage report: coverage-report/index.html"

clean:
	@rm -rf $(BUILD_DIR) $(COV_DIR)

format:
	@find neuralnethack src test -name '*.cc' -o -name '*.hh' | xargs clang-format -i
	@echo "Formatted all source files"
