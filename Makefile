BUILD_DIR := build
JOBS := $(shell nproc)

.PHONY: all test clean format

all:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) -j$(JOBS)

test: all
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	@rm -rf $(BUILD_DIR)

format:
	@find neuralnethack src test -name '*.cc' -o -name '*.hh' | xargs clang-format -i
	@echo "Formatted all source files"
