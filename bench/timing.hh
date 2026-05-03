#pragma once

#include <chrono>
#include <cstdio>

namespace bench {

using clk = std::chrono::steady_clock;

inline double seconds(clk::time_point t0, clk::time_point t1) {
	return std::chrono::duration<double>(t1 - t0).count();
}

// Emit one CSV row to stdout. Header (printed once by run.sh):
//   lib,dataset,arch,epochs,batch,threads,blas,trial,train_s,infer_us,test_acc
inline void emit(const char* lib, const char* dataset, const char* arch, int epochs, int batch,
                 int threads, const char* blas, int trial, double train_s, double infer_us,
                 double acc) {
	std::printf("%s,%s,%s,%d,%d,%d,%s,%d,%.4f,%.3f,%.4f\n", lib, dataset, arch, epochs, batch,
	            threads, blas, trial, train_s, infer_us, acc);
	std::fflush(stdout);
}

} // namespace bench
