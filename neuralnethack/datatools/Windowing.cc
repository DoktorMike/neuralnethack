#include "Windowing.hh"
#include "Pattern.hh"

#include <stdexcept>
#include <string>
#include <vector>

using namespace DataTools;
using std::vector;

std::shared_ptr<CoreDataSet> DataTools::windowLagged(DataSet& raw, uint channels, uint lags,
                                                     uint passthrough) {
	const uint T = raw.size();
	if (channels == 0 || lags == 0)
		throw std::invalid_argument("windowLagged: channels and lags must be > 0");
	if (T < lags)
		throw std::invalid_argument("windowLagged: need at least `lags` rows, got " +
		                            std::to_string(T));
	if (raw.nInput() != channels + passthrough)
		throw std::invalid_argument(
		    "windowLagged: raw rows must carry channels + passthrough inputs (got " +
		    std::to_string(raw.nInput()) + ", expected " + std::to_string(channels + passthrough) +
		    ")");

	auto core = std::make_shared<CoreDataSet>();
	vector<double> in;
	in.reserve(static_cast<size_t>(channels) * lags + passthrough);
	for (uint t = lags - 1; t < T; ++t) {
		in.clear();
		for (uint c = 0; c < channels; ++c)
			for (uint l = 0; l < lags; ++l)
				in.push_back(raw.pattern(t - l).input()[c]);
		Pattern& now = raw.pattern(t);
		for (uint p = 0; p < passthrough; ++p)
			in.push_back(now.input()[channels + p]);
		vector<double> out = now.output();
		core->addPattern(Pattern(raw.pattern(t).idstring(), in, out));
	}
	return core;
}
