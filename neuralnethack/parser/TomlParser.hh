#ifndef __TomlParser_hh__
#define __TomlParser_hh__

#include "../Config.hh"

#include <istream>
#include <string>
#include <vector>

namespace NeuralNetHack {

// Parses a TOML configuration file into a Config.
//
// Supported subset:
//   - top-level key = value
//   - [section] and dotted [a.b.c] sections
//   - [[name]] array-of-tables rows
//   - basic strings ("..."), integers, floats, booleans, single-line arrays
//   - # comments and blank lines
//
// Range strings ("1-8", "1,3-5,7") are accepted for column / row fields.
class TomlParser {
  public:
	static void parse(std::istream& in, Config& config);
	static void parseFile(const std::string& path, Config& config);

	// Expand a range string like "1-3,5,7-9" into [1,2,3,5,7,8,9].
	// "0" expands to [0] (sentinel meaning "all rows" for row_range fields).
	static std::vector<uint> expandRange(const std::string& s);
};

} // namespace NeuralNetHack

#endif
