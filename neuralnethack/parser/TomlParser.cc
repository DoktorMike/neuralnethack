#include "TomlParser.hh"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

namespace NeuralNetHack {

namespace {

// ---- Value: tagged union of the primitive TOML scalars we accept ----
struct Value {
	enum Kind { STR, INT, FLOAT, BOOL, ARRAY };
	Kind kind = STR;
	std::string s;
	long long i = 0;
	double f = 0.0;
	bool b = false;
	std::vector<Value> arr;
};

[[noreturn]] void fail(const std::string& msg, std::size_t lineno) {
	std::ostringstream os;
	os << "TomlParser: " << msg << " (line " << lineno << ")";
	throw std::runtime_error(os.str());
}

void ltrim(std::string& s) {
	std::size_t i = 0;
	while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
		++i;
	s.erase(0, i);
}
void rtrim(std::string& s) {
	while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
		s.pop_back();
}
void trim(std::string& s) {
	rtrim(s);
	ltrim(s);
}

// Strip trailing `# comment`, but only outside of "..." strings.
std::string stripComment(const std::string& line) {
	std::string out;
	bool in_str = false;
	for (std::size_t i = 0; i < line.size(); ++i) {
		char c = line[i];
		if (in_str) {
			out += c;
			if (c == '\\' && i + 1 < line.size()) {
				out += line[++i];
				continue;
			}
			if (c == '"') in_str = false;
		} else {
			if (c == '#') break;
			if (c == '"') in_str = true;
			out += c;
		}
	}
	return out;
}

std::string parseQuotedString(const std::string& s, std::size_t& i, std::size_t lineno) {
	if (i >= s.size() || s[i] != '"') fail("expected '\"'", lineno);
	++i;
	std::string out;
	while (i < s.size() && s[i] != '"') {
		if (s[i] == '\\' && i + 1 < s.size()) {
			char n = s[++i];
			switch (n) {
			case 'n':
				out += '\n';
				break;
			case 't':
				out += '\t';
				break;
			case '"':
				out += '"';
				break;
			case '\\':
				out += '\\';
				break;
			default:
				out += n;
				break;
			}
			++i;
		} else {
			out += s[i++];
		}
	}
	if (i >= s.size()) fail("unterminated string", lineno);
	++i; // closing "
	return out;
}

void skipSpace(const std::string& s, std::size_t& i) {
	while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
		++i;
}

Value parseScalar(const std::string& s, std::size_t& i, std::size_t lineno);

Value parseArray(const std::string& s, std::size_t& i, std::size_t lineno) {
	Value v;
	v.kind = Value::ARRAY;
	if (s[i] != '[') fail("expected '['", lineno);
	++i;
	skipSpace(s, i);
	while (i < s.size() && s[i] != ']') {
		v.arr.push_back(parseScalar(s, i, lineno));
		skipSpace(s, i);
		if (i < s.size() && s[i] == ',') {
			++i;
			skipSpace(s, i);
		}
	}
	if (i >= s.size()) fail("unterminated array", lineno);
	++i; // ]
	return v;
}

Value parseScalar(const std::string& s, std::size_t& i, std::size_t lineno) {
	skipSpace(s, i);
	if (i >= s.size()) fail("expected value", lineno);
	Value v;
	if (s[i] == '"') {
		v.kind = Value::STR;
		v.s = parseQuotedString(s, i, lineno);
		return v;
	}
	if (s[i] == '[') return parseArray(s, i, lineno);
	// bool / number / bare keyword
	std::size_t start = i;
	while (i < s.size() && s[i] != ',' && s[i] != ']' &&
	       !std::isspace(static_cast<unsigned char>(s[i])))
		++i;
	std::string tok = s.substr(start, i - start);
	if (tok == "true" || tok == "false") {
		v.kind = Value::BOOL;
		v.b = (tok == "true");
		return v;
	}
	// Decide int vs float: presence of '.', 'e', 'E' → float
	bool is_float = tok.find_first_of(".eE") != std::string::npos;
	char* end = nullptr;
	if (is_float) {
		v.kind = Value::FLOAT;
		v.f = std::strtod(tok.c_str(), &end);
		if (end == tok.c_str()) fail("bad number: '" + tok + "'", lineno);
	} else {
		v.kind = Value::INT;
		v.i = std::strtoll(tok.c_str(), &end, 10);
		if (end == tok.c_str()) fail("bad integer: '" + tok + "'", lineno);
	}
	return v;
}

double asNumber(const Value& v, const std::string& path, std::size_t lineno) {
	if (v.kind == Value::INT) return static_cast<double>(v.i);
	if (v.kind == Value::FLOAT) return v.f;
	fail("expected number for '" + path + "'", lineno);
}
long long asInt(const Value& v, const std::string& path, std::size_t lineno) {
	if (v.kind == Value::INT) return v.i;
	if (v.kind == Value::FLOAT) return static_cast<long long>(v.f);
	fail("expected integer for '" + path + "'", lineno);
}
bool asBool(const Value& v, const std::string& path, std::size_t lineno) {
	if (v.kind == Value::BOOL) return v.b;
	if (v.kind == Value::INT) return v.i != 0;
	fail("expected bool for '" + path + "'", lineno);
}
const std::string& asString(const Value& v, const std::string& path, std::size_t lineno) {
	if (v.kind == Value::STR) return v.s;
	fail("expected string for '" + path + "'", lineno);
}

// Pending [[model_selection.vary]] row buffer.
struct VaryEntry {
	std::string parameter;
	long long subparam = 0;
	double start = 0.0, stop = 0.0, step = 0.0;
	bool any_set = false;
};

void flushVary(VaryEntry& ve, Config& config) {
	if (!ve.any_set) return;
	auto& vary = config.vary();
	vary[ve.parameter] = {ve.start, ve.stop, ve.step};
	ve = VaryEntry{};
}

// Apply a single (path, value) pair to the Config.
void apply(const std::string& path, const Value& v, Config& config, VaryEntry& ve, bool in_vary_row,
           std::size_t lineno) {
	auto setRange = [&](void (Config::*setter)(const std::vector<uint>&)) {
		(config.*setter)(TomlParser::expandRange(asString(v, path, lineno)));
	};
	auto setSplit = [&](void (Config::*setter)(const bool&)) {
		const auto& s = asString(v, path, lineno);
		(config.*setter)(s == "rnd");
	};

	if (in_vary_row) {
		ve.any_set = true;
		if (path == "parameter")
			ve.parameter = asString(v, path, lineno);
		else if (path == "subparam")
			ve.subparam = asInt(v, path, lineno);
		else if (path == "start")
			ve.start = asNumber(v, path, lineno);
		else if (path == "stop")
			ve.stop = asNumber(v, path, lineno);
		else if (path == "step")
			ve.step = asNumber(v, path, lineno);
		else
			fail("unknown key in [[model_selection.vary]]: '" + path + "'", lineno);
		return;
	}

	if (path == "suffix")
		config.suffix(asString(v, path, lineno));
	else if (path == "seed")
		config.seed(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "normalization")
		config.normalization(asString(v, path, lineno));
	else if (path == "problem_type") {
		const auto& s = asString(v, path, lineno);
		// false = classification, true = regression (matches legacy semantics).
		config.problemType(s != "class");
	} else if (path == "data.train.file")
		config.fileName(asString(v, path, lineno));
	else if (path == "data.train.id_col")
		config.idColumn(static_cast<int>(asInt(v, path, lineno)));
	else if (path == "data.train.in_cols")
		setRange(&Config::inputColumns);
	else if (path == "data.train.out_cols")
		setRange(&Config::outputColumns);
	else if (path == "data.train.row_range")
		setRange(&Config::rowRange);
	else if (path == "data.test.file")
		config.fileNameT(asString(v, path, lineno));
	else if (path == "data.test.id_col")
		config.idColumnT(static_cast<int>(asInt(v, path, lineno)));
	else if (path == "data.test.in_cols")
		setRange(&Config::inputColumnsT);
	else if (path == "data.test.out_cols")
		setRange(&Config::outputColumnsT);
	else if (path == "data.test.row_range")
		setRange(&Config::rowRangeT);
	else if (path == "network.size") {
		if (v.kind != Value::ARRAY) fail("network.size must be an array", lineno);
		std::vector<uint> a;
		for (const auto& e : v.arr)
			a.push_back(static_cast<uint>(asInt(e, path, lineno)));
		config.architecture(a);
	} else if (path == "network.activations") {
		if (v.kind != Value::ARRAY) fail("network.activations must be an array", lineno);
		std::vector<std::string> a;
		for (const auto& e : v.arr)
			a.push_back(asString(e, path, lineno));
		config.actFcn(a);
	} else if (path == "network.error_fcn")
		config.errFcn(asString(v, path, lineno));
	else if (path == "network.skip_connections") {
		if (v.kind != Value::ARRAY) fail("network.skip_connections must be an array", lineno);
		auto& sc = config.skipConnections();
		sc.clear();
		for (const auto& pair : v.arr) {
			if (pair.kind != Value::ARRAY || pair.arr.size() != 2)
				fail("network.skip_connections entries must be [target, source] pairs", lineno);
			int target = static_cast<int>(asInt(pair.arr[0], path, lineno));
			int source = static_cast<int>(asInt(pair.arr[1], path, lineno));
			sc.emplace_back(target, source);
		}
	} else if (path == "network.softmax")
		config.softmax(asBool(v, path, lineno));
	else if (path == "training.method")
		config.minMethod(asString(v, path, lineno));
	else if (path == "training.max_epochs")
		config.maxEpochs(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "training.gd.batch_size")
		config.batchSize(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "training.gd.learning_rate")
		config.learningRate(asNumber(v, path, lineno));
	else if (path == "training.gd.lr_decay")
		config.decLearningRate(asNumber(v, path, lineno));
	else if (path == "training.gd.momentum")
		config.momentum(asNumber(v, path, lineno));
	else if (path == "training.adam.learning_rate")
		config.adamLearningRate(asNumber(v, path, lineno));
	else if (path == "training.adam.beta1")
		config.adamBeta1(asNumber(v, path, lineno));
	else if (path == "training.adam.beta2")
		config.adamBeta2(asNumber(v, path, lineno));
	else if (path == "training.adam.epsilon")
		config.adamEpsilon(asNumber(v, path, lineno));
	else if (path == "training.adam.weight_decay")
		config.adamWeightDecay(asNumber(v, path, lineno));
	else if (path == "regularization.weight_elim.enabled")
		config.weightElimOn(asBool(v, path, lineno));
	else if (path == "regularization.weight_elim.alpha")
		config.weightElimAlpha(asNumber(v, path, lineno));
	else if (path == "regularization.weight_elim.w0")
		config.weightElimW0(asNumber(v, path, lineno));
	else if (path == "ensemble.method")
		config.ensParamDataSelection(asString(v, path, lineno));
	else if (path == "ensemble.runs")
		config.ensParamN(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "ensemble.parts")
		config.ensParamK(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "ensemble.split")
		setSplit(&Config::ensParamSplitMode);
	else if (path == "ensemble.vary_weights")
		config.ensParamNewWeights(asBool(v, path, lineno));
	else if (path == "model_selection.method")
		config.msParamDataSelection(asString(v, path, lineno));
	else if (path == "model_selection.runs")
		config.msParamN(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "model_selection.parts")
		config.msParamK(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "model_selection.split")
		setSplit(&Config::msParamSplitMode);
	else if (path == "model_selection.fraction")
		config.msParamNumTrainingData(asNumber(v, path, lineno));
	else if (path == "model_selection.msg.runs")
		config.msgParamN(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "model_selection.msg.parts")
		config.msgParamK(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "model_selection.msg.split")
		setSplit(&Config::msgParamSplitMode);
	else if (path == "model_selection.msg.fraction")
		config.msgParamNumTrainingData(asNumber(v, path, lineno));
	else if (path == "output.save_session")
		config.saveSession(asBool(v, path, lineno));
	else if (path == "output.save_output_list")
		config.saveOutputList(asBool(v, path, lineno));
	else if (path == "output.info")
		config.info(static_cast<uint>(asInt(v, path, lineno)));
	else if (path == "output.learning_curve_file")
		config.learningCurveFile(asString(v, path, lineno));
	else
		fail("unknown key '" + path + "'", lineno);
}

} // namespace

std::vector<uint> TomlParser::expandRange(const std::string& s) {
	// Accepts "0", "1-8", "1,3,5", "1-3,5,7-9".
	// "0" is the historical sentinel for "all" (in row_range fields).
	std::vector<uint> out;
	std::size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
			++i;
		if (i >= s.size()) break;
		std::size_t a_start = i;
		while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
			++i;
		if (a_start == i) {
			// Skip stray separator.
			++i;
			continue;
		}
		uint a = static_cast<uint>(std::stoul(s.substr(a_start, i - a_start)));
		if (i < s.size() && s[i] == '-') {
			++i;
			std::size_t b_start = i;
			while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
				++i;
			uint b = static_cast<uint>(std::stoul(s.substr(b_start, i - b_start)));
			for (uint k = a; k <= b; ++k)
				out.push_back(k);
		} else {
			out.push_back(a);
		}
		while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\t'))
			++i;
	}
	if (out.empty()) out.push_back(0);
	return out;
}

void TomlParser::parse(std::istream& in, Config& config) {
	std::string current_section;
	bool in_vary_row = false;
	VaryEntry ve;

	std::string raw;
	std::size_t lineno = 0;
	while (std::getline(in, raw)) {
		++lineno;
		std::string line = stripComment(raw);
		trim(line);
		if (line.empty()) continue;

		if (line.front() == '[') {
			// Flush any pending vary row before changing section.
			if (in_vary_row) flushVary(ve, config);

			bool array_of_tables = (line.size() >= 2 && line[1] == '[');
			std::size_t open = array_of_tables ? 2 : 1;
			std::size_t close = line.find(array_of_tables ? "]]" : "]", open);
			if (close == std::string::npos) fail("unterminated section header", lineno);
			std::string name = line.substr(open, close - open);
			trim(name);

			if (array_of_tables) {
				if (name != "model_selection.vary")
					fail("array-of-tables only supported at model_selection.vary", lineno);
				in_vary_row = true;
				current_section = "";
			} else {
				in_vary_row = false;
				current_section = name;
			}
			continue;
		}

		// key = value
		std::size_t eq = line.find('=');
		if (eq == std::string::npos) fail("expected '=' in '" + line + "'", lineno);
		std::string key = line.substr(0, eq);
		std::string rest = line.substr(eq + 1);
		trim(key);
		std::size_t i = 0;
		Value v = parseScalar(rest, i, lineno);

		std::string path;
		if (in_vary_row)
			path = key;
		else if (current_section.empty())
			path = key;
		else
			path = current_section + "." + key;

		apply(path, v, config, ve, in_vary_row, lineno);
	}
	if (in_vary_row) flushVary(ve, config);
}

void TomlParser::parseFile(const std::string& path, Config& config) {
	std::ifstream in(path);
	if (!in) {
		std::cerr << "TomlParser: could not open '" << path << "'" << std::endl;
		std::abort();
	}
	parse(in, config);
}

} // namespace NeuralNetHack
