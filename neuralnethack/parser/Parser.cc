#include "Parser.hh"
#include "TomlParser.hh"

#include <algorithm>
#include <iterator>
#include <sstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace std;

template <class T> static string toString(T x) {
	ostringstream oss;
	oss << x;
	return oss.str();
}

void Parser::readDataFile(istream& in, const int idCol, vector<uint> inCols, vector<uint> outCols,
                          vector<uint> rowRange, CoreDataSet& dataSet) {
	checkStream(in);
	vector<string> row;
	string line;

	vector<uint>::iterator validRow = rowRange.begin();
	uint rowCount = 0;
	while (!getline(in, line, '\n').eof()) {
		if (in.fail() || !in.good() || in.bad()) {
			cerr << "Stream failed." << endl;
			break;
		}
		if (++rowCount == *validRow || rowRange.front() == 0) {
			if (*validRow != 0) validRow++;
			row.clear();
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(row));
			if (!row.size()) {
				cerr << "Found empty line." << endl;
				continue;
			}
			selectInserter inp = for_each(inCols.begin(), inCols.end(), selectInserter(row));
			selectInserter outp = for_each(outCols.begin(), outCols.end(), selectInserter(row));
			Pattern p((idCol > 0) ? row[idCol - 1] : toString(rowCount), inp.vec, outp.vec);
			dataSet.addPattern(p);
		}
	}
}

void Parser::readConfigurationFile(istream& in, Config& config) {
	TomlParser::parse(in, config);
}

void Parser::checkStream(istream& in) {
	if (!in) {
		cout << "Parser: Problems with the stream." << endl;
		cout.flush();
		abort();
	}
}
