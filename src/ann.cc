#include <neuralnethack/mlp/Mlp.hh>
#include <neuralnethack/datatools/DataSet.hh>
#include <neuralnethack/datatools/Normaliser.hh>
#include <neuralnethack/Ensemble.hh>
#include <neuralnethack/parser/NetworkParser.hh>

#include <iostream>
#include <ostream>
#include <istream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <iterator>
#include <algorithm>
#include <ext/algorithm>
#include <functional>

using namespace std;
using namespace NeuralNetHack;
using namespace MultiLayerPerceptron;
using namespace DataTools;

string parseCmdLine(int argc, char* argv[]) {
	if (argc > 1)
		return string(argv[1]);
	else {
		cerr << "Usage: " << argv[0] << " configfile" << endl;
		exit(EXIT_FAILURE);
	}
}

vector<double> getInput(uint numInput, Normaliser* normalisation) {
	vector<double> input(0);
	string s;
	getline(cin, s);
	istringstream ss(s);
	copy(istream_iterator<double>(ss), istream_iterator<double>(), back_inserter(input));
	if (input.size() > 0 && input.size() > numInput) input.resize(numInput);
	normalisation->normaliseInput(input);
	// cout<<"After Normalisation: ";
	// copy(input.begin(), input.end(), ostream_iterator<double>(cout, " "));
	// cout<<endl;
	return input;
}

void killAll(vector<Ensemble*>& ensembles, Ensemble* ensemble, Normaliser* normalisation) {
	for (vector<Ensemble*>::iterator it = ensembles.begin(); it != ensembles.end(); ++it)
		delete *it;
	delete ensemble;
	delete normalisation;
}

int main(int argc, char* argv[]) {
	NetworkParser networkParser;
	string xmlFileName = parseCmdLine(argc, argv);
	ifstream is(xmlFileName.c_str(), ios::in);
	pair<vector<Ensemble*>, Normaliser*> ensAndNorm = networkParser.parseXML(is);
	is.close();
	Ensemble* ensemble = networkParser.buildEnsemble(ensAndNorm.first);
	uint n = ensemble->mlp(0).arch().at(0);
	vector<double> input(n, 0);
	do {
		input = getInput(n, ensAndNorm.second);
		if (input.size() == n) {
			vector<double> output = ensemble->propagate(input);
			cout.precision(20);
			copy(output.begin(), output.end(), ostream_iterator<double>(cout, "\t"));
			cout << endl;
		}
	} while (input.size() == n);
	killAll(ensAndNorm.first, ensemble, ensAndNorm.second);

	return EXIT_SUCCESS;
}
