/*$Id: Parser.cc 1623 2007-05-08 08:30:14Z michael $*/

/*
  Copyright (C) 2004 Michael Green

  neuralnethack is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  Michael Green <michael@thep.lu.se>
*/


#include "Parser.hh"

#include <fstream>
#include <cctype>
#include <algorithm>
#include <iterator>
#include <functional>
#include <sstream>

using namespace NeuralNetHack;
using namespace DataTools;
using namespace std;

string toString(int x){
	ostringstream oss;
	oss<<x;
	return oss.str();
}

void Parser::readDataFile(istream& in, const int idCol, vector<uint> inCols,
		vector<uint> outCols, vector<uint> rowRange, CoreDataSet& dataSet)
{
	checkStream(in);
	vector<string> row;
	string line;

	uint rowCount = 1;
	while(!getline(in, line, '\n').eof()){
		if(in.fail() || !in.good()){ cerr<<"Stream failed."<<endl;	break; }
		uint rowNum = rowRange[rowCount-1];
		if(rowNum == rowCount){
			row.clear();
			istringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(row));
			if(!row.size()){ cerr<<"Found empty line."<<endl; continue; }
			selectInserter inp = for_each(inCols.begin(), inCols.end(), selectInserter(row));
			selectInserter outp = for_each(outCols.begin(), outCols.end(), selectInserter(row));
			Pattern p((idCol > 0) ? row[idCol-1] : toString(rowCount), inp.vec, outp.vec);
			dataSet.addPattern(p);
		}
		rowCount++;
	}
}

void Parser::readDataFile(istream& in, const int nInput, 
		const int nOutput, CoreDataSet& dataSet)
{
	checkStream(in);

	vector<double> input(nInput);
	vector<double> output(nOutput);
	while(!in.eof()){
		bool bad = false;
		for(int j=0; j<nInput; j++){
			if(in.good()) in >> input[j];	
			else bad=true;
		}
		for(int j=0; j<nOutput; j++){
			if(in.good()) in >> output[j];	
			else bad=true;
		}
		if(!bad){
			Pattern p("", input, output);
			dataSet.addPattern(p);
		}
		else
			break;
		char tmp[255]; 
		in.getline(tmp, 255); //Need this for crappy txt files.
	}
}

void Parser::readConfigurationFile(istream& in, Config& config)
{
	string token, buf;
	uint uintbuf;
	checkStream(in);

	while(!in.eof()){
		whitespace(in);
		if(in.peek() == '%')
			comment(in);
		else{
			in>>token;
			transform(token.begin(), token.end(), token.begin(), (int(*)(int)) tolower);
			if(in.good()){
				//cout<<"Processing Token: "<<token<<endl;
				if(token == "suffix"){ in>>buf; config.suffix(buf); }
				else if(token == "filename"){ in>>buf; config.fileName(buf); }
				else if(token == "idcol"){ in>>uintbuf; config.idColumn(uintbuf); }
				else if(token == "incol") config.inputColumns(parseCol(in)); 
				else if(token == "outcol") config.outputColumns(parseCol(in)); 
				else if(token == "rowrange") config.rowRange(parseCol(in)); 
				else if(token == "filenamet") config.fileNameT(parseString(in)); 
				else if(token == "idcolt"){ in>>uintbuf; config.idColumnT(uintbuf); }
				else if(token == "incolt") config.inputColumnsT(parseCol(in)); 
				else if(token == "outcolt") config.outputColumnsT(parseCol(in)); 
				else if(token == "rowranget") config.rowRangeT(parseCol(in)); 
				else if(token == "ptype") config.problemType(parsePType(in)); 
				else if(token == "nlay") config.numLayers(parseNLay(in)); 
				else if(token == "size") config.architecture(parseSize(in, config.numLayers())); 
				else if(token == "actfcn") parseActFcn(in, config); 
				else if(token == "errfcn") parseErrFcn(in, config); 
				else if(token == "minmethod") parseMinMethod(in, config); 
				else if(token == "maxepochs") parseMaxEpochs(in, config); 
				else if(token == "gdparam") parseGDParam(in, config); 
				else if(token == "weightelim") parseWeightElim(in, config); 
				else if(token == "ensparam") parseEnsParam(in, config); 
				else if(token == "msparam") parseMSParam(in, config); 
				else if(token == "msgparam") parseMSGParam(in, config); 
				else if(token == "savesession") { uint buf; in>>buf; config.saveSession(buf); }
				else if(token == "info") { uint buf; in>>buf; config.info(buf); }
				else if(token == "saveoutputlist") { uint buf; in>>buf; config.saveOutputList(buf); }
				else if(token == "seed") { uint buf; in>>buf; config.seed(buf); }
				else if(token == "normalization") { string buf; in>>buf; config.normalization(buf); }
				else if(token == "vary") { parseVary(in, config); }
				token = "";
			}
		}
	}
}

vector<uint> Parser::parseColumns(istream& in){ return parseCol(in); }

//PRIVATE--------------------------------------------------------------------//

void Parser::checkStream(istream& in){
	if(!in){
		cout<<"Parser: Problems with the stream."<<endl;
		cout.flush();
		abort();
	}
}

void Parser::whitespace(istream& in)
{
	int c = in.peek();
	while(iscntrl(c) || isspace(c)){
		in.get();
		c = in.peek();
	}
}

void Parser::comment(istream& in)
{
	//cout<<"Entering comment with char: "<<(char)in.peek()<<endl;
	in.ignore(255,10);
	//cout<<"Leaving comment with char: "<<(char)in.peek()<<endl;
}

string Parser::parseString(istream& in)
{
	string fname;
	in>>fname;
	//cout<<"Parser::parseString: fname = "<<fname<<endl;
	return fname;
}

vector<uint> Parser::parseCol(istream& in)
{
	//cout<<"Entering parseCol with char: "<<(char)in.peek()<<endl;
	uint c, buf;
	vector<uint> operands(0), operators(0), v(0);
	vector<uint>::iterator opnd, op;

	whitespace(in);

	c=in.peek();
	while(!iscntrl(c) && c!=37 && in.good()){
		switch(c){
			case 44: operators.push_back(in.get());
					 break;
			case 45: operators.push_back(in.get());
					 break;
			case 48:case 49:case 50:case 51:
			case 52:case 53:case 54:case 55:
			case 56:case 57: in>>buf;
							 operands.push_back(buf);
							 break;
			default: in.get();
					 break;
		}
		c=in.peek();
	}

	opnd=operands.begin()+1;
	v.push_back(*(opnd-1));
	for(op=operators.begin(); op!=operators.end(); ++op, ++opnd)
		if(*op==44) v.push_back(*opnd);
		else if(*op==45) for(uint i=*(opnd-1)+1; i<=*opnd; ++i) v.push_back(i);

	//cout<<"Leaving parseInCol with char: "<<(char)in.peek()<<endl;

	return v;
}

bool Parser::parsePType(istream& in)
{
	string buf;
	in>>buf;
	return buf == "class" ? false : true;
}

uint Parser::parseNLay(istream& in)
{
	uint nl;
	in>>nl;
	return nl;
}

vector<uint> Parser::parseSize(istream& in, uint nLayers)
{
	vector<uint> v(nLayers);
	uint i=0;

	while(i<nLayers && in>>v[i]) ++i;
	in.clear();
	return v;
}

void Parser::parseActFcn(istream& in, Config& config)
{
	vector<string> v(config.numLayers()-1);
	uint i=1;

	while(i<config.numLayers() && in>>v[i-1]) ++i;
	in.clear();
	config.actFcn(v);
}

void Parser::parseErrFcn(istream& in, Config& config)
{
	string buf;
	in>>buf;
	config.errFcn(buf);
}

void Parser::parseMinMethod(istream& in, Config& config)
{
	string buf;
	in>>buf;
	config.minMethod(buf);
}

void Parser::parseMaxEpochs(istream& in, Config& config)
{
	uint e;
	in>>e;
	config.maxEpochs(e);
}

void Parser::parseGDParam(istream& in, Config& config)
{
	double buf;
	in>>buf;
	config.batchSize((uint)buf);
	in>>buf;
	config.learningRate(buf);
	in>>buf;
	config.decLearningRate(buf);
	in>>buf;
	config.momentum(buf);
}

void Parser::parseMSParam(istream& in, Config& config)
{
	uint bufi;
	double buff;
	string bufs;
	in>>bufs; config.msParamDataSelection(bufs);
	in>>bufi; config.msParamN(bufi);
	in>>bufi; config.msParamK(bufi);
	string mode = "";
	in>>mode;
	config.msParamSplitMode((mode == "rnd") ? true : false);
	in>>buff; config.msParamNumTrainingData(buff);
}

void Parser::parseEnsParam(istream& in, Config& config)
{
	uint bufi;
	string bufs;
	in>>bufs; config.ensParamDataSelection(bufs);
	in>>bufi; config.ensParamN(bufi);
	in>>bufi; config.ensParamK(bufi);
	string mode = "";
	in>>mode;
	config.ensParamSplitMode((mode == "rnd") ? true : false);
	in>>bufi; config.ensParamNewWeights(bufi);
}

void Parser::parseMSGParam(istream& in, Config& config)
{
	uint buf;
	in>>buf;
	config.msgParamN(buf);
	in>>buf;
	config.msgParamK(buf);
	string mode = "";
	in>>mode;
	config.msgParamSplitMode((mode == "rnd") ? true : false);
	double buf2;
	in>>buf2;
	config.msgParamNumTrainingData(buf2);
}

void Parser::parseVary(istream& in, Config& config)
{
	string strbuf;
	uint uintbuf;
	in>>strbuf;
	in>>uintbuf;
	if(strbuf == "WeightElim" && uintbuf == 2){
		vector<double> tmp(3, 0);
		in>>tmp[0]; in>>tmp[1]; in>>tmp[2]; 
		map<string, vector<double> >& vary = config.vary();
		vary["WeightElim"] = tmp;
	}else if(strbuf == "Size"){
		vector<double> tmp(3, 0);
		in>>tmp[0]; in>>tmp[1]; in>>tmp[2]; 
		map<string, vector<double> >& vary = config.vary();
		vary["Size"] = tmp;
	}else{
		cerr<<"Warning: Vary only handles WeightElim at the moment."<<endl;
		abort();
	}
}

void Parser::parseWeightElim(istream& in, Config& config)
{
	uint on;
	in>>on;
	config.weightElimOn((on == 1) ? true : false);
	double buf;
	in>>buf;
	config.weightElimAlpha(buf);
	in>>buf;
	config.weightElimW0(buf);
}

vector<double> Parser::readRow(istream& in)
{
	vector<double> vec(0);
	string s;
	getline(in, s);
	istringstream iss(s);
	copy(istream_iterator<double>(iss), istream_iterator<double>(), back_inserter(vec));
	cout<<s<<endl;
	/*
	   uint i1, i2;
	   i1 = 0;
	   while(i1<s.size()){
	   while(i1<s.size() && isspace(s.at(i1))) ++i1;
	   if(i1<s.size()) i2=i1; else break;
	   while(i2<s.size() && !isspace(s.at(i2))) ++i2;
	   string tmp = s.substr(i1, i2-i1);
	   vec.push_back(atof(tmp.c_str()));
	   i1=i2;
	   }
	   */
	return vec;
}

void Parser::tabspace(istream& in)
{
	char c = in.peek();
	while(c=='\t' || c==' '){
		in.get();
		c=in.peek();
	}
}

