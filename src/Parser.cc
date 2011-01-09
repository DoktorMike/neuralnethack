#include "Parser_impl.hh"

#include <ctype.h>

using namespace NetHack;

void Parser::readDataFile(ifstream& in, vector<uint> inCols,
	vector<uint> outCols, DataSet& dataSet)
{
    checkStream(in);
    vector<double> row(0);
    vector<double> inp(0);
    vector<double> outp(0);
    vector<uint>::iterator it;
    
    while(!in.eof() && !in.fail() && in.good()){
	row = readRow(in);
	if(!row.size())
	    break;
	inp.clear();
	for(it=inCols.begin(); it!=inCols.end(); ++it){
	    uint index = (*it)-1;
	    double value = row.at(index);
	    inp.push_back(value);
	}

	outp.clear();
	for(it=outCols.begin(); it!=outCols.end(); ++it){
	    outp.push_back(row.at((*it)-1));
	}

	Pattern p(inp, outp);
	dataSet.addPattern(p);
    }
}

void Parser::readDataFile(ifstream& in, const int nInput, 
	const int nOutput, DataSet& dataSet)
{
    checkStream(in);

    vector<double> input(nInput);
    vector<double> output(nOutput);
    while(!in.eof()){
	bool bad = false;
	for(int j=0; j<nInput; j++){
	    if(in.good())
		in >> input[j];	
	    else 
		bad=true;
	}
	for(int j=0; j<nOutput; j++){
	    if(in.good())
		in >> output[j];	
	    else
		bad=true;
	}
	if(!bad){
	    Pattern p(input, output);
	    dataSet.addPattern(p);
	}
	else
	    break;  
	char tmp[255]; 
	in.getline(tmp, 255); //Need this for crappy txt files.
    }
}

void Parser::readConfigurationFile(ifstream& in, Config& config)
{
    string token;
    checkStream(in);

    while(!in.eof()){
	whitespace(in);
	if(in.peek() == '%')
	    comment(in);
	else{
	    in>>token;
	    if(in.good()){
		//cout<<"Processing Token: "<<token<<endl;
		if(token == "FileName")
		    config.filename(parseFileName(in)); 
		else if(token == "InCol")
		    config.inputColumns(parseCol(in)); 
		else if(token == "OutCol")
		    config.outputColumns(parseCol(in)); 
		else if(token == "NLay")
		    config.nLayers(parseNLay(in)); 
		else if(token == "Size")
		    config.arch(parseSize(in)); 
		else if(token == "ActFcn")
		    config.activationFunctions(parseActFcn(in)); 
		else if(token == "ErrFcn")
		    config.errorFunction(parseErrFcn(in)); 
		else if(token == "MinMethod")
		    config.learningAlgorithm(parseMinMethod(in)); 
		else if(token == "MaxEpochs")
		    config.nEpochs(parseMaxEpochs(in)); 
		else if(token == "GDParam")
		    parseGDParam(in, config); 
		token = "";
	    }
	}
    }
}

//PRIVATE--------------------------------------------------------------------//

void Parser::checkStream(ifstream& in){
    if(!in){
	cout<<"Parser: Problems with the stream."<<endl;
	cout.flush();
	abort();
    }
}

void Parser::whitespace(ifstream& in)
{
    int c = in.peek();
    while(iscntrl(c) || isspace(c)){
	in.get();
	c = in.peek();
    }
}

void Parser::comment(ifstream& in)
{
    //cout<<"Entering comment with char: "<<(char)in.peek()<<endl;
    in.ignore(255,10);
    //cout<<"Leaving comment with char: "<<(char)in.peek()<<endl;
}

string Parser::parseFileName(ifstream& in)
{
    string fname;
    in>>fname;
    return fname;
}

vector<uint> Parser::parseCol(ifstream& in)
{
    //cout<<"Entering parseCol with char: "<<(char)in.peek()<<endl;
    uint c, buf;
    vector<uint> operands(0), operators(0), v(0);
    vector<uint>::iterator opnd, op;

    whitespace(in);

    c=in.peek();
    while(!iscntrl(c) && c!=37){
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
	if(*op==44)
	    v.push_back(*opnd);
	else if(*op==45)
	    for(uint i=*(opnd-1)+1; i<=*opnd; ++i)
		v.push_back(i);

    //cout<<"Leaving parseInCol with char: "<<(char)in.peek()<<endl;

    return v;
}

uint Parser::parseNLay(ifstream& in)
{
    uint nl;
    in>>nl;
    return nl;
}

vector<uint> Parser::parseSize(ifstream& in)
{
    vector<uint> v(0);
    uint buf,c;

    c=in.peek();
    while(!iscntrl(c) && c!=37){
	in>>buf;
	v.push_back(buf);
	c=in.peek();
    }
    return v;
}

vector<string> Parser::parseActFcn(ifstream& in)
{
    vector<string> v(0);
    string buf;

    whitespace(in);

    uint c=in.peek();
    while(!iscntrl(c) && c!=37){
	in>>buf;
	v.push_back(buf);
	whitespace(in);
	c=in.peek();
    }

    return v;
}

string Parser::parseErrFcn(ifstream& in)
{
    string buf;
    in>>buf;
    return buf;
}

string Parser::parseMinMethod(ifstream& in)
{
    string buf;
    in>>buf;
    return buf;
}

uint Parser::parseMaxEpochs(ifstream& in)
{
    uint e;
    in>>e;
    return e;
}

void Parser::parseGDParam(ifstream& in, Config& config)
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

vector<double> Parser::readRow(ifstream& in)
{
    vector<double> vec(0);
    string s;
    getline(in, s);
    //cout<<s<<endl;
    uint i1, i2;
    i1 = 0;
    while(i1<s.size()){
	while(i1<s.size() && isspace(s.at(i1)))
	    ++i1;
	if(i1<s.size())
	    i2=i1;
	else
	    break;
	while(i2<s.size() && !isspace(s.at(i2)))
	    ++i2;
	string tmp = s.substr(i1, i2-i1);
	vec.push_back(atof(tmp.c_str()));
	i1=i2;
    }
    return vec;

    /*vector<double> row(0);
      double tmp;
      whitespace(in); 
      char c=in.peek();

      while(isdigit(c)){
      in>>tmp;
      row.push_back(tmp);
      tabspace(in);
      c=in.peek();
      }
      whitespace(in); 
      return row;*/
}

void Parser::tabspace(ifstream& in)
{
    char c = in.peek();
    while(c=='\t' || c==' '){
	in.get();
	c=in.peek();
    }
}

