#include "Pattern.hh"

using namespace DataTools;

	Pattern::Pattern(vector<double>& in, vector<double>& out):
in(in), out(out)
{}

Pattern::Pattern(){}

Pattern::Pattern(const Pattern& pattern)
{*this = pattern;}

Pattern::~Pattern(){}

Pattern& Pattern::operator=(const Pattern& pattern)
{
	if(this != &pattern){
		this->in = pattern.in;
		this->out = pattern.out;
	}
	return *this;
}

vector<double>& Pattern::input()
{return in;}

void Pattern::input(vector<double>& in)
{this->in = in;}

uint Pattern::nInput() const 
{return in.size();}

vector<double>& Pattern::output()
{return out;}

void Pattern::output(vector<double>& out)
{this->out = out;}

uint Pattern::nOutput() const
{return out.size();}

void Pattern::print(ostream& os)
{
	assert(os);
	vector<double>::iterator it;
	os.setf(ios::ios_base::fixed, ios::ios_base::floatfield);
	os.setf(ios::ios_base::right, ios::ios_base::adjustfield);
	for(it=in.begin(); it!=in.end(); it++){
		os.width(14);
		os<<*it<<"\t";
	}
	for(it=out.begin(); it!=out.end(); it++){
		os.width(14);
		os<<*it<<"\t";
	}
	os<<endl;
}
