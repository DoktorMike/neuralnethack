#include "DataManager.hh"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>
#include <fstream>
#include <iterator>

using namespace DataTools;
using namespace std;

static std::mt19937& rng() {
	static std::mt19937 gen(std::random_device{}());
	return gen;
}

DataManager::DataManager():indices(vector<uint>(0)), isRandom(true){}

DataManager::DataManager(const DataManager& ds)
{*this = ds;}

DataManager::~DataManager()
{}

DataManager& DataManager::operator=(const DataManager& ds)
{
	if(this != &ds){
		indices = ds.indices;
		isRandom = ds.isRandom;
	}
	return *this;
}

bool DataManager::random() const
{return isRandom;}

void DataManager::random(bool rnd)
{isRandom = rnd;}

pair<DataSet, DataSet>* DataManager::split(DataSet& ds, double ratio)
{
	assert(ratio < 1.0 && ratio > 0.0);
	uint n = ds.size();
	uint nTraining = (uint)nearbyint(ratio * n);
	DataSet training(ds);
	DataSet validation(ds);

	indices = ds.indices();
	if(isRandom) shuffle(indices.begin(), indices.end(), rng());

	//Tell the dataSets which data points to use.
	training.indices().assign(indices.begin(), indices.begin()+nTraining);
	validation.indices().assign(indices.begin()+nTraining, indices.end());

	return new pair<DataSet, DataSet>(training, validation);
}

vector<DataSet>* DataManager::split(DataSet& ds, uint k)
{
	uint n = ds.size();
	uint nInEachPart = n/k;
	uint nLeft = n%k;
	vector<DataSet>* splits = new vector<DataSet>(k);
	assert(k>0 && k<=n); //Probably should do better error reporting here. :-)
	if(k < 2){
		cerr<<"Warning: DataManager::split(): split with k<2."<<endl;
		splits->clear(); splits->push_back(ds);
		return splits;
	}

	indices = ds.indices();
	if(isRandom) shuffle(indices.begin(), indices.end(), rng());
	
	vector<uint> tmpIndices(0);
	vector<uint>::iterator indexIterator = indices.begin();
	for(uint i=0; i<splits->size(); ++i, indexIterator+=nInEachPart){
		tmpIndices.assign(indexIterator, indexIterator+nInEachPart);
		DataSet d(ds);
		d.indices(tmpIndices);
		splits->at(i) = d;
	}

	vector<DataSet>::iterator splitIterator = splits->begin();
	for(uint i=0; i<nLeft; ++i, ++splitIterator, ++indexIterator)
		splitIterator->indices().push_back(*indexIterator);

	return splits;
}

pair<DataSet, DataSet>* DataManager::split(DataSet& ds)
{
	uint n = ds.size();
	DataSet trn(ds);
	DataSet val(ds);
	vector<uint> origInd = ds.indices();
	buildIndicesWithReplacement(origInd);
	assert(origInd.size() == indices.size());
	if(isRandom) shuffle(indices.begin(), indices.end(), rng());
	trn.indices(indices);

	//It's ok to sort here since we don't really disrupt the trnSet.
	sort(origInd.begin(), origInd.end());
	sort(indices.begin(), indices.end());
	vector<uint> valInd(n);
	valInd.erase( 
			set_difference(
				origInd.begin(), 
				origInd.end(), 
				indices.begin(), 
				indices.end(), 
				valInd.begin()), 
			valInd.end());
	if(isRandom) shuffle(valInd.begin(), valInd.end(), rng());
	val.indices(valInd);

	return new pair<DataSet, DataSet>(trn, val);
}

DataSet DataManager::join(vector<DataSet>& splits)
{
	assert(splits.size());
	if(splits.size() < 2) return splits.front();

	indices.clear();
	vector<DataSet>::iterator its;
	for(its=splits.begin(); its!=splits.end(); ++its){
		vector<uint>& tmp = its->indices();
		indices.insert(indices.end(), tmp.begin(), tmp.end());
	}
	DataSet d(splits.front());
	d.indices(indices);
	return d;
}

//PRIVATE-----------------------------------------------------------------------

void DataManager::buildIndices(uint n)
{
	indices = vector<uint>(n);
	iota(indices.begin(), indices.end(), 0);
	shuffle(indices.begin(), indices.end(), rng());
}

void DataManager::buildIndicesWithReplacement(uint n)
{
	indices.clear();

	for(uint i=0; i<n; ++i){
		uint index = (uint) nearbyint( (n-1) * drand48());
		indices.push_back(index);
	}
}

void DataManager::buildIndicesWithReplacement(vector<uint>& orig)
{
	indices.clear();

	uint n = orig.size();
	for(uint i=0; i<n; ++i){
		uint index = (uint) nearbyint( (n-1) * drand48());
		indices.push_back(orig.at(index));
	}
}

void DataManager::printIndices(ostream& os)
{
	assert(os);
	copy(indices.begin(), indices.end(), ostream_iterator<uint>(os, " "));
	os<<endl;
}
