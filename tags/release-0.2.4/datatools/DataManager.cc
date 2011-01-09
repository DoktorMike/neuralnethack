#include "DataManager.hh"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <ext/numeric>

using namespace DataTools;
using namespace std;

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

bool DataManager::random()
{return isRandom;}

void DataManager::random(bool rnd)
{isRandom = rnd;}

pair<DataSet, DataSet> DataManager::split(DataSet& ds, double ratio)
{
	assert(ratio < 1.0 && ratio > 0.0);
	uint n = ds.size();
	uint nTraining = (uint)nearbyint(ratio * n);
	DataSet training(ds);
	DataSet validation(ds);

	indices = ds.indices();
	if(isRandom) random_shuffle(indices.begin(), indices.end());

	//Tell the dataSets which data points to use.
	vector<uint> tmpIndices(0);
	tmpIndices.assign(indices.begin(), indices.begin()+nTraining);
	training.indices(tmpIndices);
	tmpIndices.assign(indices.begin()+nTraining, indices.end());
	validation.indices(tmpIndices);

	return pair<DataSet, DataSet>(training, validation);
}

vector<DataSet> DataManager::split(DataSet& ds, uint k)
{
	uint n = ds.size();
	uint nInEachPart = n/k;
	uint nLeft = n%k;
	vector<DataSet> splits(k);
	assert(k>0 && k<=n); //Probably should do better error reporting here. :-)
	if(k < 2){
		//cerr<<"Warning: DataManager::split(): split with k<2."<<endl;
		splits.clear(); splits.push_back(ds);
		return splits;
	}

	indices = ds.indices();
	if(isRandom) random_shuffle(indices.begin(), indices.end());
	
	vector<uint> tmpIndices(0);
	vector<uint>::iterator indexIterator = indices.begin();
	for(uint i=0; i<splits.size(); ++i, indexIterator+=nInEachPart){
		tmpIndices.assign(indexIterator, indexIterator+nInEachPart);
		DataSet d(ds);
		d.indices(tmpIndices);
		splits.at(i) = d;
	}

	vector<DataSet>::iterator splitIterator = splits.begin();
	for(uint i=0; i<nLeft; ++i, ++splitIterator, ++indexIterator)
		splitIterator->indices().push_back(*indexIterator);

	return splits;
}

pair<DataSet, DataSet> DataManager::split(DataSet& ds)
{
	uint n = ds.size();
	DataSet trn(ds);
	DataSet val(ds);
	vector<uint> origInd = ds.indices();
	buildIndicesWithReplacement(origInd);
	assert(origInd.size() == indices.size());
	if(isRandom) random_shuffle(indices.begin(), indices.end());
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
	if(isRandom) random_shuffle(valInd.begin(), valInd.end());
	val.indices(valInd);

	return pair<DataSet, DataSet>(trn, val);
}

DataSet DataManager::join(vector<DataSet>& splits)
{
	if(splits.size() < 2){
		//cerr<<"Warning: DataManager::join(): join with k<2."<<endl;
		return splits.front();
	}

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
	random_shuffle(indices.begin(), indices.end());
}

void DataManager::buildIndicesWithReplacement(uint n)
{
	indices.clear();

	for(uint i=0; i<n; ++i){
		uint index = (uint) ( ((double)n) * rand() / (RAND_MAX+1.0));
		indices.push_back(index);
	}
}

void DataManager::buildIndicesWithReplacement(vector<uint>& orig)
{
	indices.clear();

	uint n = orig.size();
	for(uint i=0; i<n; ++i){
		uint index = (uint) ( ((double)n) * rand() / (RAND_MAX+1.0));
		indices.push_back(orig.at(index));
	}
}

void DataManager::printIndices(ostream& os)
{
	assert(os);

	for(uint i=0; i<indices.size(); ++i)
		os<<indices[i]<<" ";
	os<<endl;
}
