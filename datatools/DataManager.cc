#include "DataManager.hh"
#include <cmath>
#include <algorithm>
#include <list>

using namespace DataTools;
using std::list;

DataManager::DataManager()
	:indices(vector<uint>(0)), isRandom(true)
{}

DataManager::DataManager(const DataManager& ds)
{*this = ds;}

DataManager::~DataManager()
{}

DataManager& DataManager::operator=(const DataManager& ds)
{
	if(this != &ds)
	{
		indices = ds.indices;
		isRandom = ds.isRandom;
	}
	return *this;
}

bool DataManager::random()
{return isRandom;}

void DataManager::random(bool rnd)
{isRandom = rnd;}

vector<DataSet> DataManager::split(DataSet& ds, double ratio)
{
	assert(ratio < 1.0 && ratio > 0.0);
	ds.reset();
	uint n = ds.size();
	uint nTraining = (uint)nearbyint(ratio * n);
	//uint nTesting = n - nTraining;
	DataSet training(ds);
	DataSet testing(ds);
	
	vector<uint>::iterator indexIterator;
	if(isRandom){
		buildIndices(n);
		indexIterator = indices.begin();
	}
	else
		indexIterator = ds.indices().begin();

	//Tell the dataSets which data points to use.
	vector<uint> tmpIndices(0);
	tmpIndices.insert(tmpIndices.begin(), indices.begin(), indices.begin()+nTraining);
	training.indices(tmpIndices);
	tmpIndices.clear();
	tmpIndices.insert(tmpIndices.begin(), indices.begin()+nTraining, indices.end());
	testing.indices(tmpIndices);

	vector<DataSet> splits(2);
	splits[0] = training;
	splits[1] = testing;

	return splits;
}

vector<DataSet> DataManager::split(DataSet& ds, uint k)
{
	ds.reset();
	uint n = ds.size();
	assert(k>0 && k<=n); //Probably should do better error reporting here. :-)
	uint nInEachPart = n/k;
	uint nLeft = n%k;
	/*cout<<"Splitting "<<n<<" data points into "
		<<k<<" parts with "<<nInEachPart<<" in each part leaving "
		<<nLeft<<" left."<<endl;*/
	
	vector<uint>::iterator indexIterator;
	if(isRandom){
		buildIndices(n);
		indexIterator = indices.begin();
	}
	else
		indexIterator = ds.indices().begin();
	
	vector<DataSet> splits(k);
	vector<uint> tmpIndices(0);
	for(uint i=0; i<splits.size(); ++i, indexIterator+=nInEachPart){
		tmpIndices.assign(indexIterator,indexIterator+nInEachPart);
		DataSet d(ds);
		d.indices(tmpIndices);
		splits.at(i) = d;
	}
	
	vector<DataSet>::iterator splitIterator = splits.begin();
	for(uint i=0; i<nLeft; ++i, ++splitIterator, ++indexIterator)
		splitIterator->indices().push_back(*indexIterator);

	return splits;
}

DataSet DataManager::join(vector<DataSet>& splits)
{
	indices.clear();
	vector<DataSet>::iterator its;
	for(its=splits.begin(); its!=splits.end(); ++its){
		vector<uint>& tmp = its->indices();
		indices.insert(indices.end(), tmp.begin(), tmp.end());
	}
	DataSet d(splits.front());
	d.indices(indices);
	d.reset();
	return d;
}

//PRIVATE-----------------------------------------------------------------------

void DataManager::buildIndices(uint n)
{
	indices = vector<uint>(n,INT_MAX);
	//printIndices();

	//cout<<"Listing indices generated. n is "<<n<<endl;
	for(uint i=0; i<n; ++i){
		uint index = 0;
		vector<uint>::iterator result;
		do{
			//printIndices();
			index = (uint) ( ((double)n) * rand() / (RAND_MAX+1.0));
			result = find(indices.begin(), indices.end()-1, index);
			//cout<<"genererade index "<<index<<" och Hittade "<<*result<<" i listan.\n";
		}while(*result == index);
		//cout<<"Putting index "<<index<<endl;
		indices.at(i) = index;
	}
	//cout<<"\n";
}

void DataManager::printIndices()
{
	for(uint i=0; i<indices.size(); ++i)
		cout<<indices[i]<<" ";
	cout<<endl;
}
