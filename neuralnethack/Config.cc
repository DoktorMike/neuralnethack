/*$Id: Config.cc 1666 2007-08-23 08:38:39Z michael $*/

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


#include "Config.hh"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <ostream>

using namespace NeuralNetHack;
using namespace std;

Config::Config():theSuffix(""),
	theFileName(""), theIdCol(-1), theInCols(0),theOutCols(0),theRowRange(0),
	theFileNameT(""), theIdColT(-1), theInColsT(0),theOutColsT(0),theRowRangeT(0),
	theProblemType(false),theNumLayers(0),
    theArch(0),theActFcn(0),theErrFcn(""),theMinMethod(""),theMaxEpochs(0),
	theWeightElimOn(false), theWeightElimAlpha(0), theWeightElimW0(1),
	theEnsParamDataSelection(""), theEnsParamN(0), theEnsParamK(0), theEnsParamSplitMode(true), theEnsParamNewWeights(false),
	theMsParamDataSelection(""), theMsParamN(0), theMsParamK(0), theMsParamSplitMode(true), theMsParamNumTrainingData(0),
	theMsgParamN(0), theMsgParamK(0), theMsgParamSplitMode(true), theMsgParamNumTrainingData(0),
	theSaveSession(false), theInfo(0), theSaveOutputList(false), theSeed(0), theNormalization("no")
{
	gdParam.theBatchSize = 100;
	gdParam.theLearningRate = 0.1;
	gdParam.theDecLearningRate = 0.99;
	gdParam.theMomentum = 0.8;
}

Config::~Config(){}

void Config::print(std::ostream& os)
{
	os<<"Suffix\t\t"<<suffix()<<endl;
	os<<"Filename\t"<<theFileName<<endl;
	os<<"IdCol\t\t"<<theIdCol<<endl;
	os<<"InCol\t\t";
	copy(theInCols.begin(), theInCols.end()-1, ostream_iterator<uint>(os, ","));
	os<<theInCols.back()<<endl;
	os<<"OutCol\t\t";
	copy(theOutCols.begin(), theOutCols.end()-1, ostream_iterator<uint>(os, ","));
	os<<theOutCols.back()<<endl;
	os<<"RowRange\t";
	copy(theRowRange.begin(), theRowRange.end()-1, ostream_iterator<uint>(os, ","));
	os<<theRowRange.back()<<endl;
	os<<"FilenameT\t"<<theFileNameT<<endl;
	os<<"IdColT\t\t"<<theIdColT<<endl;
	os<<"InColT\t\t";
	copy(theInColsT.begin(), theInColsT.end()-1, ostream_iterator<uint>(os, ","));
	os<<theInColsT.back()<<endl;
	os<<"OutColT\t\t";
	copy(theOutColsT.begin(), theOutColsT.end()-1, ostream_iterator<uint>(os, ","));
	os<<theOutColsT.back()<<endl;
	os<<"RowRangeT\t";
	copy(theRowRangeT.begin(), theRowRangeT.end()-1, ostream_iterator<uint>(os, ","));
	os<<theRowRangeT.back()<<endl;
	os<<"PType\t\t"<<theProblemType<<endl;
	os<<"NLay\t\t"<<theNumLayers<<endl;
	os<<"Size\t\t";
	copy(theArch.begin(), theArch.end(), ostream_iterator<uint>(os, " "));
	os<<endl;
	os<<"ActFcn\t\t";
	copy(theActFcn.begin(), theActFcn.end(), ostream_iterator<string>(os, " "));
	os<<endl;
	os<<"ErrFcn\t\t"<<theErrFcn<<endl;
	os<<"MinMethod\t"<<theMinMethod<<endl;
	os<<"MaxEpochs\t"<<theMaxEpochs<<endl;
	os<<"GDParam\t\t"<<gdParam.theBatchSize<<" "<<gdParam.theLearningRate<<" "
		<<gdParam.theDecLearningRate<<" "<<gdParam.theMomentum<<endl;
	os<<"WeightElim\t"<<theWeightElimOn<<" "<<theWeightElimAlpha<<" "
		<<theWeightElimW0<<endl;
	os<<"EnsParam\t"<<theEnsParamDataSelection<<" "<<theEnsParamN<<" "
		<<theEnsParamK<<" "<<theEnsParamSplitMode<<" "<<theEnsParamNewWeights<<endl;
	os<<"MSParam\t\t"<<theMsParamDataSelection<<" "<<theMsParamN<<" "<<theMsParamK<<" "
		<<theMsParamSplitMode<<" "<<theMsParamNumTrainingData
		<<endl;
	os<<"MSGParam\t"<<theMsgParamN<<" "<<theMsgParamK<<" "
		<<theMsgParamSplitMode<<" "<<theMsgParamNumTrainingData<<endl;
	for(map<string, vector<double> >::iterator it = theVary.begin();
			it != theVary.end(); ++it){
		os<<"Vary\t\t"<<it->first<<" ";
		copy(it->second.begin(), it->second.end(), ostream_iterator<double>(os, " "));
		os<<endl;
	}
	os<<"SaveSession\t"<<theSaveSession<<endl;
	os<<"Info\t\t"<<theInfo<<endl;
	os<<"SaveOutputList\t"<<theSaveOutputList<<endl;
	os<<"Seed\t\t"<<theSeed<<endl;
	os<<"Normalization\t"<<theNormalization<<endl;
}

//PRIVATE--------------------------------------------------------------------//


