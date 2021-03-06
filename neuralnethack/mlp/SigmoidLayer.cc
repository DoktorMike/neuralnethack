/*$Id: SigmoidLayer.cc 1623 2007-05-08 08:30:14Z michael $*/

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


#include "SigmoidLayer.hh"


using namespace MultiLayerPerceptron;

SigmoidLayer::SigmoidLayer(const uint nc, const uint np):Layer(nc, np, SIGMOID)
{}

SigmoidLayer::~SigmoidLayer()
{}

//ACCESSOR AND MUTATOR FUNCTIONS

//ACCESSOR FUNCTIONS

//COUNTS AND CRAP

//PRINTS

//UTILS

//PRIVATE--------------------------------------------------------------------//

