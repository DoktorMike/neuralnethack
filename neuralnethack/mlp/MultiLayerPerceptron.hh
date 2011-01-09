/*$Id: MultiLayerPerceptron.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __MultiLayerPerceptron_hh__
#define __MultiLayerPerceptron_hh__

/**This namespace encloses the MultiLayerPerceptron library.
 * It contains all classes and methods needed to create an MLP.
 */
namespace MultiLayerPerceptron
{

///A define for the Cross Entropy Error identification string.
#define CEE "kullback"
///A define for the Summed Square Error identification string.
#define SSE "sumsqr"
///A define for the Quasi Newton minimisation identification string.
#define QN "qn"
///A define for the Gradient Descent minimisation identification string.
#define GD "gd"
///A define for the Linear activation function identification string.
#define LINEAR "purelin"
///A define for the Sigmoid activation function identification string.
#define SIGMOID "logsig"
///A define for the TanHyp activation function identification string.
#define TANHYP "tansig"
///A define for the maximum training error.
#define MAX_ERROR 0.00001

}

#endif
