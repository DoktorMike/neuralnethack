/*$Id: TanHypLayer.hh 1660 2007-07-10 11:54:49Z michael $*/

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


#ifndef __TanHypLayer_hh__
#define __TanHypLayer_hh__

#include "Layer.hh"

#include <string>
#include <cmath>

namespace MultiLayerPerceptron
{
	/**A class representing a tanhyp implementation of the layer interface.
	 * It knows the number of neurons contained in itself and its
	 * predecessor.
	 * \f[\varphi (v)=\tanh{v}\f]
	 * \sa Layer, Mlp.
	 */
	class TanHypLayer: public Layer
	{
		public:
			/**Constructor.
			 * \param nc the number of neurons in this layer.
			 * \param np the number of neurons in the previous layer.
			 */
			TanHypLayer(uint nc, uint np);

			/**Destructor.
			 */
			virtual ~TanHypLayer();

			//ACCESSOR AND MUTATOR FUNCTIONS

			//ACCESSOR FUNCTIONS

			//COUNTS AND CRAP

			//PRINTS

			//UTILS

			double fire(const double lif) const;
			double fire(const uint i) const;
			double firePrime(const double lif) const;
			double firePrime(const uint i) const;
			double firePrimePrime(const double lif) const;
			double firePrimePrime(const uint i) const;
	};

	inline double TanHypLayer::fire(const double lif) const { return tanh(lif); }

	inline double TanHypLayer::fire(const uint i) const
	{
		assert(i<theOutputs.size());
		return theOutputs[i];
	}

	inline double TanHypLayer::firePrime(const double lif) const
	{
		double tmp = fire(lif);
		return 1.0 - tmp * tmp;
	}

	inline double TanHypLayer::firePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		return 1.0 - theOutputs[i] * theOutputs[i];
	}

	inline double TanHypLayer::firePrimePrime(const double lif) const
	{
		double f = fire(lif);
		return 2*f*(f*f-1);
	}

	inline double TanHypLayer::firePrimePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		double f = theOutputs[i];
		return 2*f*(f*f-1);
	}
}
#endif
