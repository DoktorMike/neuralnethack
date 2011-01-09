/*$Id: SigmoidLayer.hh 1658 2007-07-05 21:53:26Z michael $*/

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


#ifndef __SigmoidLayer_hh__
#define __SigmoidLayer_hh__

#include "Layer.hh"

#include <string>
#include <cmath>
#include <cassert>

namespace MultiLayerPerceptron
{
	/**A class representing a sigmoid implementation of the layer interface.
	 * It knows the number of neurons contained in itself and its
	 * predecessor.
	 * \f[\varphi (v)=\frac{1}{1+\exp^{-v}}\f]
	 * \sa Layer, Mlp.
	 */
	class SigmoidLayer: public Layer
	{
		public:
			/**Constructor.
			 * \param nc the number of neurons in this layer.
			 * \param np the number of neurons in the previous layer.
			 */
			SigmoidLayer(const uint nc, const uint np);

			/**Destructor.
			 */
			virtual ~SigmoidLayer();

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

	inline double SigmoidLayer::fire(const double lif) const
	{return 1.0/(1.0+exp(-lif));}

	inline double SigmoidLayer::fire(const uint i) const
	{
		assert(i<theOutputs.size());
		return theOutputs[i];
	}

	inline double SigmoidLayer::firePrime(const double lif) const
	{
		double tmp = fire(lif);
		return tmp*(1-tmp);
	}

	inline double SigmoidLayer::firePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		return theOutputs[i]*(1-theOutputs[i]);
	}

	inline double SigmoidLayer::firePrimePrime(const double lif) const
	{
		double f = fire(lif);
		double fp = f*(1-f);
		return fp-2*f*fp;
	}

	inline double SigmoidLayer::firePrimePrime(const uint i) const
	{
		assert(i<theOutputs.size());
		double f = theOutputs[i];
		double fp = f*(1-f);
		return fp-2*f*fp;
	}
}
#endif
