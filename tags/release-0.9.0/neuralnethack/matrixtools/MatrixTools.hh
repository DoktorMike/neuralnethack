/*$Id: MatrixTools.hh 1622 2007-05-08 08:29:10Z michael $*/

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


#ifndef __MatrixTools_hh__
#define __MatrixTools_hh__

#include <vector>

/**This namespace encloses functions for vector and matrix manipulation. */
namespace MatrixTools
{

	//VECTOR FUNCTIONS------------------------------------------------------------//

	/**Add vector v2 to v1.
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 */
	void add(std::vector<double>& v1, std::vector<double>& v2);

	/**Add vector v1 and v2 and put the result in res.
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 * \param res the result from the addition.
	 */
	void add(std::vector<double>& v1, std::vector<double>& v2, 
			std::vector<double>& res);

	/**Subtract vector v2 from v1.
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 */
	void sub(std::vector<double>& v1, std::vector<double>& v2);

	/**Subtract vector v2 from v1 and put the result in res.
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 * \param res the result from the addition.
	 */
	void sub(std::vector<double>& v1, std::vector<double>& v2, 
			std::vector<double>& res);

	/**Multiply each element in vector vec with a scalar.
	 * \param vec the vector to be scaled.
	 * \param scalar the value to multiply each element in vec with.
	 */
	void mul(std::vector<double>& vec, double scalar);

	/**Multiply each element in vector vec with a scalar and put result in
	 * res.
	 * \param vec the vector to be scaled.
	 * \param scalar the value to multiply each element in vec with.
	 * \param res the result from the scaling.
	 */
	void mul(std::vector<double>& vec, double scalar, std::vector<double>& res);

	/**Divide each element in vector vec with a scalar.
	 * \param vec the vector to be scaled.
	 * \param scalar the value to divide each element in vec with.
	 */
	void div(std::vector<double>& vec, double scalar);

	/**Divide each element in vector vec with a scalar.
	 * \param vec the vector to be scaled.
	 * \param scalar the value to divide each element in vec with.
	 * \param res the result from the scaling.
	 */
	void div(std::vector<double>& vec, double scalar, std::vector<double>& res);

	/**Perform the inner vector product on the two vectors. The first vector
	 * is treated as transposed whilst the second is treated normally. Thus 
	 * \f[\vec{v_1}^T\cdot\vec{v_2}\f]
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 * \return the resulting scalar.
	 */
	double innerProduct(std::vector<double>& v1, std::vector<double>& v2);

	/**Perform the outer vector product on the two vectors. The first vector
	 * is treated normally whilst the other is treated as transposed. Thus 
	 * \f[\vec{v_1}\cdot\vec{v_2}^T\f]
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 * \param res the matrix to put the result into.
	 */
	void outerProduct(std::vector<double>& v1, std::vector<double>& v2,
			std::vector< std::vector<double> >& res);

	/**Perform the outer vector product on the two vectors. The first vector
	 * is treated normally whilst the other is treated as transposed. Thus 
	 * \f[\vec{v_1}\cdot\vec{v_2}^T\f]
	 * \param v1 the first vector.
	 * \param v2 the second vector.
	 * \return the resulting matrix.
	 */
	std::vector< std::vector<double> > outerProduct(std::vector<double>& v1,
			std::vector<double>& v2);

	/**Prints out the contents of a vector.
	 * \param v the vector to print.
	 */
	void print(std::vector<double>& v);

	/**Prints out the contents of a vector.
	 * \param v the vector to print.
	 */
	void print(std::vector<uint>& v);

	//MATRIX FUNCTIONS------------------------------------------------------------//

	/**Creates a matrix with row rows and col columns.
	 * \param row the number of rows.
	 * \param col the number of columns.
	 * \return the resulting matrix.
	 */
	std::vector< std::vector<double> > matrix(uint row, uint col);

	/**Creates the identity matrix with n rows columns.
	 * \param n the number of rows and columns.
	 * \return the resulting identity matrix.
	 */
	std::vector< std::vector<double> > identity(uint n);

	/**Add matrix m2 to matrix m1.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 */
	void add(std::vector< std::vector<double> >& m1, std::vector< std::vector<double> >& m2);

	/**Perform addition of two matrices.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 * \param res the matrix to put the result into.
	 */
	void add(std::vector< std::vector<double> >& m1, std::vector< std::vector<double> >& m2,
			std::vector< std::vector<double> >& res);

	/*Perform addition of two matrices.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 * \return the resulting matrix.
	 *
	 std::vector< std::vector<double> > add(std::vector< std::vector<double> >& m1, 
	 std::vector< std::vector<double> >& m2);*/

	/**Subtract matrix m2 from matrix m1.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 */
	void sub(std::vector< std::vector<double> >& m1, std::vector< std::vector<double> >& m2);

	/**Perform subtraction of two matrices.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 * \param res the matrix to put the result into.
	 */
	void sub(std::vector< std::vector<double> >& m1, std::vector< std::vector<double> >& m2,
			std::vector< std::vector<double> >& res);

	/* Perform subtraction of two matrices.
	 * \param m1 the first matrix.
	 * \param m2 the second matrix.
	 * \return the resulting matrix.
	 *
	 std::vector< std::vector<double> > sub(std::vector< std::vector<double> >& m1, 
	 std::vector< std::vector<double> >& m2);*/

	/**Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 */
	void mul(std::vector< std::vector<double> >& m, double scale);

	/**Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 * \param res the matrix to put the result into.
	 */
	void mul(std::vector< std::vector<double> >& m, double scale,
			std::vector< std::vector<double> >& res);

	/*Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 * \return the resulting matrix.
	 *
	 std::vector< std::vector<double> > mul(std::vector< std::vector<double> >& m, double scale);*/

	/**Perform the dot product between a matrix and a vector.
	 * \f[M\cdot\vec{v}^T\f]
	 * \param m the matrix.
	 * \param v the vector.
	 * \param res the vector to put the result into.
	 */
	void mul(std::vector< std::vector<double> >& m, std::vector<double>& v,
			std::vector<double>& res);

	/**Perform the dot product between a matrix and a vector.
	 * \f[M\cdot\vec{v}^T\f]
	 * \param m the matrix.
	 * \param v the vector.
	 * \return the resulting vector.
	 */
	std::vector<double> mul(std::vector< std::vector<double> >& m, std::vector<double>& v);

	/**Perform the dot product between a vector and a matrix.
	 * \f[\vec{v}^T\cdot M\f]
	 * \param m the matrix.
	 * \param v the vector.
	 * \param res the vector to put the result into.
	 */
	void mul(std::vector<double>& v, std::vector< std::vector<double> >& m, 
			std::vector<double>& res);

	/**Perform the dot product between a vector and a matrix.
	 * \f[\vec{v}^T\cdot M\f]
	 * \param m the matrix.
	 * \param v the vector.
	 * \return the resulting vector.
	 */
	std::vector<double> mul(std::vector<double>& v, std::vector< std::vector<double> >& m);

	/**Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 */
	void div(std::vector< std::vector<double> >& m, double scale);

	/**Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 * \param res the matrix to put the result into.
	 */
	void div(std::vector< std::vector<double> >& m, double scale,
			std::vector< std::vector<double> >& res);

	/*Perform scaling of a matrix.
	 * \param m the matrix to scale.
	 * \param scale the scale.
	 * \return the resulting matrix.
	 *
	 std::vector< std::vector<double> > div(std::vector< std::vector<double> >& m, double scale);*/

	/**Prints out the contents of a matrix.
	 * \param m the matrix to print.
	 */
	void print(std::vector< std::vector<double> >& m);
}
#endif
