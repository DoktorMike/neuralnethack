#ifndef __MatrixTools_hh__
#define __MatrixTools_hh__

#include <vector>

/**This namespace encloses functions for vector and matrix manipulation. */
namespace MatrixTools{

    using std::vector;

//VECTOR FUNCTIONS------------------------------------------------------------//
    
    /**Add vector v1 and v2 and put the result in res.
     * \param v1 the first vector.
     * \param v2 the second vector.
     * \param res the result from the addition.
     */
    void add(vector<double>& v1, vector<double>& v2, 
	    vector<double>& res);

    /**Subtract vector v2 from v1 and put the result in res.
     * \param v1 the first vector.
     * \param v2 the second vector.
     * \param res the result from the addition.
     */
    void sub(vector<double>& v1, vector<double>& v2, 
	    vector<double>& res);

    /**Multiply each element in vector vec with a scalar.
     * \param vec the vector to be scaled.
     * \param scalar the value to multiply each element in vec with.
     */
    void mul(vector<double>& vec, double scalar);

    /**Divide each element in vector vec with a scalar.
     * \param vec the vector to be scaled.
     * \param scalar the value to divide each element in vec with.
     */
    void div(vector<double>& vec, double scalar);
    
    /**Perform the inner vector product on the two vectors. The first vector
     * is treated as transposed whilst the second is treated normally. Thus 
     * \f[\vec{v_1}^T\cdot\vec{v_2}\f]
     * \param v1 the first vector.
     * \param v2 the second vector.
     * \return the resulting scalar.
     */
    double innerProduct(vector<double>& v1, vector<double>& v2);

    /**Perform the outer vector product on the two vectors. The first vector
     * is treated normally whilst the other is treated as transposed. Thus 
     * \f[\vec{v_1}\cdot\vec{v_2}^T\f]
     * \param v1 the first vector.
     * \param v2 the second vector.
     * \param res the matrix to put the result into.
     */
    void outerProduct(vector<double>& v1, vector<double>& v2,
	    vector< vector<double> >& res);
    
    /**Perform the outer vector product on the two vectors. The first vector
     * is treated normally whilst the other is treated as transposed. Thus 
     * \f[\vec{v_1}\cdot\vec{v_2}^T\f]
     * \param v1 the first vector.
     * \param v2 the second vector.
     * \return the resulting matrix.
     */
    vector< vector<double> > outerProduct(vector<double>& v1,
	    vector<double>& v2);

    /**Prints out the contents of a vector.
     * \param v the vector to print.
     */
    void print(vector<double>& v);

//MATRIX FUNCTIONS------------------------------------------------------------//

    /**Creates a matrix with row rows and col columns.
     * \param row the number of rows.
     * \param col the number of columns.
     * \return the resulting matrix.
     */
    vector< vector<double> > matrix(uint row, uint col);

    /**Creates the identity matrix with n rows columns.
     * \param n the number of rows and columns.
     * \return the resulting identity matrix.
     */
    vector< vector<double> > identity(uint n);

    /**Perform addition of two matrices.
     * \param m1 the first matrix.
     * \param m2 the second matrix.
     * \param res the matrix to put the result into.
     */
    void add(vector< vector<double> >& m1, vector< vector<double> >& m2,
	    vector< vector<double> >& res);

    /**Perform addition of two matrices.
     * \param m1 the first matrix.
     * \param m2 the second matrix.
     * \return the resulting matrix.
     */
    vector< vector<double> > add(vector< vector<double> >& m1, 
	    vector< vector<double> >& m2);

    /**Perform subtraction of two matrices.
     * \param m1 the first matrix.
     * \param m2 the second matrix.
     * \param res the matrix to put the result into.
     */
    void sub(vector< vector<double> >& m1, vector< vector<double> >& m2,
	    vector< vector<double> >& res);

    /**Perform subtraction of two matrices.
     * \param m1 the first matrix.
     * \param m2 the second matrix.
     * \return the resulting matrix.
     */
    vector< vector<double> > sub(vector< vector<double> >& m1, 
	    vector< vector<double> >& m2);

    /**Perform scaling of a matrix.
     * \param m the matrix to scale.
     * \param scale the scale.
     * \param res the matrix to put the result into.
     */
    void mul(vector< vector<double> >& m, double scale,
	    vector< vector<double> >& res);

    /**Perform scaling of a matrix.
     * \param m the matrix to scale.
     * \param scale the scale.
     * \return the resulting matrix.
     */
    vector< vector<double> > mul(vector< vector<double> >& m, double scale);

    /**Perform the dot product between a matrix and a vector.
     * \f[M\cdot\vec{v}^T\f]
     * \param m the matrix.
     * \param v the vector.
     * \param res the vector to put the result into.
     */
    void mul(vector< vector<double> >& m, vector<double>& v,
	vector<double>& res);

    /**Perform the dot product between a matrix and a vector.
     * \f[M\cdot\vec{v}^T\f]
     * \param m the matrix.
     * \param v the vector.
     * \return the resulting vector.
     */
    vector<double> mul(vector< vector<double> >& m, vector<double>& v);

    /**Perform the dot product between a vector and a matrix.
     * \f[\vec{v}^T\cdot M\f]
     * \param m the matrix.
     * \param v the vector.
     * \param res the vector to put the result into.
     */
    void mul(vector<double>& v, vector< vector<double> >& m, 
	vector<double>& res);

    /**Perform the dot product between a vector and a matrix.
     * \f[\vec{v}^T\cdot M\f]
     * \param m the matrix.
     * \param v the vector.
     * \return the resulting vector.
     */
    vector<double> mul(vector<double>& v, vector< vector<double> >& m);

    /**Perform scaling of a matrix.
     * \param m the matrix to scale.
     * \param scale the scale.
     * \param res the matrix to put the result into.
     */
    void div(vector< vector<double> >& m, double scale,
	    vector< vector<double> >& res);

    /**Perform scaling of a matrix.
     * \param m the matrix to scale.
     * \param scale the scale.
     * \return the resulting matrix.
     */
    vector< vector<double> > div(vector< vector<double> >& m, double scale);

    /**Prints out the contents of a matrix.
     * \param m the matrix to print.
     */
    void print(vector< vector<double> >& m);
}
#endif
