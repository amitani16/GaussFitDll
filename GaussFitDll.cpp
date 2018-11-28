// GaussFitDll.cpp : Defines the exported functions for the DLL application.
//
#define _CRT_SECURE_NO_WARNINGS	// This suppresses some stupid VC++ errors.


#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <fstream>

#include "GaussFitDll.h"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/NonLinearOptimization"
#include "unsupported/Eigen/NumericalDiff"

#define lengthof(x) (sizeof(x) / sizeof(*(x)))

using namespace Eigen;
using namespace std;

const int NUM_PARAM            = 6;
const int MAX_ITERATION        = 1000;
const double CHI_SQUARE_CHANGE = 1e-10;

double alpha[6][6] = { 0.0 };
double beta[6][1] = { 0.0 };



/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
template<class T>
inline void SWAP(T &a, T &b)
{
	T dum = a; a = b; b = dum;
}


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
double getCurrentTimeInSec()
{
	SYSTEMTIME st;
	GetSystemTime(&st);

	return (double)(st.wSecond + st.wMilliseconds / 1000.0);
}


/*****************************************************************************/
/* Levenberg-Marquardt with numerical differentiation                        */
/*****************************************************************************/
// Generic functor
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

	int inputs() const { return m_inputs; }
	int values() const { return m_values; }

};


/*****************************************************************************/
/* Levenberg-Marquardt with numerical differentiation                        */
/*****************************************************************************/
struct Gauss_functor_ND : Functor<double>
{
	Gauss_functor_ND(int inputs, int values, double *x, double *y, double *z)
		: inputs_(inputs), values_(values), x(x), y(y), z(z) {}

	const int inputs_;	// parameters to be determined
	const int values_;	// num of data points
	double *x;
	double *y;
	double *z;

	int operator()(const VectorXd& b, VectorXd& fvec) const
	{
		for (int i = 0; i < values_; ++i)
			fvec[i] = b[0] *
		 			  exp(- ( (x[i] - b[1]) * (x[i] - b[1]) ) / (2 * b[2] * b[2])
						  - ( (y[i] - b[3]) * (y[i] - b[3]) ) / (2 * b[4] * b[4])
						 ) +
					  b[5] - z[i];

		return 0;
	}

	int inputs() const { return inputs_; }
	int values() const { return values_; }
};


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
struct Gauss_functor
{
	Gauss_functor(int inputs, int values, double *x, double *y, double *z)
		: inputs_(inputs), values_(values), x(x), y(y), z(z) {}

	const int inputs_;	// parameters to be determined
	const int values_;	// num of data points
	double *x;
	double *y;
	double *z;

	int operator()(const VectorXd& b, VectorXd& fvec) const
	{
		for (int i = 0; i < values_; ++i)
			fvec[i] = b[0] *
			exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
			    -((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
			) +
			b[5] - z[i];

		return 0;
	}

	int df(const VectorXd& b, MatrixXd& fjac)
	{
		for (int i = 0; i < values_; ++i) {
			fjac(i, 0) =
				exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
				    -((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
				   );

			fjac(i, 1) =
				b[0] *
				( (x[i] - b[1]) / (b[2] * b[2]) ) *
				exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
				    -((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
				);

			fjac(i, 2) =
				b[0] *
				( ((x[i] - b[1]) * (x[i] - b[1])) / (b[2] * b[2] * b[2]) ) *
				exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
				    -((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
				);

			fjac(i, 3) =
				b[0] *
				(x[i] - b[3]) / (b[4] * b[4]) *
				exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
				- ((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
				);

			fjac(i, 4) =
				b[0] *
				( ((x[i] - b[3]) * (x[i] - b[3])) / (b[4] * b[4] * b[4]) ) *
				exp(-((x[i] - b[1]) * (x[i] - b[1])) / (2 * b[2] * b[2])
				    -((y[i] - b[3]) * (y[i] - b[3])) / (2 * b[4] * b[4])
				);

			fjac(i, 5) = 1.0;
		}
		return 0;
	}

	int inputs() const { return inputs_; }
	int values() const { return values_; }
};


/*****************************************************************************/
/* Levenberg-Marquardt with numerical differentiation                        */
/*****************************************************************************/
int GaussFitLMwND(int size_, double* x_, double* y_, double* z_, double* p_)
{
	const int n = 6;
	int info;
	Eigen::VectorXd p(n);
	p << p_[0],	p_[1], p_[2], p_[3], p_[4], p_[5];

	std::vector<double> x(&x_[0], &x_[size_]);
	std::vector<double> y(&y_[0], &y_[size_]);
	std::vector<double> z(&z_[0], &z_[size_]);

	Gauss_functor_ND functor(n, x.size(), &x[0], &y[0], &z[0]);
	Eigen::NumericalDiff<Gauss_functor_ND> numDiff(functor);
	Eigen::LevenbergMarquardt<NumericalDiff<Gauss_functor_ND>> lm(numDiff);

	lm.parameters.maxfev = 100;
	lm.parameters.xtol = 1.0e-10;

	double start_time = getCurrentTimeInSec();
	info = lm.minimize(p);
	double end_time = getCurrentTimeInSec();
	cout << "time = " << 1000 * (end_time - start_time) << " ms" << endl << endl;

	for (int i = 0; i < n; i++)
		p_[i] = p[i];

	return 0;
}


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
int GaussFitLM(int size_, double* x_, double* y_, double*z_, double* p_)
{
	const int n = 6;
	int info;
	Eigen::VectorXd p(n);
	p << p_[0], p_[1], p_[2], p_[3], p_[4], p_[5];

	std::vector<double> x(&x_[0], &x_[size_]);
	std::vector<double> y(&y_[0], &y_[size_]);
	std::vector<double> z(&z_[0], &z_[size_]);

	Gauss_functor functor(n, x.size(), &x[0], &y[0], &z[0]);
	Eigen::LevenbergMarquardt<Gauss_functor> lm(functor);

	lm.parameters.maxfev = 100;
	lm.parameters.xtol = 1.0e-10;

	double start_time = getCurrentTimeInSec();
	info = lm.minimize(p);
	double end_time = getCurrentTimeInSec();
	cout << "time = " << 1000 * (end_time - start_time) << " ms" << endl << endl;

	for (int i = 0; i < n; i++)
		p_[i] = p[i];

	return 0;
}


/*****************************************************************************/
/* Numerical Recipes                                                         */
/*****************************************************************************/
/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
//////////////////////////////////////////////////////////////////////////
// 	z = A * exp(-(x-xc)^2/(2*sx^2))*exp(-(y-yc)^2/(2*sy^2)) + B
//////////////////////////////////////////////////////////////////////////
double Gauss2D(double x_, double y_, double p_[], double dzdp_[])
{
	double dx  = x_ - p_[1];				// (x-xc)
	double dy  = y_ - p_[3];				// (y-yc)
	double dx2 = dx * dx;					// (x-xc)^2
	double dy2 = dy * dy;					// (y-yc)^2
	double sx2 = p_[2] * p_[2];				// 2 * sx^2
	double sy2 = p_[4] * p_[4];				// 2 * sy^2
	double ex  = exp((double)-dx2/(2 * sx2));
	double ey  = exp((double)-dy2/(2 * sy2));

	dzdp_[0] = ex * ey;							// dz/dA

	double G2D = p_[0] * dzdp_[0];				// 2D Gaussian without background

	dzdp_[1] = (dx / sx2) * G2D;				// dz/dxc
	dzdp_[2] = (dx2 / (sx2 * p_[2])) * G2D;		// dz/dxs

	dzdp_[3] = (dy / sy2) * G2D;				// dz/dyc
	dzdp_[4] = (dy2 / (sy2 * p_[4])) * G2D;		// dz/dys

	dzdp_[5] = (double)1.0;						// dz/dz0

	return G2D + p_[5];
}


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
double get_matrices(int size_, double x_[], double y_[], double z_[], double p_[])
{
	double dzdp[6] = { 0.0 };
	double G2D;
	double chi_square = 0.0;

	for (int i = 0; i < size_; i++) {
		G2D = Gauss2D(x_[i], y_[i], p_, dzdp);		// get dz/dp at (xi, yi) with p

		for (int r = 0; r < NUM_PARAM; r++) {		// compute upper right elements
			for (int c = r; c < NUM_PARAM; c++) {	// include diagonal elements
				alpha[r][c] += (dzdp[r] * dzdp[c]);
			}
		}

		for (int r = 0; r < NUM_PARAM; r++) {
			beta[r][0] += (z_[i] - G2D) * dzdp[r];
		}

		chi_square += (z_[i] - G2D) * (z_[i] - G2D);
	}

	for (int r = 1; r < NUM_PARAM; r++)
		for (int c = 0; c < r; c++)
			alpha[r][c] = alpha[c][r];

	return chi_square;
}


/*****************************************************************************/
/* from NR3 gaussj.h                                                         */
/*****************************************************************************/
void Gauss_Jordan()
{
	int icol;
	int irow;

	const int N = 6;	// # of rows
	const int M = 1;	// # of cols

	double big;
	double dum;
	double pivinv;

	int indxc[N];	// these keep pivoting info
	int indxr[N];
	int ipiv[N];

	for (int j = 0; j < N; j++) { ipiv[j] = 0; }

	for (int i = 0; i < N; i++) {
		big = 0.0;

		for (int row = 0; row < N; row++) {
			if (ipiv[row] != 1) {
				for (int col = 0; col < N; col++) {			// searches pivot point
					if (ipiv[col] == 0) {
						if (abs(alpha[row][col]) >= big) {
							big = abs(alpha[row][col]);
							irow = row;
							icol = col;
						}
					}
				}
			}
		}

		++(ipiv[icol]);

		if (irow != icol) {
			for (int l = 0; l < N; l++) SWAP(alpha[irow][l], alpha[icol][l]);
			for (int l = 0; l < M; l++) SWAP(beta[irow][l],  beta[icol][l]);
		}

		indxr[i] = irow;
		indxc[i] = icol;

		if (alpha[icol][icol] == 0.0)
			throw("gaussj: Singular Matrix");

		pivinv = 1.0 / alpha[icol][icol];
		alpha[icol][icol] = 1.0;

		for (int l = 0; l < N; l++) alpha[icol][l] *= pivinv;

		for (int l = 0; l < M; l++) beta[icol][l] *= pivinv;

		for (int ll = 0; ll < N; ll++) {
			if (ll != icol) {
				dum = alpha[ll][icol];
				alpha[ll][icol] = 0.0;
				for (int l = 0; l < N; l++)
					alpha[ll][l] -= alpha[icol][l] * dum;

				for (int l = 0; l < M; l++)
					beta[ll][l] -= beta[icol][l] * dum;
			}
		}
	}

	for (int l = N - 1; l >= 0; l--) {
		if (indxr[l] != indxc[l])
			for (int k = 0; k < N; k++)
				SWAP(alpha[k][indxr[l]], alpha[k][indxc[l]]);
	}
}


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
void fit(int ndata, double x_[], double y_[], double z_[], double p_[])
{
	fill((double*)alpha, (double*)(alpha + lengthof(alpha)), 0.0);
	fill((double*)beta, (double*)(beta + lengthof(beta)), 0.0);

	int    num_iteration  = 0;
	double lambda         = 0.001;
	double chi_square     = 0.0;
	double old_chi_square = 0.0;
	int    flag_done      = 0;

	chi_square = get_matrices(ndata, x_, y_, z_, p_);	// cout << "chi^2 = " << chi_square << endl;
	old_chi_square = chi_square;
	do {
		for (int i = 0; i < NUM_PARAM; i++) { alpha[i][i] *= (1.0 + lambda); }

		Gauss_Jordan();				// cout << "x = " << p_[1] << "\ty = " << p_[3] << endl;

		for (int i = 0; i < NUM_PARAM; i++)		p_[i] += beta[i][0];					//

		chi_square = get_matrices(ndata, x_, y_, z_, p_);	// cout << "chi^2 = " << chi_square << endl;

		double tmp1 = fabs(old_chi_square - chi_square);
		double tmp2 = old_chi_square;

		if ((tmp1/tmp2) < CHI_SQUARE_CHANGE) { flag_done = 1; }	//if small enough, stop.

		if (chi_square <= old_chi_square) {		// if chi square is smaller, weight Gauss-Newton method
			lambda *= 0.1;
			//cout << "GN" << endl;
			old_chi_square = chi_square;
		} else {								// if chi square is larger, switch to the steepest descent method
			lambda *= 10.0;
			//cout << "SD" << endl;
			chi_square = old_chi_square;
		}

		num_iteration++;
	} while ((flag_done == 0) && (num_iteration <= MAX_ITERATION));

}


/*****************************************************************************/
/*                                                                           */
/*****************************************************************************/
bool GaussFitNR(int size_, double x_[], double y_[], double z_[], double p_[])
{
	double start_time = getCurrentTimeInSec();
	fit(size_, x_, y_, z_, p_);
	double end_time = getCurrentTimeInSec();
	cout << "time = " << 1000 * (end_time - start_time) << " ms" << endl << endl;

	return true;
}
