#pragma once
#include <glm/mat3x3.hpp>
#include "Vector.h"
#include "SquareMatrix.h"

namespace CsyVk {

	template <typename T, int Dim> class Vector;

	/*
	 * SquareMatrix<T,3> are defined for C++ fundamental integers types and floating-point types
	 * Elements are stored in a column-major order
	 */
	template <typename T>
	class SquareMatrix<T, 3>
	{
	public:
		typedef T VarType;

		DYN_FUNC SquareMatrix();
		DYN_FUNC explicit SquareMatrix(T);
		DYN_FUNC SquareMatrix(T x00, T x01, T x02, T x10, T x11, T x12, T x20, T x21, T x22);
		DYN_FUNC SquareMatrix(const Vector<T, 3> &row1, const Vector<T, 3> &row2, const Vector<T, 3> &row3);

		DYN_FUNC SquareMatrix(const SquareMatrix<T, 3>&);
		DYN_FUNC ~SquareMatrix();

		DYN_FUNC  static unsigned int rows() { return 3; }
		DYN_FUNC  static unsigned int cols() { return 3; }

		DYN_FUNC T& operator() (unsigned int i, unsigned int j);
		DYN_FUNC const T& operator() (unsigned int i, unsigned int j) const;

		DYN_FUNC const Vector<T, 3> row(unsigned int i) const;
		DYN_FUNC const Vector<T, 3> col(unsigned int i) const;

		DYN_FUNC void setRow(unsigned int i, const Vector<T, 3>& vec);
		DYN_FUNC void setCol(unsigned int j, const Vector<T, 3>& vec);

		DYN_FUNC const SquareMatrix<T, 3> operator+ (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator+= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator- (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator-= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator* (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator*= (const SquareMatrix<T, 3> &);
		DYN_FUNC const SquareMatrix<T, 3> operator/ (const SquareMatrix<T, 3> &) const;
		DYN_FUNC SquareMatrix<T, 3>& operator/= (const SquareMatrix<T, 3> &);

		DYN_FUNC SquareMatrix<T, 3>& operator= (const SquareMatrix<T, 3> &);

		DYN_FUNC bool operator== (const SquareMatrix<T, 3> &) const;
		DYN_FUNC bool operator!= (const SquareMatrix<T, 3> &) const;

		DYN_FUNC const SquareMatrix<T, 3> operator* (const T&) const;
		DYN_FUNC SquareMatrix<T, 3>& operator*= (const T&);
		DYN_FUNC const SquareMatrix<T, 3> operator/ (const T&) const;
		DYN_FUNC SquareMatrix<T, 3>& operator/= (const T&);

		DYN_FUNC const Vector<T, 3> operator* (const Vector<T, 3> &) const;

		DYN_FUNC const SquareMatrix<T, 3> operator- (void) const;

		DYN_FUNC const SquareMatrix<T, 3> transpose() const;
		DYN_FUNC const SquareMatrix<T, 3> inverse() const;

		DYN_FUNC T determinant() const;
		DYN_FUNC T trace() const;
		DYN_FUNC T doubleContraction(const SquareMatrix<T, 3> &) const;//double contraction
		DYN_FUNC T frobeniusNorm() const;
		DYN_FUNC T oneNorm() const;
		DYN_FUNC T infNorm() const;

		DYN_FUNC static const SquareMatrix<T, 3> identityMatrix();

		DYN_FUNC T* getDataPtr() { return &data_[0].x; }

	protected:
		Vector<T, 3> data_[3]; //default: zero matrix
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC  const SquareMatrix<T, 3> operator* (S scale, const SquareMatrix<T, 3> &mat)
	{
		return mat * scale;
	}

	template class SquareMatrix<float, 3>;
	template class SquareMatrix<double, 3>;
	//convenient typedefs
	typedef SquareMatrix<float, 3> Mat3f;
	typedef SquareMatrix<double, 3> Mat3d;

}  //end of namespace dyno

#include <cmath>
#include <limits>

#include "Vector.h"

namespace CsyVk
{
	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::SquareMatrix()
		:SquareMatrix(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::SquareMatrix(T value)
		: SquareMatrix(value, value, value, value, value, value, value, value, value) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::SquareMatrix(T x00, T x01, T x02, T x10, T x11, T x12, T x20, T x21, T x22)
	{
		data_[0] = Vector<T, 3>(x00, x10, x20);
		data_[1] = Vector<T, 3>(x01, x11, x21);
		data_[2] = Vector<T, 3>(x02, x12, x22);
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::SquareMatrix(const Vector<T, 3> &row1, const Vector<T, 3> &row2, const Vector<T, 3> &row3)
	{
		data_[0] = Vector<T, 3>(row1[0], row2[0], row3[0]);
		data_[1] = Vector<T, 3>(row1[1], row2[1], row3[1]);
		data_[2] = Vector<T, 3>(row1[2], row2[2], row3[2]);
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::SquareMatrix(const SquareMatrix<T, 3>& mat)
	{
		data_[0] = mat.data_[0];
		data_[1] = mat.data_[1];
		data_[2] = mat.data_[2];
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>::~SquareMatrix()
	{

	}

	template <typename T>
	DYN_FUNC T& SquareMatrix<T, 3>::operator() (unsigned int i, unsigned int j)
	{
		return const_cast<T &>(static_cast<const SquareMatrix<T, 3> &>(*this)(i, j));
	}

	template <typename T>
	DYN_FUNC const T& SquareMatrix<T, 3>::operator() (unsigned int i, unsigned int j) const
	{
		return data_[j][i];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> SquareMatrix<T, 3>::row(unsigned int i) const
	{
		Vector<T, 3> result((*this)(i, 0), (*this)(i, 1), (*this)(i, 2));
		return result;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> SquareMatrix<T, 3>::col(unsigned int i) const
	{
		Vector<T, 3> result((*this)(0, i), (*this)(1, i), (*this)(2, i));
		return result;
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 3>::setRow(unsigned int i, const Vector<T, 3>& vec)
	{
		data_[0][i] = vec[0];
		data_[1][i] = vec[1];
		data_[2][i] = vec[2];
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 3>::setCol(unsigned int j, const Vector<T, 3>& vec)
	{
		data_[j][0] = vec[0];
		data_[j][1] = vec[1];
		data_[j][2] = vec[2];
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator+ (const SquareMatrix<T, 3> &mat2) const
	{
		return SquareMatrix<T, 3>(*this) += mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator+= (const SquareMatrix<T, 3> &mat2)
	{
		data_[0] += mat2.data_[0];
		data_[1] += mat2.data_[1];
		data_[2] += mat2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator- (const SquareMatrix<T, 3> &mat2) const
	{
		return SquareMatrix<T, 3>(*this) -= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator-= (const SquareMatrix<T, 3> &mat2)
	{
		data_[0] -= mat2.data_[0];
		data_[1] -= mat2.data_[1];
		data_[2] -= mat2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator=(const SquareMatrix<T, 3> &mat2)
	{
		data_[0] = mat2.data_[0];
		data_[1] = mat2.data_[1];
		data_[2] = mat2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 3>::operator== (const SquareMatrix<T, 3> &mat2) const
	{
		return data_ == mat2.data_;
	}

	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 3>::operator!= (const SquareMatrix<T, 3> &mat2) const
	{
		return !((*this) == mat2);
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator* (const T& scale) const
	{
		return SquareMatrix<T, 3>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator*= (const T& scale)
	{
		data_[0] *= scale;
		data_[1] *= scale;
		data_[2] *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> SquareMatrix<T, 3>::operator* (const Vector<T, 3> &vec) const
	{
		Vector<T, 3> result(0);
		for (unsigned int i = 0; i < 3; ++i)
			for (unsigned int j = 0; j < 3; ++j)
				result[i] += (*this)(i, j)*vec[j];
		return result;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator* (const SquareMatrix<T, 3> &mat2) const
	{
		return SquareMatrix<T, 3>(*this) *= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator*= (const SquareMatrix<T, 3> &mat2)
	{
		SquareMatrix<T, 3> t = transpose();
		(*this)(0, 0) = t.data_[0].dot(mat2.data_[0]);
		(*this)(0, 1) = t.data_[0].dot(mat2.data_[1]);
		(*this)(0, 2) = t.data_[0].dot(mat2.data_[2]);
		(*this)(1, 0) = t.data_[1].dot(mat2.data_[0]);
		(*this)(1, 1) = t.data_[1].dot(mat2.data_[1]);
		(*this)(1, 2) = t.data_[1].dot(mat2.data_[2]);
		(*this)(2, 0) = t.data_[2].dot(mat2.data_[0]);
		(*this)(2, 1) = t.data_[2].dot(mat2.data_[1]);
		(*this)(2, 2) = t.data_[2].dot(mat2.data_[2]);

		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator/ (const SquareMatrix<T, 3> &mat2) const
	{
		return SquareMatrix<T, 3>(*this) *= mat2.inverse();
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator/= (const SquareMatrix<T, 3> &mat2)
	{
		SquareMatrix<T, 3> t = SquareMatrix<T, 3>(*this) * mat2.inverse();
		*this = t;

		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator/ (const T& scale) const
	{
		SquareMatrix<T, 3> res;
		res.data_[0] = data_[0] / scale;
		res.data_[1] = data_[1] / scale;
		res.data_[2] = data_[2] / scale;

		return res;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 3>& SquareMatrix<T, 3>::operator/= (const T& scale)
	{
		data_[0] /= scale;
		data_[1] /= scale;
		data_[2] /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::operator- (void) const
	{
		SquareMatrix<T, 3> res;
		res.data_[0] = -data_[0];
		res.data_[1] = -data_[1];
		res.data_[2] = -data_[2];
		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::transpose() const
	{
		SquareMatrix<T, 3> res;
		res.data_[0][0] = data_[0][0];
		res.data_[0][1] = data_[1][0];
		res.data_[0][2] = data_[2][0];

		res.data_[1][0] = data_[0][1];
		res.data_[1][1] = data_[1][1];
		res.data_[1][2] = data_[2][1];

		res.data_[2][0] = data_[0][2];
		res.data_[2][1] = data_[1][2];
		res.data_[2][2] = data_[2][2];

		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::inverse() const
	{
		T OneOverDeterminant = static_cast<T>(1) / determinant();

		SquareMatrix<T, 3> res;
		res.data_[0][0] = +(data_[1][1] * data_[2][2] - data_[2][1] * data_[1][2]) * OneOverDeterminant;
		res.data_[1][0] = -(data_[1][0] * data_[2][2] - data_[2][0] * data_[1][2]) * OneOverDeterminant;
		res.data_[2][0] = +(data_[1][0] * data_[2][1] - data_[2][0] * data_[1][1]) * OneOverDeterminant;
		res.data_[0][1] = -(data_[0][1] * data_[2][2] - data_[2][1] * data_[0][2]) * OneOverDeterminant;
		res.data_[1][1] = +(data_[0][0] * data_[2][2] - data_[2][0] * data_[0][2]) * OneOverDeterminant;
		res.data_[2][1] = -(data_[0][0] * data_[2][1] - data_[2][0] * data_[0][1]) * OneOverDeterminant;
		res.data_[0][2] = +(data_[0][1] * data_[1][2] - data_[1][1] * data_[0][2]) * OneOverDeterminant;
		res.data_[1][2] = -(data_[0][0] * data_[1][2] - data_[1][0] * data_[0][2]) * OneOverDeterminant;
		res.data_[2][2] = +(data_[0][0] * data_[1][1] - data_[1][0] * data_[0][1]) * OneOverDeterminant;

		return res;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::determinant() const
	{
		return data_[0][0] * (data_[1][1] * data_[2][2] - data_[2][1] * data_[1][2])
			 - data_[1][0] * (data_[0][1] * data_[2][2] - data_[2][1] * data_[0][2])
			 + data_[2][0] * (data_[0][1] * data_[1][2] - data_[1][1] * data_[0][2]);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::trace() const
	{
		return (*this)(0, 0) + (*this)(1, 1) + (*this)(2, 2);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::doubleContraction(const SquareMatrix<T, 3> &mat2) const
	{
		T result = 0;
		for (unsigned int i = 0; i < 3; ++i)
			for (unsigned int j = 0; j < 3; ++j)
				result += (*this)(i, j)*mat2(i, j);
		return result;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::frobeniusNorm() const
	{
		T result = 0;
		for (unsigned int i = 0; i < 3; ++i)
			for (unsigned int j = 0; j < 3; ++j)
				result += (*this)(i, j)*(*this)(i, j);
		return glm::sqrt(result);
	}


	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::oneNorm() const
	{
		const SquareMatrix<T, 3>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(1, 0)) + fabs(A(2, 0));
		const T sum2 = fabs(A(0, 1)) + fabs(A(1, 1)) + fabs(A(2, 1));
		const T sum3 = fabs(A(0, 2)) + fabs(A(1, 2)) + fabs(A(2, 2));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 3>::infNorm() const
	{
		const SquareMatrix<T, 3>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(0, 1)) + fabs(A(0, 2));
		const T sum2 = fabs(A(1, 0)) + fabs(A(1, 1)) + fabs(A(1, 2));
		const T sum3 = fabs(A(2, 0)) + fabs(A(2, 1)) + fabs(A(2, 2));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		if (sum3 > maxSum)
			maxSum = sum3;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 3> SquareMatrix<T, 3>::identityMatrix()
	{
		return SquareMatrix<T, 3>(1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0);
	}

}

