#pragma once
#include <glm/mat2x2.hpp>
#include "SquareMatrix.h"

namespace CsyVk {

	template <typename T, int Dim> class Vector;

	template <typename T>
	class SquareMatrix<T, 2>
	{
	public:
		typedef T VarType;

		DYN_FUNC SquareMatrix();
		DYN_FUNC explicit SquareMatrix(T);
		DYN_FUNC SquareMatrix(T x00, T x01, T x10, T x11);
		DYN_FUNC SquareMatrix(const Vector<T, 2> &row1, const Vector<T, 2> &row2);

		DYN_FUNC SquareMatrix(const SquareMatrix<T, 2> &);
		DYN_FUNC ~SquareMatrix();

		DYN_FUNC static unsigned int rows() { return 2; }
		DYN_FUNC static unsigned int cols() { return 2; }

		DYN_FUNC T& operator() (unsigned int i, unsigned int j);
		DYN_FUNC const T& operator() (unsigned int i, unsigned int j) const;

		DYN_FUNC const Vector<T, 2> row(unsigned int i) const;
		DYN_FUNC const Vector<T, 2> col(unsigned int i) const;

		DYN_FUNC void setRow(unsigned int i, const Vector<T, 2>& vec);
		DYN_FUNC void setCol(unsigned int j, const Vector<T, 2>& vec);

		DYN_FUNC const SquareMatrix<T, 2> operator+ (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator+= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator- (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator-= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator* (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator*= (const SquareMatrix<T, 2> &);
		DYN_FUNC const SquareMatrix<T, 2> operator/ (const SquareMatrix<T, 2> &) const;
		DYN_FUNC SquareMatrix<T, 2>& operator/= (const SquareMatrix<T, 2> &);

		DYN_FUNC SquareMatrix<T, 2>& operator= (const SquareMatrix<T, 2> &);

		DYN_FUNC bool operator== (const SquareMatrix<T, 2> &) const;
		DYN_FUNC bool operator!= (const SquareMatrix<T, 2> &) const;

		DYN_FUNC const SquareMatrix<T, 2> operator* (const T&) const;
		DYN_FUNC SquareMatrix<T, 2>& operator*= (const T&);
		DYN_FUNC const SquareMatrix<T, 2> operator/ (const T&) const;
		DYN_FUNC SquareMatrix<T, 2>& operator/= (const T&);

		DYN_FUNC const Vector<T, 2> operator* (const Vector<T, 2> &) const;

		DYN_FUNC const SquareMatrix<T, 2> operator- (void) const;

		DYN_FUNC const SquareMatrix<T, 2> transpose() const;
		DYN_FUNC const SquareMatrix<T, 2> inverse() const;

		DYN_FUNC T determinant() const;
		DYN_FUNC T trace() const;
		DYN_FUNC T doubleContraction(const SquareMatrix<T, 2> &) const;//double contraction
		DYN_FUNC T frobeniusNorm() const;
		DYN_FUNC T oneNorm() const;
		DYN_FUNC T infNorm() const;

		DYN_FUNC static const SquareMatrix<T, 2> identityMatrix();

		DYN_FUNC T* getDataPtr() { return &data_[0].x; }

	protected:
		glm::tmat2x2<T> data_; //default: zero matrix
	};

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC const SquareMatrix<T, 2> operator* (S scale, const SquareMatrix<T, 2> &mat)
	{
		return mat * scale;
	}

	template class SquareMatrix<float, 2>;
	template class SquareMatrix<double, 2>;
	//convenient typedefs
	typedef SquareMatrix<float, 2> Mat2f;
	typedef SquareMatrix<double, 2> Mat2d;

}  //end of namespace dyno

#include <cmath>
#include <limits>

#include "Vector.h"

namespace CsyVk 
{
	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::SquareMatrix()
		:SquareMatrix(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::SquareMatrix(T value)
		: SquareMatrix(value, value, value, value) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::SquareMatrix(T x00, T x01, T x10, T x11)
		: data_(x00, x10, x01, x11)
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::SquareMatrix(const Vector<T, 2> &row1, const Vector<T, 2> &row2)
		: data_(row1[0], row2[0], row1[1], row2[1])
	{
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::SquareMatrix(const SquareMatrix<T, 2> &mat)
	{
		data_ = mat.data_;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>::~SquareMatrix()
	{

	}

	template <typename T>
	DYN_FUNC T& SquareMatrix<T, 2>::operator() (unsigned int i, unsigned int j)
	{
		return const_cast<T &>(static_cast<const SquareMatrix<T, 2> &>(*this)(i, j));
	}

	template <typename T>
	DYN_FUNC const T& SquareMatrix<T, 2>::operator() (unsigned int i, unsigned int j) const
	{
		return data_[j][i];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> SquareMatrix<T, 2>::row(unsigned int i) const
	{
		Vector<T, 2> result((*this)(i, 0), (*this)(i, 1));
		return result;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> SquareMatrix<T, 2>::col(unsigned int i) const
	{
		Vector<T, 2> result((*this)(0, i), (*this)(1, i));
		return result;
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 2>::setRow(unsigned int i, const Vector<T, 2>& vec)
	{
		data_[0][i] = vec[0];
		data_[1][i] = vec[1];
	}

	template <typename T>
	DYN_FUNC void SquareMatrix<T, 2>::setCol(unsigned int j, const Vector<T, 2>& vec)
	{
		data_[j][0] = vec[0];
		data_[j][1] = vec[1];
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator+ (const SquareMatrix<T, 2> &mat2) const
	{
		return SquareMatrix<T, 2>(*this) += mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator+= (const SquareMatrix<T, 2> &mat2)
	{
		data_ += mat2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator- (const SquareMatrix<T, 2> &mat2) const
	{
		return SquareMatrix<T, 2>(*this) -= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator-= (const SquareMatrix<T, 2> &mat2)
	{
		data_ -= mat2.data_;
		return *this;
	}


	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator=(const SquareMatrix<T, 2> &mat)
	{
		data_ = mat.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 2>::operator== (const SquareMatrix<T, 2> &mat2) const
	{
		return data_ == mat2.data_;
	}

	template <typename T>
	DYN_FUNC bool SquareMatrix<T, 2>::operator!= (const SquareMatrix<T, 2> &mat2) const
	{
		return !((*this) == mat2);
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator* (const T& scale) const
	{
		return SquareMatrix<T, 2>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator*= (const T& scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 2> SquareMatrix<T, 2>::operator* (const Vector<T, 2> &vec) const
	{
		Vector<T, 2> result(0);
		for (unsigned int i = 0; i < 2; ++i)
			for (unsigned int j = 0; j < 2; ++j)
				result[i] += (*this)(i, j) * vec[j];
		return result;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator* (const SquareMatrix<T, 2> &mat2) const
	{
		return SquareMatrix<T, 2>(*this) *= mat2;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator*= (const SquareMatrix<T, 2> &mat2)
	{
		data_ *= mat2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator/ (const SquareMatrix<T, 2> &mat2) const
	{
		return SquareMatrix<T, 2>(*this) *= mat2.inverse();
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator/= (const SquareMatrix<T, 2> &mat2)
	{
		data_ *= glm::inverse(mat2.data_);
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator/ (const T& scale) const
	{
		return SquareMatrix<T, 2>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC SquareMatrix<T, 2>& SquareMatrix<T, 2>::operator/= (const T& scale)
	{
		// #ifndef __CUDA_ARCH__
		//     if (abs(scale) <= std::numeric_limits<T>::epsilon())
		//         throw PhysikaException("Matrix Divide by zero error!");
		// #endif
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::operator- (void) const
	{
		SquareMatrix<T, 2> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::transpose() const
	{
		SquareMatrix<T, 2> res;
		res.data_ = glm::transpose(data_);
		return res;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::inverse() const
	{
		SquareMatrix<T, 2> res;
		res.data_ = glm::inverse(data_);

		return res;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::determinant() const
	{
		return glm::determinant(data_);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::trace() const
	{
		return (*this)(0, 0) + (*this)(1, 1);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::doubleContraction(const SquareMatrix<T, 2> &mat2) const
	{
		T result = 0;
		for (unsigned int i = 0; i < 2; ++i)
			for (unsigned int j = 0; j < 2; ++j)
				result += (*this)(i, j)*mat2(i, j);
		return result;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::frobeniusNorm() const
	{
		T result = 0;
		for (unsigned int i = 0; i < 2; ++i)
			for (unsigned int j = 0; j < 2; ++j)
				result += (*this)(i, j)*(*this)(i, j);
		return glm::sqrt(result);
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::oneNorm() const
	{
		const SquareMatrix<T, 2>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(1, 0));
		const T sum2 = fabs(A(0, 1)) + fabs(A(1, 1));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC T SquareMatrix<T, 2>::infNorm() const
	{
		const SquareMatrix<T, 2>& A = (*this);
		const T sum1 = fabs(A(0, 0)) + fabs(A(0, 1));
		const T sum2 = fabs(A(1, 0)) + fabs(A(1, 1));
		T maxSum = sum1;
		if (sum2 > maxSum)
			maxSum = sum2;
		return maxSum;
	}

	template <typename T>
	DYN_FUNC const SquareMatrix<T, 2> SquareMatrix<T, 2>::identityMatrix()
	{
		return SquareMatrix<T, 2>(1.0, 0.0,
			0.0, 1.0);
	}
}

