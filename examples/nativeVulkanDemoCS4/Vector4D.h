#pragma once
#include <glm/vec4.hpp>
#include <iostream>

#include "VectorBase.h"

namespace CsyVk {

	template <typename T, int Dim> class SquareMatrix;

	/*
	 * Vector<T,4> are defined for C++ fundamental integer types and floating-point types
	 */

	template <typename T>
	class Vector<T, 4>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y, T z, T w);
		DYN_FUNC Vector(const Vector<T, 4>&);
		DYN_FUNC ~Vector();

		DYN_FUNC  static int dims() { return 4; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 4> operator+ (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator+= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator- (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator-= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator* (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator*= (const Vector<T, 4> &);
		DYN_FUNC const Vector<T, 4> operator/ (const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4>& operator/= (const Vector<T, 4> &);

		DYN_FUNC Vector<T, 4>& operator= (const Vector<T, 4> &);

		DYN_FUNC bool operator== (const Vector<T, 4> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 4> &) const;

		DYN_FUNC const Vector<T, 4> operator+ (T) const;
		DYN_FUNC const Vector<T, 4> operator- (T) const;
		DYN_FUNC const Vector<T, 4> operator* (T) const;
		DYN_FUNC const Vector<T, 4> operator/ (T) const;

		DYN_FUNC Vector<T, 4>& operator+= (T);
		DYN_FUNC Vector<T, 4>& operator-= (T);
		DYN_FUNC Vector<T, 4>& operator*= (T);
		DYN_FUNC Vector<T, 4>& operator/= (T);

		DYN_FUNC const Vector<T, 4> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 4>& normalize();
		DYN_FUNC T dot(const Vector<T, 4>&) const;
		//    DYN_FUNC const SquareMatrix<T,4> outerProduct(const Vector<T,4>&) const;

		DYN_FUNC Vector<T, 4> minimum(const Vector<T, 4> &) const;
		DYN_FUNC Vector<T, 4> maximum(const Vector<T, 4> &) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

		friend std::ostream& operator<<(std::ostream &out, const Vector<T, 4>& vec)
		{
			out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << ")";
			return out;
		}
	public:
		union
		{
			glm::tvec4<T> data_; //default: zero vector
			struct { T x, y, z, w; };
		};
		
	};

	template class Vector<float, 4>;
	template class Vector<double, 4>;
	//convenient typedefs
	typedef Vector<float, 4> Vec4f;
	typedef Vector<double, 4> Vec4d;

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector()
		:Vector(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(T x)
		: Vector(x, x, x, x) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(T x, T y, T z, T w)
		: data_(x, y, z, w)
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::Vector(const Vector<T, 4>& vec2)
		: data_(vec2.data_)
	{

	}

	template <typename T>
	DYN_FUNC Vector<T, 4>::~Vector()
	{

	}

	template <typename T>
	DYN_FUNC T& Vector<T, 4>::operator[] (unsigned int idx)
	{
		return const_cast<T &> (static_cast<const Vector<T, 4> &>(*this)[idx]);
	}

	template <typename T>
	DYN_FUNC const T& Vector<T, 4>::operator[] (unsigned int idx) const
	{
		// #ifndef __CUDA_ARCH__
		//     if(idx>=4)
		//         throw PhysikaException("Vector index out of range!");
		// #endif
		return data_[idx];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator+ (const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(*this) += vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator+= (const Vector<T, 4> &vec2)
	{
		data_ += vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator- (const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(*this) -= vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator-= (const Vector<T, 4> &vec2)
	{
		data_ -= vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator*(const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(data_[0] * vec2.data_[0], data_[1] * vec2.data_[1], data_[2] * vec2.data_[2], data_[3] * vec2.data_[3]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator*=(const Vector<T, 4> &vec2)
	{
		data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2];	data_[3] *= vec2.data_[3];
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator/(const Vector<T, 4> &vec2) const
	{
		return Vector<T, 4>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2], data_[3] / vec2[3]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator/=(const Vector<T, 4> &vec2)
	{
		data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2];	data_[3] /= vec2.data_[3];
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator=(const Vector<T, 4> & vec2)
	{
		data_ = vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 4>::operator== (const Vector<T, 4> &vec2) const
	{
		return data_ == vec2.data_;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 4>::operator!= (const Vector<T, 4> &vec2) const
	{
		return !((*this) == vec2);
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator+(T value) const
	{
		return Vector<T, 4>(*this) += value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator+= (T value)
	{
		data_ += value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator-(T value) const
	{
		return Vector<T, 4>(*this) -= value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator-= (T value)
	{
		data_ -= value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator* (T scale) const
	{
		return Vector<T, 4>(*this) *= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator*= (T scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator/ (T scale) const
	{
		return Vector<T, 4>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::operator/= (T scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 4> Vector<T, 4>::operator-(void) const
	{
		Vector<T, 4> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::norm() const
	{
		return glm::length(data_);
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::normSquared() const
	{
		return glm::length2(data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4>& Vector<T, 4>::normalize()
	{
		data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec4<T>(0, 0, 0, 0);
		return *this;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 4>::dot(const Vector<T, 4>& vec2) const
	{
		return glm::dot(data_, vec2.data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 4> Vector<T, 4>::minimum(const Vector<T, 4>& vec2) const
	{
		Vector<T, 4> res;
		res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
		res[3] = data_[3] < vec2[3] ? data_[3] : vec2[3];
		return res;
	}

	template <typename T>
	DYN_FUNC Vector<T, 4> Vector<T, 4>::maximum(const Vector<T, 4>& vec2) const
	{
		Vector<T, 4> res;
		res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
		res[3] = data_[3] > vec2[3] ? data_[3] : vec2[3];
		return res;
	}

	template <typename S, typename T>
	DYN_FUNC  const Vector<T, 4> operator *(S scale, const Vector<T, 4> &vec)
	{
		return vec * (T)scale;
	}
}
