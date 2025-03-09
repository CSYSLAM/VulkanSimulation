#pragma once
#include <glm/vec3.hpp>
#include <iostream>

#include "VectorBase.h"

namespace CsyVk {

	template <typename T, int Dim> class SquareMatrix;

	/*
	 * Vector<T,3> are defined for C++ fundamental integer types and floating-point types
	 */

	template <typename T>
	class Vector<T, 3>
	{
	public:
		typedef T VarType;

		DYN_FUNC Vector();
		DYN_FUNC explicit Vector(T);
		DYN_FUNC Vector(T x, T y, T z);
		DYN_FUNC Vector(const Vector<T, 3>&);
		DYN_FUNC ~Vector();

		DYN_FUNC static int dims() { return 3; }

		DYN_FUNC T& operator[] (unsigned int);
		DYN_FUNC const T& operator[] (unsigned int) const;

		DYN_FUNC const Vector<T, 3> operator+ (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator+= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator- (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator-= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator* (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator*= (const Vector<T, 3> &);
		DYN_FUNC const Vector<T, 3> operator/ (const Vector<T, 3> &) const;
		DYN_FUNC Vector<T, 3>& operator/= (const Vector<T, 3> &);


		DYN_FUNC Vector<T, 3>& operator= (const Vector<T, 3> &);

		DYN_FUNC bool operator== (const Vector<T, 3> &) const;
		DYN_FUNC bool operator!= (const Vector<T, 3> &) const;

		DYN_FUNC const Vector<T, 3> operator* (T) const;
		DYN_FUNC const Vector<T, 3> operator- (T) const;
		DYN_FUNC const Vector<T, 3> operator+ (T) const;
		DYN_FUNC const Vector<T, 3> operator/ (T) const;

		DYN_FUNC Vector<T, 3>& operator+= (T);
		DYN_FUNC Vector<T, 3>& operator-= (T);
		DYN_FUNC Vector<T, 3>& operator*= (T);
		DYN_FUNC Vector<T, 3>& operator/= (T);

		DYN_FUNC const Vector<T, 3> operator - (void) const;

		DYN_FUNC T norm() const;
		DYN_FUNC T normSquared() const;
		DYN_FUNC Vector<T, 3>& normalize();
		DYN_FUNC Vector<T, 3> cross(const Vector<T, 3> &) const;
		DYN_FUNC T dot(const Vector<T, 3>&) const;
		//    DYN_FUNC const SquareMatrix<T,3> outerProduct(const Vector<T,3>&) const;

		DYN_FUNC Vector<T, 3> minimum(const Vector<T, 3>&) const;
		DYN_FUNC Vector<T, 3> maximum(const Vector<T, 3>&) const;

		DYN_FUNC T* getDataPtr() { return &data_.x; }

		friend std::ostream& operator<<(std::ostream &out, const Vector<T, 3>& vec)
		{
			out << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
			return out;
		}

	public:
		union
		{
			DYN_ALIGN_16 glm::tvec3<T> data_; //default: zero vector
			struct { T x, y, z, dummy; };
		};
	};

	template class Vector<float, 3>;
	template class Vector<double, 3>;
	//convenient typedefs 
	typedef Vector<float, 3>	Vec3f;
	typedef Vector<double, 3>	Vec3d;
	typedef Vector<int, 3>		Vec3i;
	typedef Vector<uint, 3>		Vec3u;
	typedef Vector<char, 3>		Vec3c;
	typedef Vector<uchar, 3> Vec3uc;


	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector()
		:Vector(0) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(T x)
		: Vector(x, x, x) //delegating ctor
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(T x, T y, T z)
		: data_(x, y, z)
	{
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::Vector(const Vector<T, 3>& vec)
		: data_(vec.data_)
	{

	}

	template <typename T>
	DYN_FUNC Vector<T, 3>::~Vector()
	{
	}

	template <typename T>
	DYN_FUNC T& Vector<T, 3>::operator[] (unsigned int idx)
	{
		return const_cast<T &> (static_cast<const Vector<T, 3> &>(*this)[idx]);
	}

	template <typename T>
	DYN_FUNC const T& Vector<T, 3>::operator[] (unsigned int idx) const
	{
		return data_[idx];
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator+ (const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(*this) += vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator+= (const Vector<T, 3> &vec2)
	{
		data_ += vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator- (const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(*this) -= vec2;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator-= (const Vector<T, 3> &vec2)
	{
		data_ -= vec2.data_;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator*(const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(data_[0] * vec2[0], data_[1] * vec2[1], data_[2] * vec2[2]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator*=(const Vector<T, 3> &vec2)
	{
		data_[0] *= vec2.data_[0];	data_[1] *= vec2.data_[1];	data_[2] *= vec2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator/(const Vector<T, 3> &vec2) const
	{
		return Vector<T, 3>(data_[0] / vec2[0], data_[1] / vec2[1], data_[2] / vec2[2]);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator/=(const Vector<T, 3> &vec2)
	{
		data_[0] /= vec2.data_[0];	data_[1] /= vec2.data_[1];	data_[2] /= vec2.data_[2];
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator=(const Vector<T, 3> &vec2)
	{
		data_ = vec2.data_;
		return *this;
	}


	template <typename T>
	DYN_FUNC bool Vector<T, 3>::operator== (const Vector<T, 3> &vec2) const
	{
		return data_ == vec2.data_;
	}

	template <typename T>
	DYN_FUNC bool Vector<T, 3>::operator!= (const Vector<T, 3> &vec2) const
	{
		return !((*this) == vec2);
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator+(T value) const
	{
		return Vector<T, 3>(*this) += value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator+= (T value)
	{
		data_ += value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator-(T value) const
	{
		return Vector<T, 3>(*this) -= value;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator-= (T value)
	{
		data_ -= value;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator* (T scale) const
	{
		return Vector<T, 3>(*this) *= scale;
	}

	template <typename T>
	Vector<T, 3>& Vector<T, 3>::operator*= (T scale)
	{
		data_ *= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator/ (T scale) const
	{
		return Vector<T, 3>(*this) /= scale;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::operator/= (T scale)
	{
		data_ /= scale;
		return *this;
	}

	template <typename T>
	DYN_FUNC const Vector<T, 3> Vector<T, 3>::operator-(void) const
	{
		Vector<T, 3> res;
		res.data_ = -data_;
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::norm() const
	{
		return glm::length(data_);
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::normSquared() const
	{
		return glm::length2(data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3>& Vector<T, 3>::normalize()
	{
		data_ = glm::length(data_) > glm::epsilon<T>() ? glm::normalize(data_) : glm::tvec3<T>(0, 0, 0);
		return *this;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::cross(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res.data_ = glm::cross(data_, vec2.data_);
		return res;
	}

	template <typename T>
	DYN_FUNC T Vector<T, 3>::dot(const Vector<T, 3>& vec2) const
	{
		return glm::dot(data_, vec2.data_);
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::minimum(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res[0] = data_[0] < vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] < vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] < vec2[2] ? data_[2] : vec2[2];
		return res;
	}

	template <typename T>
	DYN_FUNC Vector<T, 3> Vector<T, 3>::maximum(const Vector<T, 3>& vec2) const
	{
		Vector<T, 3> res;
		res[0] = data_[0] > vec2[0] ? data_[0] : vec2[0];
		res[1] = data_[1] > vec2[1] ? data_[1] : vec2[1];
		res[2] = data_[2] > vec2[2] ? data_[2] : vec2[2];
		return res;
	}

	//make * operator commutative
	template <typename S, typename T>
	DYN_FUNC const Vector<T, 3> operator *(S scale, const Vector<T, 3> &vec)
	{
		return vec * (T)scale;
	}

}

