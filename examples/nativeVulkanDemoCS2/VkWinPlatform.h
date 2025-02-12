#pragma once
#define PERIDYNO_VERSION 0.0.1
#define PERIDYNO_VERSION_MAJOR 0
#define PERIDYNO_VERSION_MINOR 0
#define PERIDYNO_VERSION_PATCH 1

#define PERIDYNO_EXPORT
#define PERIDYNO_IMPORT

#define PERIDYNO_API PERIDYNO_IMPORT

#define VK_BACKEND
#define DYN_FUNC
#define GPU_FUNC 
#define CPU_FUNC 

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

#define PRECISION_FLOAT

#include <list>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <limits>


typedef float Real;

namespace CsyVk {

	using uint = unsigned int;

	using uchar = unsigned char;
	using uint64 = unsigned long long;
	using int64 = signed long long;

#define INVALID -1
#define M_PI 3.14159265358979323846
#define M_E 2.71828182845904523536

	constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();
	constexpr Real REAL_MAX = (std::numeric_limits<Real>::max)();
	constexpr Real REAL_MIN = (std::numeric_limits<Real>::min)();
	constexpr uint BLOCK_SIZE = 64;

	class Bool
	{
	public:
		DYN_FUNC Bool(bool v = false) { val = v ? 1 : 0; }

		DYN_FUNC inline bool operator! () const {
			return 1 - val ? true : false;
		}

		DYN_FUNC inline bool operator== (bool v) const {
			uint tmpV = v ? 1 : 0;
			return val == tmpV;
		}

		DYN_FUNC inline bool operator== (const Bool& v) const {
			return val == v.val;
		}

		DYN_FUNC inline Bool& operator= (const bool v) {
			val = v ? 1 : 0;
			return *this;
		}

		DYN_FUNC inline Bool& operator= (const Bool& v) {
			val = v.val;
			return *this;
		}

		DYN_FUNC inline Bool& operator&= (const bool v) {
			val &= (v ? 1 : 0);
			return *this;
		}

		DYN_FUNC inline Bool& operator|= (const bool v) {
			val |= (v ? 1 : 0);
			return *this;
		}

		DYN_FUNC inline Bool operator& (const bool v) const {
			Bool ret;
			ret.val = val & (v ? 1 : 0);
			return ret;
		}

		DYN_FUNC inline Bool operator| (const bool v) const {
			Bool ret;
			ret.val = val | (v ? 1 : 0);
			return ret;
		}

		DYN_FUNC inline Bool& operator&= (const Bool& v) {
			val &= v.val;
			return *this;
		}

		DYN_FUNC inline Bool& operator|= (const Bool& v) {
			val |= v.val;
			return *this;
		}

		DYN_FUNC inline Bool operator& (const Bool& v) const {
			Bool ret;
			ret.val = val & v.val;
			return ret;
		}

		DYN_FUNC inline Bool operator| (const Bool& v) const {
			Bool ret;
			ret.val = val | v.val;
			return ret;
		}

		DYN_FUNC inline bool operator&& (const Bool& v) const {
			return val & v.val;
		}

		DYN_FUNC inline bool operator|| (const Bool& v) const {
			return val | v.val;
		}

		DYN_FUNC inline bool operator&& (const bool& v) const {
			uint tmpV = v ? 1 : 0;
			return val & tmpV;
		}

		DYN_FUNC inline bool operator|| (const bool& v) const {
			uint tmpV = v ? 1 : 0;
			return val | tmpV;
		}

		DYN_FUNC inline operator bool() const {
			return val == 1;
		}


	private:
		uint val = 0;
	};
}// end of namespace dyno


namespace TypeInfo
{
	template<class T, class ... Args>
	std::shared_ptr<T> New(Args&& ... args) { std::shared_ptr<T> p(new T(std::forward<Args>(args)...)); return p; }

	template<class TA, class TB>
	inline TA* cast(TB* b)
	{
		TA* ptr = dynamic_cast<TA*>(b);
		return ptr;
	}

	template<class TA, class TB>
	inline std::shared_ptr<TA> cast(std::shared_ptr<TB> b)
	{
		std::shared_ptr<TA> ptr = std::dynamic_pointer_cast<TA>(b);
		return ptr;
	}
}



