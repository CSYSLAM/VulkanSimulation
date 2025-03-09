/**
 * Copyright 2021 Xiaowei He
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "FBase.h"
#include "Vector.h"

#include "Array.h"

namespace CsyVk {
	/*!
	*	\class	Variable
	*	\brief	Variables of build-in data types.
	*/
	template<typename T>
	class FVar : public FBase
	{
	public:
		typedef T				VarType;
		typedef T				DataType;
		typedef FVar<T>			FieldType;

		FVar() : FBase("", "") {}
		FVar(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

		FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent);
		~FVar() override;

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }
		const std::string getClassName() override { return "FVar"; }

		uint size() override { return 1; }

		void setValue(T val);
		T getValue();

		std::string serialize() override { return "Unknown"; }
		bool deserialize(const std::string& str) override { return false; }

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}

		bool connect(FieldType* dst)
		{
			this->connectField(dst);
			return true;
		}

		bool connect(FBase* dst) override {
			FieldType* derived = dynamic_cast<FieldType*>(dst);
			if (derived == nullptr) return false;
			return this->connect(derived);
		}

		DataType getData() {
			auto dataPtr = this->constDataPtr();
			assert(dataPtr != nullptr);
			return *dataPtr;
		}

		std::shared_ptr<DataType>& constDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

	private:
		std::shared_ptr<DataType>& getDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

		std::shared_ptr<DataType> m_data = nullptr;
	};

	template<typename T>
	FVar<T>::FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
		: FBase(name, description, fieldType, parent)
	{
		this->setValue(value);
	}

	template<typename T>
	FVar<T>::~FVar()
	{
	};

	template<typename T>
	void FVar<T>::setValue(T val)
	{
		std::shared_ptr<T>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<T>(val);
		}
		else
		{
			*data = val;
		}

		this->update();

		this->tick();
	}


	template<typename T>
	T FVar<T>::getValue()
	{
		std::shared_ptr<T>& data = this->constDataPtr();

		return *data;
	}


	template<typename T>
	using HostVarField = FVar<T>;

	template<typename T>
	using DeviceVarField = FVar<T>;

	/**
 * Define field for Array
 */
	template<typename T, DeviceType deviceType>
	class FArray : public FBase
	{
	public:
		typedef T							VarType;
		typedef Array<T, deviceType>		DataType;
		typedef FArray<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray);

		~FArray() override;

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint num);
		void reset();

		void clear();

		void assign(const T& val);
		void assign(const std::vector<T>& vals);
#ifndef NO_BACKEND
		void assign(const DArray<T>& vals);
#endif
		void assign(const CArray<T>& vals);

		bool isEmpty() override {
			return this->size() == 0;
		}
	};

	template<typename T, DeviceType deviceType>
	FArray<T, deviceType>::~FArray()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::resize(uint num)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr) {
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->resize(num);

		//this->tick();
	}

	template<typename T, DeviceType deviceType>
	void CsyVk::FArray<T, deviceType>::assign(const T& val)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(val);

		//this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const std::vector<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);

		//this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const CArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);

		//this->tick();
	}

#ifndef NO_BACKEND
	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const DArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);

		//this->tick();
	}
#endif

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::reset()
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->reset();

		//this->tick();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::clear()
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->clear();

		//this->tick();
	}

	template<typename T>
	using HostArrayField = FArray<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArrayField = FArray<T, DeviceType::GPU>;

	template<>
	std::string FVar<bool>::serialize()
	{
		if (isEmpty())
			return "";

		bool b = this->getValue();
		return b ? "true" : "false";
	}

	template<>
	bool FVar<bool>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		bool b = str == std::string("true") ? true : false;
		this->setValue(b);

		return true;
	}

	template<>
	std::string FVar<int>::serialize()
	{
		if (isEmpty())
			return "";

		int val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<int>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		int val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<uint>::serialize()
	{
		if (isEmpty())
			return "";

		uint val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<uint>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		uint val = std::stoi(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<float>::serialize()
	{
		if (isEmpty())
			return "";

		float val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<float>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		float val = std::stof(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<double>::serialize()
	{
		if (isEmpty())
			return "";

		double val = this->getValue();

		std::stringstream ss;
		ss << val;

		return ss.str();
	}

	template<>
	bool FVar<double>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		double val = std::stod(str);
		this->setValue(val);

		return true;
	}

	template<>
	std::string FVar<Vec3f>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3f val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3f>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;
		
		ss >> substr;
		float x = std::stof(substr);

		ss >> substr;
		float y = std::stof(substr);

		ss >> substr;
		float z = std::stof(substr);


		this->setValue(Vec3f(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3i>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3i val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3i>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		int x = std::stoi(substr);

		ss >> substr;
		int y = std::stoi(substr);

		ss >> substr;
		int z = std::stoi(substr.c_str());

		this->setValue(Vec3i(x, y, z));

		return true;
	}

	template<>
	std::string FVar<Vec3d>::serialize()
	{
		if (isEmpty())
			return "";

		Vec3d val = this->getValue();

		std::stringstream ss;
		ss << val.x << " " << val.y << " " << val.z;

		return ss.str();
	}

	template<>
	bool FVar<Vec3d>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		std::stringstream ss(str);
		std::string substr;

		ss >> substr;
		double x = std::stod(substr);

		ss >> substr;
		double y = std::stod(substr);

		ss >> substr;
		double z = std::stod(substr);

		this->setValue(Vec3d(x, y, z));

		return true;
	}

	template<>
	std::string FVar<std::string>::serialize()
	{
		if (isEmpty())
			return "";

		std::string val = this->getValue();

		return val;
	}

	template<>
	bool FVar<std::string>::deserialize(const std::string& str)
	{
		if (str.empty())
			return false;

		this->setValue(str);

		return true;
	}

	template class FVar<bool>;
	template class FVar<int>;
	template class FVar<uint>;
	template class FVar<float>;
	template class FVar<double>;
	template class FVar<Vec3f>;
	template class FVar<Vec3d>;
	template class FVar<Vec3i>;
	template class FVar<std::string>;
}

