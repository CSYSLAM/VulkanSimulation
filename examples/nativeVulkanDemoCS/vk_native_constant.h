#pragma once
#include "vk_native_variable.h"

namespace CsyVkN {

	template<typename T>
	class VkConstant : public VkVariable
	{
	public:
		VkConstant();
		VkConstant(T val);
		~VkConstant();

		void setValue(const T val);
		T getValue();

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T); }

		void* data() const override { return (void*)&mVal; }

	protected:
		T mVal;
	};

	template<typename T>
	VkConstant<T>::VkConstant() : VkVariable() {}

	template<typename T>
	VkConstant<T>::VkConstant(T val)
	{
		mVal = val;
	}

	template<typename T>
	VkConstant<T>::~VkConstant() {}

	template<typename T>
	void VkConstant<T>::setValue(const T val)
	{
		mVal = val;
	}

	template<typename T>
	T VkConstant<T>::getValue()
	{
		return mVal;
	}

	template<typename T>
	VariableType VkConstant<T>::type()
	{
		return VariableType::Constant;
	}
}