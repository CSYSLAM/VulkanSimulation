#pragma once
#include "VkVariable.h"
#include "VkBuffer.h"

namespace CsyVk {

	template<typename T>
	class VkHostArray : public VkVariable
	{

	public:
		VkHostArray() {};

		~VkHostArray();

		void resize(uint32_t num, const T* data = nullptr);
		inline uint32_t size() const { return m_num; }

		VariableType type() override;

		uint32_t bufferSize() override { return sizeof(T)*m_num; }

		void clear();

		void* mapped();
		void unmap();

		uint32_t m_num = 0;
	};

	template<typename T>
	VkHostArray<T>::~VkHostArray()
	{
	}

	template<typename T>
	void VkHostArray<T>::resize(uint32_t num, const T* data)
	{
		uint32_t newSize = num * sizeof(T);
		uint32_t bufferSize = this->bufferSize();

		if (newSize > bufferSize)
		{
			m_num = num;

			buffer->destroy();

			if (num > 0) {
				if (ctx->useMemoryPool) {
					buffer->size = newSize;
					buffer->usageFlags = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
					buffer->memoryPropertyFlags =
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
					ctx->createBuffer(VkContext::HostPool, buffer, data);
				}
				else {
					ctx->createBuffer(
						VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
						VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						buffer,
						newSize,
						data);
				}
			}
		}
		else
		{
			m_num = num;
		}
	}

	template<typename T>
	VariableType VkHostArray<T>::type()
	{
		return VariableType::HostBuffer;
	}

	template<typename T>
	void VkHostArray<T>::clear()
	{
		buffer->destroy();
	}

	template<typename T>
	void* VkHostArray<T>::mapped()
	{
		VK_CHECK_RESULT(buffer->map());
		return buffer->mapped;
	}

	template<typename T>
	void VkHostArray<T>::unmap()
	{
		buffer->unmap();
	}
}