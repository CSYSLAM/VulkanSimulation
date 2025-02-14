#pragma once
#include "VkTypeDefine.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include "VkDeviceArray.h"
#include "VkTransfer.h"

namespace CsyVk {

	template<typename T, DeviceType deviceType> class Array;

	template<typename T>
	class Array<T, DeviceType::CPU>
	{
	public:
		Array() {};

		Array(uint num)
		{
			mData.resize((size_t)num);
		}

		~Array() {};

		void resize(uint n);
		void reset();
		void clear();

		inline const T*	begin() const { return mData.size() == 0 ? nullptr : &mData[0]; }
		inline T*	begin() { return mData.size() == 0 ? nullptr : &mData[0]; }

		inline const std::vector<T>* handle() const { return &mData; }
		inline std::vector<T>* handle() { return &mData; }

		DeviceType	deviceType() { return DeviceType::CPU; }

		inline T& operator [] (unsigned int id)
		{
			return mData[id];
		}

		inline const T& operator [] (unsigned int id) const
		{
			return mData[id];
		}

		inline uint size() const { return (uint)mData.size(); }
		inline bool isCPU() const { return true; }
		inline bool isGPU() const { return false; }
		inline bool isEmpty() const { return mData.empty(); }

		inline void pushBack(T ele) { mData.push_back(ele); }

		void assign(const T& val);
		void assign(uint num, const T& val);

	#ifndef NO_BACKEND
		void assign(const Array<T, DeviceType::GPU>& src);
	#endif

		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		friend std::ostream& operator<<(std::ostream &out, const Array<T, DeviceType::CPU>& cArray)
		{
			for (uint i = 0; i < cArray.size(); i++)
			{
				out << i << ": " << cArray[i] << std::endl;
			}

			return out;
		}

	private:
		std::vector<T> mData;
	};

	template<typename T>
	void Array<T, DeviceType::CPU>::resize(const uint n)
	{
		mData.resize(n);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::clear()
	{
		mData.clear();
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::reset()
	{
		memset((void*)mData.data(), 0, mData.size()*sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		memcpy(this->begin(), src.begin(), src.size() * sizeof(T));
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const std::vector<T>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		mData.assign(src.begin(), src.end());
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const T& val)
	{
		mData.assign(mData.size(), val);
	}

	template<typename T>
	void Array<T, DeviceType::CPU>::assign(uint num, const T& val)
	{
		mData.assign(num, val);
	}

	template<typename T>
	using CArray = Array<T, DeviceType::CPU>;
}

namespace CsyVk 
{
	template<typename T>
	void Array<T, DeviceType::CPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
	}

	template<typename T>
	class Array<T, DeviceType::GPU>
	{
	public:
		Array() {};

		Array(uint num)
		{
			this->resize(num);
		}

		~Array() {};

		void resize(const uint n);

		void reset();

		void clear();

		inline const VkDeviceArray<T>* handle() const { return &mData; }
		inline VkDeviceArray<T>* handle() { return &mData; }

		VkBuffer buffer() const { return mData.bufferHandle(); }

		uint32_t bufferSize() { return mData.bufferSize(); }

		DeviceType	deviceType() { return DeviceType::GPU; }

		 inline T& operator [] (unsigned int id) {
			return mData[id];
		}

		inline T& operator [] (unsigned int id) const {
			return mData[id];
		}

		inline uint size() const { return mData.size(); }
		inline bool isCPU() const { return false; }
		inline bool isGPU() const { return true; }
		inline bool isEmpty() const { return mData.size() == 0; }

		void assign(const Array<T, DeviceType::GPU>& src);
		void assign(const Array<T, DeviceType::CPU>& src);
		void assign(const std::vector<T>& src);

		void assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
		void assign(const std::vector<T>& src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);

		void assignFromVkBuffer(VkBuffer vkBuffer, VkDeviceSize size, VkDevice device, VkQueue queue, VkCommandPool commandPool) {
			// 创建一个命令缓冲区来执行复制操作
			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandPool = commandPool;
			allocInfo.commandBufferCount = 1;

			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

			vkBeginCommandBuffer(commandBuffer, &beginInfo);

			// 设置缓冲区复制区域
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, vkBuffer, mData.buffer(), 1, &copyRegion);

			vkEndCommandBuffer(commandBuffer);

			// 提交命令缓冲区
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;

			vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
			vkQueueWaitIdle(queue);

			// 释放命令缓冲区
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}

		friend std::ostream& operator<<(std::ostream& out, const Array<T, DeviceType::GPU>& dArray)
		{
			Array<T, DeviceType::CPU> hArray;
			hArray.assign(dArray);

			out << hArray;

			return out;
		}

	private:
		VkDeviceArray<T> mData;
	};

	template<typename T>
	using DArray = Array<T, DeviceType::GPU>;

	template<typename T>
	void Array<T, DeviceType::GPU>::resize(const uint n)
	{
		if (mData.size() == n) return;

		if (n == 0) {
			mData.clear();
			return;
		}

		mData.resize(n);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::clear()
	{
		mData.clear();
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::reset()
	{
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src)
	{
		if (src.size() == 0)
		{
			mData.clear();
			return;
		}

		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src)
	{
		if (mData.size() != src.size())
			this->resize(src.size());

		vkTransfer(mData, *src.handle());
	}


	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src)
	{
		if (mData.size() != src.size())
			this->resize((uint)src.size());

		vkTransfer(mData, src);
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const std::vector<T>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::CPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
	}

	template<typename T>
	void Array<T, DeviceType::GPU>::assign(const Array<T, DeviceType::GPU>& src, const uint count, const uint dstOffset, const uint srcOffset)
	{
		vkTransfer(mData, (uint64_t)dstOffset, *src.handle(), (uint64_t)srcOffset, (uint64_t)count);
	}
}
