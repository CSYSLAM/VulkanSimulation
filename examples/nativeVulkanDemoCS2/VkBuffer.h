#pragma once

#include <vector>

#include "vulkan/vulkan.h"


VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace csyvk
{
	class Buffer
	{
	public:
		Buffer(VkDevice dev = nullptr);
		~Buffer();
		VkResult map(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		void unmap();
		VkResult bind(VkDeviceSize offset = 0);
		void setupDescriptor(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		void copyTo(void* data, VkDeviceSize size);
		VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		VkResult invalidate(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
		void destroy();

		VkDevice device;
		VkBuffer buffer = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkDescriptorBufferInfo descriptor;
		VkDeviceSize size = 0;
		VkDeviceSize alignment = 0;
		VkDeviceSize offset = 0;
		VmaAllocation allocation = VK_NULL_HANDLE;
		VmaAllocator allocator = VK_NULL_HANDLE;
		VkBool32 usePool = VK_FALSE;
		void* mapped = nullptr;
		VkBufferUsageFlags usageFlags;
		VkMemoryPropertyFlags memoryPropertyFlags;
	};
}