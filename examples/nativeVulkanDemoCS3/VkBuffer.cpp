#include "VkBuffer.h"
#include "VkTools.h"
#include "VkSystem.h"
#include "VkContext.h"
#include "vk_mem_alloc.h"

namespace csyvk
{	
	Buffer::Buffer(VkDevice dev)
	{
		auto ctx = CsyVk::VkSystem::instance()->currentContext();

		if (dev == nullptr && ctx == VK_NULL_HANDLE) {
			// "Vulkan library should be initialized first!;
		}

		device = dev == nullptr ? ctx->deviceHandle() : dev;
	}


	Buffer::~Buffer()
	{
	    destroy();
	}

	VkResult Buffer::map(VkDeviceSize size, VkDeviceSize offset)
	{
        if (usePool) {
            return vmaMapMemory(allocator, allocation, &mapped);
        } else {
            return vkMapMemory(device, memory, offset, size, 0, &mapped);
        }
	}

	void Buffer::unmap()
	{
        if (mapped) {
            if (usePool) {
                vmaUnmapMemory(allocator, allocation);
            } else {
                vkUnmapMemory(device, memory);
            }
            mapped = nullptr;
        }
	}

	VkResult Buffer::bind(VkDeviceSize offset)
	{
        if (usePool) {
            return vmaBindBufferMemory(allocator, allocation, buffer);
        } else {
            return vkBindBufferMemory(device, buffer, memory, offset);
        }
	}

	void Buffer::setupDescriptor(VkDeviceSize size, VkDeviceSize offset)
	{
		descriptor.offset = offset;
		descriptor.buffer = buffer;
		descriptor.range = size;
	}

	void Buffer::copyTo(void* data, VkDeviceSize size)
	{
		assert(mapped);
		memcpy(mapped, data, size);
	}

	VkResult Buffer::flush(VkDeviceSize size, VkDeviceSize offset)
	{
        if (usePool) {
            return vmaFlushAllocation(allocator, allocation, this->offset, this->size);
        } else {
            VkMappedMemoryRange mappedRange = {};
            mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mappedRange.memory = memory;
            mappedRange.offset = offset;
            mappedRange.size = size;
            return vkFlushMappedMemoryRanges(device, 1, &mappedRange);
        }
	}

	VkResult Buffer::invalidate(VkDeviceSize size, VkDeviceSize offset)
	{
	    if (usePool) {
	        return vmaInvalidateAllocation(allocator, allocation, this->offset, this->size);
	    } else {
            VkMappedMemoryRange mappedRange = {};
            mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            mappedRange.memory = memory;
            mappedRange.offset = offset;
            mappedRange.size = size;
            return vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);
        }
	}

	void Buffer::destroy()
	{
        unmap();
        if (usePool) {
            if (buffer) {
                vmaDestroyBuffer(allocator, buffer, allocation);
                buffer = VK_NULL_HANDLE;
            }
        } else {
            if (buffer) {
                vkDestroyBuffer(device, buffer, nullptr);
                buffer = VK_NULL_HANDLE;
            }
            if (memory) {
                vkFreeMemory(device, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
        }
		size = 0;
	}
};
