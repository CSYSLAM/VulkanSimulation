#pragma once

#include <vulkan/vulkan.h>
#include"vk_base_device_csy.h"

namespace BufferUtilsCsy {
    void CreateBuffer(DeviceCsy* device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void CopyBuffer(DeviceCsy* device, VkCommandPool commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void CreateBufferFromData(DeviceCsy* device, VkCommandPool commandPool, void* bufferData, VkDeviceSize bufferSize, VkBufferUsageFlags bufferUsage, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
}
