#pragma once
#include "VkBuffer.h"
#include "VkTools.h"
#include "vulkan/vulkan.h"
#include <algorithm>
#include <assert.h>
#include <exception>
#include <map>

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaPool)

namespace CsyVk {

	class VkContext
	{
	public:
		explicit VkContext(VkPhysicalDevice physicalDevice);
		~VkContext();

		VkResult createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char *> enabledExtensions, void *pNextChain, bool useSwapChain = true, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);

		void createPipelineCache();

		inline	VkDevice		deviceHandle() { return logicalDevice; }
		inline	VkQueue			graphicsQueueHandle() { return graphicsQueue; }
		inline	VkQueue			computeQueueHandle() { return computeQueue; }
		inline	VkQueue			transferQueueHandle() { return transferQueue; }

		inline	VkPhysicalDevice	physicalDeviceHandle() { return physicalDevice; }

		inline  VkPipelineCache	pipelineCacheHandle() { return pipelineCache; }

		// Check whether the compute queue family is distinct from the graphics queue family
		bool isComputeQueueSpecial();

		uint32_t        getMemoryType(uint32_t  typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound = nullptr) const;
		uint32_t        getQueueFamilyIndex(VkQueueFlagBits queueFlags) const;
		VkResult        createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory, void *data = nullptr);
		VkResult        createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, std::shared_ptr<csyvk::Buffer> &buffer, VkDeviceSize size, const void *data = nullptr);
        VkResult        createBuffer(uint32_t poolType, std::shared_ptr<csyvk::Buffer> &buffer, const void *data = nullptr);
        VkResult        createMemoryPool(VkInstance instance, uint32_t apiVerion);
		void            copyBuffer(csyvk::Buffer *src, csyvk::Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion = nullptr);
		VkCommandPool   createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin = false);
		VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin = false);
		void            flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free = true);
		void            flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free = true);
		bool            extensionSupported(std::string extension);
		VkFormat        getSupportedDepthFormat(bool checkSamplingSupport);

	public:

		VkPhysicalDevice physicalDevice;
		VkDevice logicalDevice;
		VkPhysicalDeviceProperties properties;
		VkPhysicalDeviceFeatures features;
		VkPhysicalDeviceFeatures enabledFeatures;
		VkPhysicalDeviceMemoryProperties memoryProperties;
		std::vector<VkQueueFamilyProperties> queueFamilyProperties;
		std::vector<std::string> supportedExtensions;
		VkCommandPool commandPool = VK_NULL_HANDLE;

		VkQueue graphicsQueue;
		VkQueue computeQueue;
		VkQueue transferQueue;
		bool enableDebugMarkers = false;

		// Pipeline cache object
		VkPipelineCache pipelineCache;

		struct
		{
			uint32_t graphics;
			uint32_t compute;
			uint32_t transfer;
		} queueFamilyIndices;
		operator VkDevice() const
		{
			return logicalDevice;
		};

		enum MemPoolType
        {
		    DevicePool,
		    HostPool,
		    UniformPool,
		    EndType
        };

		struct MemoryPoolInfo
        {
		    VmaPool pool;
		    int32_t usage;
        };

		std::map<VkFlags, MemoryPoolInfo> poolMap;
		VmaAllocator g_Allocator;
		bool useMemoryPool = false;
	};
}