#include "VkContext.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace CsyVk {
	VkContext::VkContext(VkPhysicalDevice physicalDevice)
	{
		assert(physicalDevice);
		this->physicalDevice = physicalDevice;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		assert(queueFamilyCount > 0);
		queueFamilyProperties.resize(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

		uint32_t extCount = 0;
		vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
		if (extCount > 0)
		{
			std::vector<VkExtensionProperties> extensions(extCount);
			if (vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
			{
				for (auto ext : extensions)
				{
					supportedExtensions.push_back(ext.extensionName);
				}
			}
		}
	}

	VkContext::~VkContext()
	{
	    for (const auto &pool : poolMap)
        {
	        vmaDestroyPool(g_Allocator, pool.second.pool);
        }
	    vmaDestroyAllocator(g_Allocator);

		if (commandPool)
		{
			vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
		}
		if (logicalDevice)
		{
			vkDestroyDevice(logicalDevice, nullptr);
		}
	}


	bool VkContext::isComputeQueueSpecial()
	{
		return this->queueFamilyIndices.graphics != this->queueFamilyIndices.compute;
	}

	uint32_t VkContext::getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32 *memTypeFound) const
	{
		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					if (memTypeFound)
					{
						*memTypeFound = true;
					}
					return i;
				}
			}
			typeBits >>= 1;
		}

		if (memTypeFound)
		{
			*memTypeFound = false;
			return 0;
		}
		else
		{
			throw std::runtime_error("Could not find a matching memory type");
		}
	}

	uint32_t VkContext::getQueueFamilyIndex(VkQueueFlagBits queueFlags) const
	{
		if (queueFlags & VK_QUEUE_COMPUTE_BIT)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
			{
				if ((queueFamilyProperties[i].queueFlags & queueFlags) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0))
				{
					return i;
				}
			}
		}

		if (queueFlags & VK_QUEUE_TRANSFER_BIT)
		{
			for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
			{
				if ((queueFamilyProperties[i].queueFlags & queueFlags) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0))
				{
					return i;
				}
			}
		}

		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++)
		{
			if (queueFamilyProperties[i].queueFlags & queueFlags)
			{
				return i;
			}
		}

		throw std::runtime_error("Could not find a matching queue family index");
	}

	VkResult VkContext::createLogicalDevice(VkPhysicalDeviceFeatures enabledFeatures, std::vector<const char*> enabledExtensions, void* pNextChain, bool useSwapChain, VkQueueFlags requestedQueueTypes)
	{
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
		const float defaultQueuePriority(0.0f);

		// Graphics queue
		if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT)
		{
			queueFamilyIndices.graphics = getQueueFamilyIndex(VK_QUEUE_GRAPHICS_BIT);
			VkDeviceQueueCreateInfo queueInfo{};
			queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueInfo.queueFamilyIndex = queueFamilyIndices.graphics;
			queueInfo.queueCount = 1;
			queueInfo.pQueuePriorities = &defaultQueuePriority;
			queueCreateInfos.push_back(queueInfo);
		}
		else
		{
			queueFamilyIndices.graphics = 0;
		}

		// Dedicated compute queue
		if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT)
		{
			queueFamilyIndices.compute = getQueueFamilyIndex(VK_QUEUE_COMPUTE_BIT);
			if (queueFamilyIndices.compute != queueFamilyIndices.graphics)
			{
				// If compute family index differs, we need an additional queue create info for the compute queue
				VkDeviceQueueCreateInfo queueInfo{};
				queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueInfo.queueFamilyIndex = queueFamilyIndices.compute;
				queueInfo.queueCount = 1;
				queueInfo.pQueuePriorities = &defaultQueuePriority;
				queueCreateInfos.push_back(queueInfo);
			}
		}
		else
		{
			// Else we use the same queue
			queueFamilyIndices.compute = queueFamilyIndices.graphics;
		}

		// Dedicated transfer queue
		if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT)
		{
			queueFamilyIndices.transfer = getQueueFamilyIndex(VK_QUEUE_TRANSFER_BIT);
			if ((queueFamilyIndices.transfer != queueFamilyIndices.graphics) && (queueFamilyIndices.transfer != queueFamilyIndices.compute))
			{
				// If compute family index differs, we need an additional queue create info for the compute queue
				VkDeviceQueueCreateInfo queueInfo{};
				queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueInfo.queueFamilyIndex = queueFamilyIndices.transfer;
				queueInfo.queueCount = 1;
				queueInfo.pQueuePriorities = &defaultQueuePriority;
				queueCreateInfos.push_back(queueInfo);
			}
		}
		else
		{
			// Else we use the same queue
			queueFamilyIndices.transfer = queueFamilyIndices.graphics;
		}

		// Create the logical device representation
		std::vector<const char*> deviceExtensions(enabledExtensions);
		if (useSwapChain)
		{
			deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
		}

		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
		deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
		deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

		VkPhysicalDeviceShaderAtomicFloatFeaturesEXT physicalDeviceShaderAtomicFloatFeaturesEXT{};
		physicalDeviceShaderAtomicFloatFeaturesEXT.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
		physicalDeviceShaderAtomicFloatFeaturesEXT.shaderBufferFloat32AtomicAdd = true;
		deviceCreateInfo.pNext = &physicalDeviceShaderAtomicFloatFeaturesEXT;

		// If a pNext(Chain) has been passed, we need to add it to the device creation info
		VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
		if (pNextChain) {
			physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			physicalDeviceFeatures2.features = enabledFeatures;
			physicalDeviceFeatures2.pNext = pNextChain;
			deviceCreateInfo.pEnabledFeatures = nullptr;
			deviceCreateInfo.pNext = &physicalDeviceFeatures2;
		}

		// Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
		if (extensionSupported(VK_EXT_DEBUG_MARKER_EXTENSION_NAME))
		{
			deviceExtensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);
			enableDebugMarkers = true;
		}

		if (extensionSupported(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
			deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
		}

		if (extensionSupported(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
			deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
		}

		if (deviceExtensions.size() > 0)
		{
			for (const char* enabledExtension : deviceExtensions)
			{
				if (!extensionSupported(enabledExtension)) {
					std::cerr << "Enabled device extension \"" << enabledExtension << "\" is not present at device level\n";
				}
			}

			deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
			deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		}

		this->enabledFeatures = enabledFeatures;

		VkResult result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice);
		if (result != VK_SUCCESS)
		{
			return result;
		}

		// Create a default command pool for graphics command buffers
		commandPool = createCommandPool(queueFamilyIndices.graphics);

		// Get a graphics queue from the device
		vkGetDeviceQueue(logicalDevice, this->queueFamilyIndices.graphics, 0, &graphicsQueue);

		vkGetDeviceQueue(logicalDevice, this->queueFamilyIndices.compute, 0, &computeQueue);

		vkGetDeviceQueue(logicalDevice, this->queueFamilyIndices.transfer, 0, &transferQueue);

		createPipelineCache();

		return result;
	}


	void VkContext::createPipelineCache()
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(logicalDevice, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
	}

	VkResult VkContext::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory, void *data)
	{
		// Create the buffer handle
		VkBufferCreateInfo bufferCreateInfo = csyvk::initializers::bufferCreateInfo(usageFlags, size);
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(logicalDevice, &bufferCreateInfo, nullptr, buffer));

		VkMemoryRequirements memReqs;
		VkMemoryAllocateInfo memAlloc = csyvk::initializers::memoryAllocateInfo();
		vkGetBufferMemoryRequirements(logicalDevice, *buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
		VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
		if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			memAlloc.pNext = &allocFlagsInfo;
		}
		VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &memAlloc, nullptr, memory));

		if (data != nullptr)
		{
			void *mapped;
			VK_CHECK_RESULT(vkMapMemory(logicalDevice, *memory, 0, size, 0, &mapped));
			memcpy(mapped, data, size);
			if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
			{
				VkMappedMemoryRange mappedRange = csyvk::initializers::mappedMemoryRange();
				mappedRange.memory = *memory;
				mappedRange.offset = 0;
				mappedRange.size = size;
				vkFlushMappedMemoryRanges(logicalDevice, 1, &mappedRange);
			}
			vkUnmapMemory(logicalDevice, *memory);
		}

		// Attach the memory to the buffer object
		VK_CHECK_RESULT(vkBindBufferMemory(logicalDevice, *buffer, *memory, 0));

		return VK_SUCCESS;
	}

	VkResult VkContext::createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, std::shared_ptr<csyvk::Buffer> &buffer, VkDeviceSize size, const void *data)
	{
		buffer->device = logicalDevice;

		// Create the buffer handle
		VkBufferCreateInfo bufferCreateInfo = csyvk::initializers::bufferCreateInfo(usageFlags, size);
		VK_CHECK_RESULT(vkCreateBuffer(logicalDevice, &bufferCreateInfo, nullptr, &buffer->buffer));

		// Create the memory backing up the buffer handle
		VkMemoryRequirements memReqs;
		VkMemoryAllocateInfo memAlloc = csyvk::initializers::memoryAllocateInfo();
		vkGetBufferMemoryRequirements(logicalDevice, buffer->buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Find a memory type index that fits the properties of the buffer
		memAlloc.memoryTypeIndex = getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags);
		// If the buffer has VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set we also need to enable the appropriate flag during allocation
		VkMemoryAllocateFlagsInfoKHR allocFlagsInfo{};
		if (usageFlags & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO_KHR;
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
			memAlloc.pNext = &allocFlagsInfo;
		}
		VK_CHECK_RESULT(vkAllocateMemory(logicalDevice, &memAlloc, nullptr, &buffer->memory));

		buffer->usePool = VK_FALSE;
		buffer->alignment = memReqs.alignment;
		buffer->size = size;
		buffer->usageFlags = usageFlags;
		buffer->memoryPropertyFlags = memoryPropertyFlags;

		// If a pointer to the buffer data has been passed, map the buffer and copy over the data
		if (data != nullptr)
		{
			VK_CHECK_RESULT(buffer->map());
			memcpy(buffer->mapped, data, size);
			if ((memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)
				buffer->flush();

			buffer->unmap();
		}

		// Initialize a default descriptor that covers the whole buffer size
		buffer->setupDescriptor();

		// Attach the memory to the buffer object
		return buffer->bind();
	}

    VkResult VkContext::createBuffer(uint32_t poolType, std::shared_ptr<csyvk::Buffer> &buffer, const void *data)
    {
        VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferCreateInfo.size = buffer->size;
        bufferCreateInfo.usage = buffer->usageFlags;

        VmaAllocationCreateInfo allocationCreateInfo = {};
        allocationCreateInfo.usage = static_cast<VmaMemoryUsage>(poolMap[poolType].usage);
        allocationCreateInfo.pool = poolMap[poolType].pool;

        VmaAllocationInfo allocationInfo;
        VkResult ret = vmaCreateBuffer(g_Allocator, &bufferCreateInfo, &allocationCreateInfo, &buffer->buffer, &buffer->allocation, &allocationInfo);
        VK_CHECK_RESULT(ret);

        buffer->usePool = VK_TRUE;
        buffer->memory = allocationInfo.deviceMemory;
        buffer->offset = allocationInfo.offset;
        buffer->allocator = g_Allocator;
        buffer->device = deviceHandle();

        if (data != nullptr) {
            ret = vmaMapMemory(buffer->allocator, buffer->allocation, (void**)&buffer->mapped);
            VK_CHECK_RESULT(ret);
            memcpy(buffer->mapped, data, buffer->size);

            if ((buffer->memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
                vmaFlushAllocation(buffer->allocator, buffer->allocation, allocationInfo.offset, allocationInfo.size);
            }
            vmaUnmapMemory(buffer->allocator, buffer->allocation);
            buffer->mapped = nullptr;
        }
        buffer->setupDescriptor();
        return VK_SUCCESS;
    }

    VkResult VkContext::createMemoryPool(VkInstance instance, uint32_t apiVerion)
    {
        useMemoryPool = false;
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.vulkanApiVersion = apiVerion;
        allocatorInfo.physicalDevice = physicalDeviceHandle();
        allocatorInfo.device = deviceHandle();
        allocatorInfo.instance = instance;
        vmaCreateAllocator(&allocatorInfo, &g_Allocator);

        VmaPoolCreateInfo poolCreateInfo = {};
        poolCreateInfo.blockSize = 20 * 1024 * 1024;    // 20M
        poolCreateInfo.minBlockCount = 1;
        poolCreateInfo.maxBlockCount = 2;

        VmaAllocationCreateInfo allocationCreateInfo = {};
        VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferCreateInfo.size = 1024;

        MemoryPoolInfo memoryPoolInfo = {};
        for (uint32_t type = DevicePool; type < EndType; ++type) {
            switch (type) {
                case DevicePool: {
                    // Device memory pool
                    allocationCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
                    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
                    break;
                }
                case HostPool: {
                    // Host memory pool
                    allocationCreateInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                    break;
                }
                case UniformPool: {
                    // Uniform memory pool
                    allocationCreateInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
                    allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                    break;
                }
                default:
                    break;
            }

            memoryPoolInfo.usage = allocationCreateInfo.usage;
            vmaFindMemoryTypeIndexForBufferInfo(g_Allocator, &bufferCreateInfo, &allocationCreateInfo, &poolCreateInfo.memoryTypeIndex);
            VkResult res = vmaCreatePool(g_Allocator, &poolCreateInfo, &memoryPoolInfo.pool);
            if (res != VK_SUCCESS) {
                csyvk::tools::exitFatal("Could not create memory pool : \n" + csyvk::tools::errorString(res), res);
                return res;
            }
            poolMap[type] = memoryPoolInfo;
        }

        return VK_SUCCESS;
    }

	void VkContext::copyBuffer(csyvk::Buffer *src, csyvk::Buffer *dst, VkQueue queue, VkBufferCopy *copyRegion)
	{
		assert(dst->size <= src->size);
		assert(src->buffer);
		VkCommandBuffer copyCmd = createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy bufferCopy{};
		if (copyRegion == nullptr)
		{
			bufferCopy.size = src->size;
		}
		else
		{
			bufferCopy = *copyRegion;
		}

		vkCmdCopyBuffer(copyCmd, src->buffer, dst->buffer, 1, &bufferCopy);

		flushCommandBuffer(copyCmd, queue);
	}

	VkCommandPool VkContext::createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags createFlags)
	{
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = createFlags;
		VkCommandPool cmdPool;
		VK_CHECK_RESULT(vkCreateCommandPool(logicalDevice, &cmdPoolInfo, nullptr, &cmdPool));
		return cmdPool;
	}

	VkCommandBuffer VkContext::createCommandBuffer(VkCommandBufferLevel level, VkCommandPool pool, bool begin)
	{
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = csyvk::initializers::commandBufferAllocateInfo(pool, level, 1);
		VkCommandBuffer cmdBuffer;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(logicalDevice, &cmdBufAllocateInfo, &cmdBuffer));
		// If requested, also start recording for the new command buffer
		if (begin)
		{
			VkCommandBufferBeginInfo cmdBufInfo = csyvk::initializers::commandBufferBeginInfo();
			VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
		}
		return cmdBuffer;
	}

	VkCommandBuffer VkContext::createCommandBuffer(VkCommandBufferLevel level, bool begin)
	{
		return createCommandBuffer(level, commandPool, begin);
	}

	void VkContext::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, VkCommandPool pool, bool free)
	{
		if (commandBuffer == VK_NULL_HANDLE)
		{
			return;
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

		VkSubmitInfo submitInfo = csyvk::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		// Create fence to ensure that the command buffer has finished executing
		VkFenceCreateInfo fenceInfo = csyvk::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence));
		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		// Wait for the fence to signal that command buffer has finished executing
		VK_CHECK_RESULT(vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
		vkDestroyFence(logicalDevice, fence, nullptr);
		if (free)
		{
			vkFreeCommandBuffers(logicalDevice, pool, 1, &commandBuffer);
		}
	}

	void VkContext::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free)
	{
		return flushCommandBuffer(commandBuffer, queue, commandPool, free);
	}

	bool VkContext::extensionSupported(std::string extension)
	{
		return (std::find(supportedExtensions.begin(), supportedExtensions.end(), extension) != supportedExtensions.end());
	}

	VkFormat VkContext::getSupportedDepthFormat(bool checkSamplingSupport)
	{
		// All depth formats may be optional, so we need to find a suitable depth format to use
		std::vector<VkFormat> depthFormats = { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D16_UNORM };
		for (auto& format : depthFormats)
		{
			VkFormatProperties formatProperties;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);
			// Format must support depth stencil attachment for optimal tiling
			if (formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
			{
				if (checkSamplingSupport) {
					if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
						continue;
					}
				}
				return format;
			}
		}
		throw std::runtime_error("Could not find a matching depth format");
	}
}