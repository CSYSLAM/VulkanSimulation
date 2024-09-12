///*
//* Vulkan Example - Minimal headless compute example
//*
//* Copyright (C) 2017-2022 by Sascha Willems - www.saschawillems.de
//*
//* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
//*/
//
//
#pragma comment(linker, "/subsystem:console")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>

#include <vulkan/vulkan.h>
#include "VulkanTools.h"
#include "CommandLineParser.hpp"

#include "vkcsy.h"
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <thread>


#define MU_SHADER_PATH "D:/cg/vulkan/csy_codehub/VulkanSimulation/shaders/glsl/computesph/"
#define NUM_PARTICLES 20000
#define WORK_GROUP_SIZE 128
#define PARTICLE_RADIUS 0.005f
// work group count is the ceiling of particle count divided by work group size
#define NUM_WORK_GROUPS ((NUM_PARTICLES + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE)
#define DEBUG (!NDEBUG)
#define BUFFER_ELEMENTS 32
#define LOG(...) printf(__VA_ARGS__)


static VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageCallback(
	VkDebugReportFlagsEXT flags,
	VkDebugReportObjectTypeEXT objectType,
	uint64_t object,
	size_t location,
	int32_t messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData)
{
	LOG("[VALIDATION]: %s - %s\n", pLayerPrefix, pMessage);
	return VK_FALSE;
}

CommandLineParser commandLineParser;

class VulkanExample
{
public:
	GLFWwindow* window_ = NULL;
	uint32_t windowHeight_ = 1000;
	uint32_t windowWidth_ = 1000;
	bool paused_ = false;
	std::atomic_uint64_t frameNumber_ = 1;

	VkInstance instance_;
	VkSurfaceKHR surface_;
	VkPhysicalDevice physicalDevice_;
	VkPhysicalDeviceFeatures physicalDeviceFeatures_;
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties_;
	VkDevice device_;
	uint32_t queueFamilyIndex_ = UINT32_MAX;
	VkQueue presentationQueue_;
	VkQueue graphicsQueue_;
	VkQueue computeQueue_;
	VkSurfaceFormatKHR surfaceFormat_;
	VkSwapchainKHR swapchain_;
	std::vector<VkImage> swapchainImage_;
	std::vector<VkImageView> swapchainImageView_;
	VkRenderPass renderPass_;
	std::vector<VkFramebuffer> swapchainFrameBuffer_;
	VkPipelineCache globalPipelineCache_;
	VkDescriptorPool globalDescriptorPool_;
	VkBuffer packedParticlesBuffer_;
	VkDeviceMemory packedParticlesMemory_;
	VkPipelineLayout graphicsPipelineLayout_;
	VkPipeline graphicsPipeline_;
	VkCommandPool graphicsCommandPool_;
	std::vector<VkCommandBuffer> graphicsCommandBuffer_;
	// synchronization
	VkSemaphore imageAvailableSemaphore_;
	VkSemaphore renderFinishedSemaphore_;
	VkDescriptorSetLayout computeDescriptorSetLayout_;
	VkDescriptorSet computeDescriptorSet_;
	VkPipelineLayout computePipelineLayout_;
	VkPipeline computePipeline_[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
	VkCommandPool computeCommandPool_;
	VkCommandBuffer computeCommandBuffer_;
	uint32_t imageIndex_;
	VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	VkSubmitInfo computeSubmitInfo_
	{
		VK_STRUCTURE_TYPE_SUBMIT_INFO,
		NULL,
		0,
		NULL,
		0,
		1,
		&computeCommandBuffer_,
		0,
		NULL
	};

	VkSubmitInfo graphicsSubmitInfo_
	{
		VK_STRUCTURE_TYPE_SUBMIT_INFO,
		NULL,
		1,
		&imageAvailableSemaphore_,
		&wait_dst_stage_mask,
		1,
		VK_NULL_HANDLE,
		1,
		&renderFinishedSemaphore_
	};

	VkPresentInfoKHR presentInfo_
	{
		VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		NULL,
		1,
		&renderFinishedSemaphore_,
		1,
		&swapchain_,
		&imageIndex_,
		NULL
	};

	VkPipelineCache pipelineCache;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence fence;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule shaderModule;

	// ssbo sizes
	const uint64_t positionSsboSize_ = sizeof(glm::vec2) * NUM_PARTICLES;
	const uint64_t velocitySsboSize_ = sizeof(glm::vec2) * NUM_PARTICLES;
	const uint64_t forceSsboSize_ = sizeof(glm::vec2) * NUM_PARTICLES;
	const uint64_t densitySsboSize_ = sizeof(float) * NUM_PARTICLES;
	const uint64_t pressureSsboSize_ = sizeof(float) * NUM_PARTICLES;

	const uint64_t packedBufferSize_ = positionSsboSize_ + velocitySsboSize_ + forceSsboSize_ + densitySsboSize_ + pressureSsboSize_;
	// ssbo offsets
	const uint64_t positionSsboOffset_ = 0;
	const uint64_t velocitySsboOffset_ = positionSsboSize_;
	const uint64_t forceSsboOffset_ = velocitySsboOffset_ + velocitySsboSize_;
	const uint64_t densitySsboOffset_ = forceSsboOffset_ + forceSsboSize_;
	const uint64_t pressureSsboOffset_ = densitySsboOffset_ + densitySsboSize_;

	VkDebugReportCallbackEXT debugReportCallback{};

	void InitializeWindow()
	{
		if (!glfwInit())
		{
			throw std::runtime_error("glfw initialization failed");
		}
		if (!glfwVulkanSupported())
		{
			throw std::runtime_error("failed to find the Vulkan loader");
		}
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window_ = glfwCreateWindow(windowHeight_, windowWidth_, "", NULL, NULL);
		if (!window_)
		{
			glfwTerminate();
			throw std::runtime_error("window creation failed");
		}
		glfwMakeContextCurrent(window_);
		glfwSwapInterval(0);

		// pass Application pointer to the callback using GLFW user pointer
		glfwSetWindowUserPointer(window_, reinterpret_cast<void*>(this));

		// set key callback
		auto key_callback = [](GLFWwindow* window, int key, int, int action, int)
		{
			auto app_ptr = reinterpret_cast<VulkanExample*>(glfwGetWindowUserPointer(window));
			if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
			{
				app_ptr->paused_ = !app_ptr->paused_;
			}
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			{
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}
		};

		glfwSetKeyCallback(window_, key_callback);
	}

	//VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer* buffer, VkDeviceMemory* memory, VkDeviceSize size, void* data = nullptr)
	//{
	//	// Create the buffer handle
	//	VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
	//	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	//	VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer));

	//	// Create the memory backing up the buffer handle
	//	VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
	//	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
	//	VkMemoryRequirements memReqs;
	//	VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
	//	vkGetBufferMemoryRequirements(device, *buffer, &memReqs);
	//	memAlloc.allocationSize = memReqs.size;
	//	// Find a memory type index that fits the properties of the buffer
	//	bool memTypeFound = false;
	//	for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
	//		if ((memReqs.memoryTypeBits & 1) == 1) {
	//			if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
	//				memAlloc.memoryTypeIndex = i;
	//				memTypeFound = true;
	//				break;
	//			}
	//		}
	//		memReqs.memoryTypeBits >>= 1;
	//	}
	//	assert(memTypeFound);
	//	VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, memory));

	//	if (data != nullptr) {
	//		void* mapped;
	//		VK_CHECK_RESULT(vkMapMemory(device, *memory, 0, size, 0, &mapped));
	//		memcpy(mapped, data, size);
	//		vkUnmapMemory(device, *memory);
	//	}

	//	VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *memory, 0));

	//	return VK_SUCCESS;
	//}

	void CreateSwapchain()
	{
		VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		VkExtent2D extent = { windowWidth_, windowHeight_ };

		// Query the surface capabilities and select the swapchain's extent (width, height).
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		{
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_, &surfaceCapabilities);
			if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
				extent = surfaceCapabilities.currentExtent;
			}
		}

		// Select a surface format.
		{
			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, NULL);
			std::vector<VkSurfaceFormatKHR> surfaceFormats;
			surfaceFormats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, surfaceFormats.data());

			for (VkSurfaceFormatKHR entry : surfaceFormats) {
				if ((entry.format == VK_FORMAT_B8G8R8A8_SRGB) && (entry.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
					surfaceFormat_ = entry;
					break;
				}
			}
		}

		// For better performance, use "min + 1";
		uint32_t imageCount = surfaceCapabilities.minImageCount + 1;

		VkSwapchainCreateInfoKHR create_info;
		{
			create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			create_info.pNext = NULL;
			create_info.flags = 0;
			create_info.surface = surface_;
			create_info.minImageCount = imageCount;
			create_info.imageFormat = surfaceFormat_.format;
			create_info.imageColorSpace = surfaceFormat_.colorSpace;
			create_info.imageExtent = extent;
			create_info.imageArrayLayers = 1;
			create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
			// If the graphics and presentation queue is different this should not be exclusive.
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			create_info.queueFamilyIndexCount = 0;
			create_info.pQueueFamilyIndices = NULL;
			create_info.preTransform = surfaceCapabilities.currentTransform;
			create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			create_info.presentMode = swapchainPresentMode;
			create_info.clipped = VK_TRUE;
			create_info.oldSwapchain = VK_NULL_HANDLE;
		}

		if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		std::cout << "Successfully created swapchain" << std::endl;
	}

	void GetSwapchainImages()
	{
		uint32_t swapchainImageCount;
		vkGetSwapchainImagesKHR(device_, swapchain_, &swapchainImageCount, NULL);
		swapchainImage_.resize(swapchainImageCount);
		vkGetSwapchainImagesKHR(device_, swapchain_, &swapchainImageCount, swapchainImage_.data());
		std::cout << "Successfully get swapchain image" << std::endl;
	}

	void CreateSwapchainImageViews()
	{
		swapchainImageView_.resize(swapchainImage_.size());
		for (uint32_t i = 0; i < swapchainImageView_.size(); i++)
		{
			VkImageViewCreateInfo imageViewCreateInfo
			{
				VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				NULL,
				0,
				swapchainImage_[i],
				VK_IMAGE_VIEW_TYPE_2D,
				surfaceFormat_.format,
				{
					VK_COMPONENT_SWIZZLE_IDENTITY, // r
					VK_COMPONENT_SWIZZLE_IDENTITY, // g
					VK_COMPONENT_SWIZZLE_IDENTITY, // b
					VK_COMPONENT_SWIZZLE_IDENTITY // a
				},
				{
					VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
					0, // baseMipLevel
					1, // levelCount
					0, // baseArrayLayer
					1, // layerCount
				}
			};
			VK_CHECK_RESULT(vkCreateImageView(device_, &imageViewCreateInfo, NULL, &swapchainImageView_[i]));
		}
		std::cout << "Successfully create swapchain ImageViews" << std::endl;
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription attachmentDescription
		{
			0,
			surfaceFormat_.format,
			VK_SAMPLE_COUNT_1_BIT,
			VK_ATTACHMENT_LOAD_OP_CLEAR,
			VK_ATTACHMENT_STORE_OP_STORE,
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			VK_ATTACHMENT_STORE_OP_DONT_CARE,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		};

		VkAttachmentReference colorAttachmentReference
		{
			0,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		VkSubpassDescription subpassDescription
		{
			0,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			0,
			NULL,
			1,
			&colorAttachmentReference,
			NULL,
			NULL,
			0,
			NULL
		};

		VkRenderPassCreateInfo renderPassCreateInfo
		{
			VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			NULL,
			0,
			1,
			&attachmentDescription,
			1,
			&subpassDescription,
			0,
			NULL
		};
		VK_CHECK_RESULT(vkCreateRenderPass(device_, &renderPassCreateInfo, NULL, &renderPass_));
		std::cout << "Successfully create render pass" << std::endl;
	}

	void CreateSwapchainFrameBuffers()
	{
		swapchainFrameBuffer_.resize(swapchainImageView_.size());
		for (size_t index = 0; index < swapchainImageView_.size(); index++)
		{
			VkFramebufferCreateInfo framebufferCreateInfo
			{
				VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				NULL,
				0,
				renderPass_,
				1,
				&swapchainImageView_[index],
				windowWidth_,
				windowHeight_,
				1
			};
			VK_CHECK_RESULT(vkCreateFramebuffer(device_, &framebufferCreateInfo, NULL, &swapchainFrameBuffer_[index]));
		}
		std::cout << "Successfully create swapchain framebuffers" << std::endl;
	}

	void CreatePipelineCache()
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
			NULL,
			0,
			0,
			NULL
		};
		VK_CHECK_RESULT(vkCreatePipelineCache(device_, &pipelineCacheCreateInfo, NULL, &globalPipelineCache_));
		std::cout << "Successfully create pipelineCache" << std::endl;
	}

	void CreateDescriptorPool()
	{
		VkDescriptorPoolSize descriptorPoolSize
		{
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			5
		};

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo
		{
			VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			NULL,
			0,
			1,
			1,
			&descriptorPoolSize
		};
		VK_CHECK_RESULT(vkCreateDescriptorPool(device_, &descriptorPoolCreateInfo, NULL, &globalDescriptorPool_));
		std::cout << "Successfully create descriptor pool" << std::endl;
	}

	uint32_t findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties = CsySmallVk::Query::physicalDeviceMemoryProperties(physicalDevice_);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
		{
			if (requirements.memoryTypeBits & (1 << i) &&
				(memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				std::cout << "pick memory type [" << i << "]\n";
				return i;
			}
		}
	}

	void CreateBuffers()
	{
		VkBufferCreateInfo particlesBufferCreateInfo = CsySmallVk::bufferCreateInfo();
		particlesBufferCreateInfo.size = packedBufferSize_;
		particlesBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		particlesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		particlesBufferCreateInfo.queueFamilyIndexCount = 0;
		particlesBufferCreateInfo.pQueueFamilyIndices = nullptr;
		vkCreateBuffer(device_, &particlesBufferCreateInfo, NULL, &packedParticlesBuffer_);

		VkMemoryRequirements positionBufferMemoryRequirements = CsySmallVk::Query::memoryRequirements(device_, packedParticlesBuffer_);
		VkMemoryAllocateInfo particleBufferMemoryAllocationInfo = CsySmallVk::memoryAllocateInfo();
		particleBufferMemoryAllocationInfo.allocationSize = positionBufferMemoryRequirements.size;
		particleBufferMemoryAllocationInfo.memoryTypeIndex = findMemoryType(positionBufferMemoryRequirements,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device_, &particleBufferMemoryAllocationInfo, NULL, &packedParticlesMemory_));
		// bind the memory to the buffer object
		vkBindBufferMemory(device_, packedParticlesBuffer_, packedParticlesMemory_, 0);
		std::cout << "Successfully create buffers" << std::endl;
	}

	void CreateGraphicsPipelineLayout()
	{
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			NULL,
			0,
			0,
			NULL,
			0,
			NULL
		};
		VK_CHECK_RESULT(vkCreatePipelineLayout(device_, &pipelineLayoutCreateInfo, NULL, &graphicsPipelineLayout_));
		std::cout << "Successfully create graphics pipeline layout" << std::endl;
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModule shaderModule;
		VkShaderModuleCreateInfo createInfo = CsySmallVk::shaderModuleCreateInfo();
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VK_CHECK_RESULT(vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule));
		return shaderModule;
	}

	void CreateGraphicsPipeline()
	{
		std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;
		// create shader stage infos
		auto vertexShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "particle.vert.spv");
		VkShaderModule vertexShaderModule = CreateShaderModule(vertexShaderCode);
		auto fragmentShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "particle.frag.spv");
		VkShaderModule fragmentShaderModule = CreateShaderModule(fragmentShaderCode);

		VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		vertexShaderStageCreateInfo.module = vertexShaderModule;
		vertexShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertexShaderStageCreateInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		fragmentShaderStageCreateInfo.module = fragmentShaderModule;
		fragmentShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragmentShaderStageCreateInfo.pName = "main";

		shaderStageCreateInfos.push_back(vertexShaderStageCreateInfo);
		shaderStageCreateInfos.push_back(fragmentShaderStageCreateInfo);

		VkVertexInputBindingDescription vertexInputBindingDescription
		{
			0,
			sizeof(glm::vec2),
			VK_VERTEX_INPUT_RATE_VERTEX
		};

		// layout(location = 0) in vec2 position;
		VkVertexInputAttributeDescription vertexInputAttributeDescription
		{
			0,
			0,
			VK_FORMAT_R32G32_SFLOAT,
			0
		};

		VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			NULL,
			0,
			1,
			&vertexInputBindingDescription,
			1,
			&vertexInputAttributeDescription
		};

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			NULL,
			0,
			VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
			VK_FALSE
		};

		VkViewport viewport
		{
			0,
			0,
			static_cast<float>(windowWidth_),
			static_cast<float>(windowHeight_),
			0,
			1
		};

		VkRect2D scissor
		{
			{ 0, 0 },
			{ windowWidth_, windowHeight_ }
		};

		VkPipelineViewportStateCreateInfo viewportStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			NULL,
			0,
			1,
			&viewport,
			1,
			&scissor
		};

		VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			NULL,
			0,
			VK_FALSE,
			VK_FALSE,
			VK_POLYGON_MODE_FILL,
			VK_CULL_MODE_NONE,
			VK_FRONT_FACE_COUNTER_CLOCKWISE,
			VK_FALSE,
			0,
			0,
			0,
			1
		};

		VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			NULL,
			0,
			VK_SAMPLE_COUNT_1_BIT,
			VK_FALSE,
			0,
			NULL,
			VK_FALSE,
			VK_FALSE
		};

		VkPipelineColorBlendAttachmentState colorBlendAttachment
		{
			VK_FALSE,
			VK_BLEND_FACTOR_ONE,
			VK_BLEND_FACTOR_ZERO,
			VK_BLEND_OP_ADD,
			VK_BLEND_FACTOR_ONE,
			VK_BLEND_FACTOR_ZERO,
			VK_BLEND_OP_ADD,
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
		};

		VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			NULL,
			0,
			VK_FALSE,
			VK_LOGIC_OP_COPY,
			1,
			&colorBlendAttachment,
			{0, 0, 0, 0}
		};

		VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo
		{
			VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			NULL,
			0,
			static_cast<uint32_t>(shaderStageCreateInfos.size()),
			shaderStageCreateInfos.data(),
			&vertexInputStateCreateInfo,
			&inputAssemblyStateCreateInfo,
			NULL,
			&viewportStateCreateInfo,
			&rasterizationStateCreateInfo,
			&multisampleStateCreateInfo,
			NULL,
			&colorBlendStateCreateInfo,
			NULL,
			graphicsPipelineLayout_,
			renderPass_,
			0,
			VK_NULL_HANDLE,
			-1
		};
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device_, globalPipelineCache_, 1, &graphicsPipelineCreateInfo, NULL, &graphicsPipeline_));
		std::cout << "Successfully create graphics pipeline" << std::endl;
	}

	void CreateGraphicsCommandPool()
	{
		VkCommandPoolCreateInfo graphicsCommandPoolCreateInfo = CsySmallVk::commandPoolCreateInfo();
		graphicsCommandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex_;
		VK_CHECK_RESULT(vkCreateCommandPool(device_, &graphicsCommandPoolCreateInfo, NULL, &graphicsCommandPool_));
		std::cout << "Successfully create graphics command pool" << std::endl;
	}

	void CreateGraphicsCommandBuffers()
	{
		graphicsCommandBuffer_.resize(swapchainFrameBuffer_.size());
		VkCommandBufferAllocateInfo graphicsCommandBufferAllocationInfo
		{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			NULL,
			graphicsCommandPool_,
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			static_cast<uint32_t>(graphicsCommandBuffer_.size())
		};
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device_, &graphicsCommandBufferAllocationInfo, graphicsCommandBuffer_.data()));
		for (size_t i = 0; i < graphicsCommandBuffer_.size(); i++)
		{
			VkCommandBufferBeginInfo commandBufferBeginInfo
			{
				VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				NULL,
				VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				NULL
			};
			vkBeginCommandBuffer(graphicsCommandBuffer_[i], &commandBufferBeginInfo);
			VkClearValue clear_value{ 0.92f, 0.92f, 0.92f, 1.0f };
			VkRenderPassBeginInfo renderPassBeginInfo
			{
				VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				NULL,
				renderPass_,
				swapchainFrameBuffer_[i],
				{
					{ 0, 0 },
					{ windowWidth_, windowHeight_ }
				},
				1,
				&clear_value
			};
			vkCmdBeginRenderPass(graphicsCommandBuffer_[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			VkViewport viewport
			{
				0,
				0,
				static_cast<float>(windowWidth_),
				static_cast<float>(windowHeight_),
				0,
				1
			};

			VkRect2D scissor
			{
				{ 0, 0 },
				{ windowWidth_, windowHeight_ }
			};

			vkCmdSetViewport(graphicsCommandBuffer_[i], 0, 1, &viewport);
			vkCmdSetScissor(graphicsCommandBuffer_[i], 0, 1, &scissor);
			vkCmdBindPipeline(graphicsCommandBuffer_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

			VkDeviceSize offsets = 0;
			vkCmdBindVertexBuffers(graphicsCommandBuffer_[i], 0, 1, &packedParticlesBuffer_, &offsets);
			vkCmdDraw(graphicsCommandBuffer_[i], NUM_PARTICLES, 1, 0, 0);
			vkCmdEndRenderPass(graphicsCommandBuffer_[i]);

			if (vkEndCommandBuffer(graphicsCommandBuffer_[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("command buffer creation failed");
			}
		}
		std::cout << "Successfully create graphics command buffers" << std::endl;
	}

	void CreateSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo
		{
			VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			NULL,
			0
		};
		VK_CHECK_RESULT(vkCreateSemaphore(device_, &semaphoreCreateInfo, NULL, &imageAvailableSemaphore_));
		VK_CHECK_RESULT(vkCreateSemaphore(device_, &semaphoreCreateInfo, NULL, &renderFinishedSemaphore_));
		std::cout << "Successfully create semaphores" << std::endl;
	}

	void CreateComputeDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5];
		descriptorSetLayoutBindings[0].binding = 0;
		descriptorSetLayoutBindings[0].descriptorCount = 1;
		descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetLayoutBindings[0].pImmutableSamplers = nullptr;
		descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		descriptorSetLayoutBindings[1].binding = 1;
		descriptorSetLayoutBindings[1].descriptorCount = 1;
		descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetLayoutBindings[1].pImmutableSamplers = nullptr;
		descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		descriptorSetLayoutBindings[2].binding = 2;
		descriptorSetLayoutBindings[2].descriptorCount = 1;
		descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetLayoutBindings[2].pImmutableSamplers = nullptr;
		descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		descriptorSetLayoutBindings[3].binding = 3;
		descriptorSetLayoutBindings[3].descriptorCount = 1;
		descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetLayoutBindings[3].pImmutableSamplers = nullptr;
		descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		descriptorSetLayoutBindings[4].binding = 4;
		descriptorSetLayoutBindings[4].descriptorCount = 1;
		descriptorSetLayoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorSetLayoutBindings[4].pImmutableSamplers = nullptr;
		descriptorSetLayoutBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = CsySmallVk::descriptorSetLayoutCreateInfo();
		descriptorSetLayoutCreateInfo.bindingCount = 5;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
		if (vkCreateDescriptorSetLayout(device_, &descriptorSetLayoutCreateInfo, NULL, &computeDescriptorSetLayout_) != VK_SUCCESS)
		{
			throw std::runtime_error("compute descriptor layout creation failed");
		}
		std::cout << "Successfully create compute descriptorSet layout" << std::endl;
	}

	void UpdateComputeDescriptorSets()
	{
		// allocate descriptor sets
		VkDescriptorSetAllocateInfo allocInfo = CsySmallVk::descriptorSetAllocateInfo();
		allocInfo.descriptorPool = globalDescriptorPool_;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &computeDescriptorSetLayout_;
		if (vkAllocateDescriptorSets(device_, &allocInfo, &computeDescriptorSet_) != VK_SUCCESS)
		{
			throw std::runtime_error("compute descriptor set allocation failed");
		}

		VkDescriptorBufferInfo descriptorBufferInfos[5];
		descriptorBufferInfos[0].buffer = packedParticlesBuffer_;
		descriptorBufferInfos[0].offset = positionSsboOffset_;
		descriptorBufferInfos[0].range = positionSsboSize_;
		descriptorBufferInfos[1].buffer = packedParticlesBuffer_;
		descriptorBufferInfos[1].offset = velocitySsboOffset_;
		descriptorBufferInfos[1].range = velocitySsboSize_;
		descriptorBufferInfos[2].buffer = packedParticlesBuffer_;
		descriptorBufferInfos[2].offset = forceSsboOffset_;
		descriptorBufferInfos[2].range = forceSsboSize_;
		descriptorBufferInfos[3].buffer = packedParticlesBuffer_;
		descriptorBufferInfos[3].offset = densitySsboOffset_;
		descriptorBufferInfos[3].range = densitySsboSize_;
		descriptorBufferInfos[4].buffer = packedParticlesBuffer_;
		descriptorBufferInfos[4].offset = pressureSsboOffset_;
		descriptorBufferInfos[4].range = pressureSsboSize_;

		// write descriptor sets
		VkWriteDescriptorSet writeDescriptorSets[5];
		for (int index = 0; index < 5; index++)
		{
			VkWriteDescriptorSet write = CsySmallVk::writeDescriptorSet();
			write.descriptorCount = 1;
			write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			write.dstBinding = index;
			write.dstArrayElement = 0;
			write.dstSet = computeDescriptorSet_;
			write.pBufferInfo = &descriptorBufferInfos[index];
			writeDescriptorSets[index] = write;
		}

		vkUpdateDescriptorSets(device_, 5, writeDescriptorSets, 0, NULL);
		std::cout << "Successfully update compute descriptorsets" << std::endl;
	}

	void CreateComputePipelineLayout()
	{
		VkPipelineLayoutCreateInfo layoutCreateInfo = CsySmallVk::pipelineLayoutCreateInfo();
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &computeDescriptorSetLayout_;
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(device_, &layoutCreateInfo, nullptr, &computePipelineLayout_) != VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");
		std::cout << "Successfully create compute pipeline layout" << std::endl;
	}

	void CreateComputePipelines()
	{
		// first
		auto computeDensityPressureShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "compute_density_pressure.comp.spv");
		VkShaderModule computeDensityPressureShaderModule = CreateShaderModule(computeDensityPressureShaderCode);
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		shaderStageCreateInfo.module = computeDensityPressureShaderModule;
		shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageCreateInfo.pName = "main";

		VkComputePipelineCreateInfo createInfo = CsySmallVk::computePipelineCreateInfo();
		createInfo.basePipelineHandle = VK_NULL_HANDLE;
		createInfo.basePipelineIndex = 0;
		createInfo.stage = shaderStageCreateInfo;
		createInfo.layout = computePipelineLayout_;
		if (vkCreateComputePipelines(device_, globalPipelineCache_, 1, &createInfo, NULL, &computePipeline_[0]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}

		//second
		auto computeForceShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "compute_force.comp.spv");
		VkShaderModule computeForceShaderModule = CreateShaderModule(computeForceShaderCode);
		shaderStageCreateInfo.module = computeForceShaderModule;
		createInfo.stage = shaderStageCreateInfo;
		if (vkCreateComputePipelines(device_, globalPipelineCache_, 1, &createInfo, NULL, &computePipeline_[1]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}

		//third
		auto integrateShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "integrate.comp.spv");
		VkShaderModule integrateShaderModule = CreateShaderModule(integrateShaderCode);
		shaderStageCreateInfo.module = integrateShaderModule;
		createInfo.stage = shaderStageCreateInfo;
		if (vkCreateComputePipelines(device_, globalPipelineCache_, 1, &createInfo, NULL, &computePipeline_[2]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}
		std::cout << "Successfully create compute pipelines" << std::endl;
	}

	void CreateComputeCommandPool()
	{
		VkCommandPoolCreateInfo createInfo = CsySmallVk::commandPoolCreateInfo();
		createInfo.queueFamilyIndex = queueFamilyIndex_;
		createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(device_, &createInfo, NULL, &computeCommandPool_) != VK_SUCCESS)
		{
			throw std::runtime_error("command pool creation failed");
		}

		std::cout << "Successfully create compute command pool" << std::endl;
	}

	void CreateComputeCommandBuffer()
	{
		VkCommandBufferAllocateInfo allocInfo = CsySmallVk::commandBufferAllocateInfo();
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = computeCommandPool_;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		if (vkAllocateCommandBuffers(device_, &allocInfo, &computeCommandBuffer_) != VK_SUCCESS)
		{
			throw std::runtime_error("buffer allocation failed");
		}
		VkCommandBufferBeginInfo beginInfo = CsySmallVk::commandBufferBeginInfo();
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		if (vkBeginCommandBuffer(computeCommandBuffer_, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer begin failed");
		}
		vkCmdBindDescriptorSets(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout_, 0, 1, &computeDescriptorSet_, 0, NULL);
		// First dispatch
		vkCmdBindPipeline(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_[0]);
		vkCmdDispatch(computeCommandBuffer_, NUM_WORK_GROUPS, 1, 1);

		// Barrier: compute to compute dependencies
		// First dispatch writes to a storage buffer, second dispatch reads from that storage buffer
		vkCmdPipelineBarrier(computeCommandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);

		// Second dispatch
		vkCmdBindPipeline(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_[1]);
		vkCmdDispatch(computeCommandBuffer_, NUM_WORK_GROUPS, 1, 1);

		// Barrier: compute to compute dependencies
		// Second dispatch writes to a storage buffer, third dispatch reads from that storage buffer
		vkCmdPipelineBarrier(computeCommandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);

		// Third dispatch
		// Third dispatch writes to the storage buffer. Later, vkCmdDraw reads that buffer as a vertex buffer with vkCmdBindVertexBuffers.
		vkCmdBindPipeline(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_[2]);
		vkCmdDispatch(computeCommandBuffer_, NUM_WORK_GROUPS, 1, 1);

		vkCmdPipelineBarrier(computeCommandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
		vkEndCommandBuffer(computeCommandBuffer_);
		std::cout << "Successfully create compute command buffer" << std::endl;
	}

	void SetInitialParticleData()
	{
		// staging buffer
		VkBuffer stagingBufferHandle = VK_NULL_HANDLE;
		VkDeviceMemory stagingBufferMemoryDeviceHandle = VK_NULL_HANDLE;
		VkBufferCreateInfo stagingBufferCreateInfo = CsySmallVk::bufferCreateInfo();
		stagingBufferCreateInfo.size = packedBufferSize_;
		stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		stagingBufferCreateInfo.queueFamilyIndexCount = 0;
		stagingBufferCreateInfo.pQueueFamilyIndices = nullptr;

		vkCreateBuffer(device_, &stagingBufferCreateInfo, NULL, &stagingBufferHandle);

		VkMemoryRequirements stagingBufferMemoryRequirements;
		vkGetBufferMemoryRequirements(device_, stagingBufferHandle, &stagingBufferMemoryRequirements);

		VkMemoryAllocateInfo allocInfo = CsySmallVk::memoryAllocateInfo();
		allocInfo.allocationSize = stagingBufferMemoryRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(stagingBufferMemoryRequirements,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (vkAllocateMemory(device_, &allocInfo, NULL, &stagingBufferMemoryDeviceHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("memory allocation failed");
		}

		// bind the memory to the buffer object
		vkBindBufferMemory(device_, stagingBufferHandle, stagingBufferMemoryDeviceHandle, 0);

		void* mappedMemory = NULL;
		vkMapMemory(device_, stagingBufferMemoryDeviceHandle, 0, stagingBufferMemoryRequirements.size, 0, &mappedMemory);

		// set the initial particles data
		std::vector<glm::vec2> initialParticlePosition(NUM_PARTICLES);
		for (auto i = 0, x = 0, y = 0; i < NUM_PARTICLES; i++)
		{
			initialParticlePosition[i].x = -0.625f + PARTICLE_RADIUS * 2 * x;
			initialParticlePosition[i].y = -1 + PARTICLE_RADIUS * 2 * y;
			x++;
			if (x >= 125)
			{
				x = 0;
				y++;
			}
		}
		// zero all 
		std::memset(mappedMemory, 0, packedBufferSize_);
		std::memcpy(mappedMemory, initialParticlePosition.data(), positionSsboSize_);
		vkUnmapMemory(device_, stagingBufferMemoryDeviceHandle);

		// submit a command buffer to copy staging buffer to the particle buffer 
		VkCommandBuffer copyCommandBufferHandle;
		VkCommandBufferAllocateInfo copyCommandBufferAllocationInfo = CsySmallVk::commandBufferAllocateInfo();
		copyCommandBufferAllocationInfo.commandBufferCount = 1;
		copyCommandBufferAllocationInfo.commandPool = computeCommandPool_;
		copyCommandBufferAllocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device_, &copyCommandBufferAllocationInfo, &copyCommandBufferHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer creation failed");
		}

		VkCommandBufferBeginInfo commandBufferBeginInfo = CsySmallVk::commandBufferBeginInfo();
		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		if (vkBeginCommandBuffer(copyCommandBufferHandle, &commandBufferBeginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer begin failed");
		}

		VkBufferCopy bufferCopyRegion
		{
			0,
			0,
			stagingBufferMemoryRequirements.size
		};

		vkCmdCopyBuffer(copyCommandBufferHandle, stagingBufferHandle, packedParticlesBuffer_, 1, &bufferCopyRegion);
		if (vkEndCommandBuffer(copyCommandBufferHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer end failed");
		}
		VkSubmitInfo submitInfo = CsySmallVk::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCommandBufferHandle;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.signalSemaphoreCount = 0;
		if (vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer submission failed");
		}
		if (vkQueueWaitIdle(computeQueue_) != VK_SUCCESS)
		{
			throw std::runtime_error("vkQueueWaitIdle failed");
		}
		vkFreeCommandBuffers(device_, computeCommandPool_, 1, &copyCommandBufferHandle);
		vkFreeMemory(device_, stagingBufferMemoryDeviceHandle, NULL);
		vkDestroyBuffer(device_, stagingBufferHandle, NULL);
		std::cout << "Successfully set initial particle data" << std::endl;
	}

	void RunSimulation()
	{
		if (vkQueueSubmit(computeQueue_, 1, &computeSubmitInfo_, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("compute queue submission failed");
		}
	}

	void Render()
	{
		// submit graphics command buffer
		vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailableSemaphore_, VK_NULL_HANDLE, &imageIndex_);
		graphicsSubmitInfo_.pCommandBuffers = graphicsCommandBuffer_.data() + imageIndex_;
		if (vkQueueSubmit(graphicsQueue_, 1, &graphicsSubmitInfo_, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("graphics queue submission failed");
		}
		// queue the image for presentation
		vkQueuePresentKHR(presentationQueue_, &presentInfo_);

		vkQueueWaitIdle(presentationQueue_);
	}

	void MainLoop()
	{
		static std::chrono::high_resolution_clock::time_point frame_start;
		static std::chrono::high_resolution_clock::time_point frame_end;
		static int64_t total_frame_time_ns;

		frame_start = std::chrono::high_resolution_clock::now();

		// process user inputs
		glfwPollEvents();

		// step through the simulation if not paused
		if (!paused_)
		{
			RunSimulation();
			frameNumber_++;
		}

		Render();
		frame_end = std::chrono::high_resolution_clock::now();
		// measure performance
		total_frame_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - frame_start).count();
		std::stringstream title;
		title.precision(3);
		title.setf(std::ios_base::fixed, std::ios_base::floatfield);
		title << "Vulkan| "
			<< NUM_PARTICLES << " particles | "
			"frame #" << frameNumber_ << " | "
			"render latency: " << 1e-6 * total_frame_time_ns << " ms | "
			"FPS: " << 1.0 / (1e-9 * total_frame_time_ns);
		glfwSetWindowTitle(window_, title.str().c_str());
	}

	void Run()
	{
		// to measure performance
		std::thread
		(
			[this]()
			{
				std::this_thread::sleep_for(std::chrono::seconds(20));
				std::cout << "[INFO] frame count after 20 seconds after setup (do not pause or move the window): " << frameNumber_ << std::endl;
			}
		).detach();

		while (!glfwWindowShouldClose(window_))
		{
			MainLoop();
		}
	}

	void destroyWindow()
	{
		glfwDestroyWindow(window_);
		glfwTerminate();
	}

	void destroyVulkan()
	{
		vkDestroySwapchainKHR(device_, swapchain_, NULL);
		vkDestroySurfaceKHR(instance_, surface_, NULL);
		vkDestroyDevice(device_, NULL);
		vkDestroyInstance(instance_, NULL);
	}

	VulkanExample()
	{
		LOG("Running headless compute example\n");
		InitializeWindow();

		// CreateInstance
		VkApplicationInfo vkAppInfo = CsySmallVk::applicationInfo();
		vkAppInfo.pApplicationName = "SPH Simulation Vulkan";
		vkAppInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 1);
		vkAppInfo.pEngineName = "Csy SPH Simulation Engine";
		vkAppInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		vkAppInfo.apiVersion = VK_API_VERSION_1_3;

		uint32_t instanceLayerCount;
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, NULL);
		std::vector<VkLayerProperties> availableInstanceLayers(instanceLayerCount);
		vkEnumerateInstanceLayerProperties(&instanceLayerCount, availableInstanceLayers.data());

		std::cout << "[INFO] available vulkan layers:" << std::endl;
		for (const auto& layer : availableInstanceLayers)
		{
			std::cout << "[INFO]     name: " << layer.layerName << " desc: " << layer.description << " impl_ver: "
				<< VK_VERSION_MAJOR(layer.implementationVersion) << "."
				<< VK_VERSION_MINOR(layer.implementationVersion) << "."
				<< VK_VERSION_PATCH(layer.implementationVersion)
				<< " spec_ver: "
				<< VK_VERSION_MAJOR(layer.specVersion) << "."
				<< VK_VERSION_MINOR(layer.specVersion) << "."
				<< VK_VERSION_PATCH(layer.specVersion)
				<< std::endl;
		}

		std::vector<VkExtensionProperties> availableInstanceExtensions = CsySmallVk::Query::instanceExtensionProperties();
		for (const auto& extension : availableInstanceExtensions)
		{
			std::cout << "[INFO]     name: " << extension.extensionName << " spec_ver: "
				<< VK_VERSION_MAJOR(extension.specVersion) << "."
				<< VK_VERSION_MINOR(extension.specVersion) << "."
				<< VK_VERSION_PATCH(extension.specVersion) << std::endl;
		}

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<const char*> instanceExtensions(glfwExtensionCount);
		std::memcpy(instanceExtensions.data(), glfwExtensions, sizeof(char*) * glfwExtensionCount);

		VkInstanceCreateInfo instanceCreateInfo = CsySmallVk::instanceCreateInfo();
		instanceCreateInfo.pApplicationInfo = &vkAppInfo;
		instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
		VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, NULL, &instance_));

		// CreateSurface
		if (glfwCreateWindowSurface(instance_, window_, NULL, &surface_) != VK_SUCCESS)
		{
			throw std::runtime_error("surface creation failed");
		}

		/*
			Vulkan device creation
		*/
		// Physical device (always use first)
		uint32_t deviceCount = 0;
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance_, &deviceCount, physicalDevices.data()));
		physicalDevice_ = physicalDevices[0];

		// get this device properties
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice_, &deviceProperties);
		LOG("GPU: %s\n", deviceProperties.deviceName);

		// get this device features
		vkGetPhysicalDeviceFeatures(physicalDevice_, &physicalDeviceFeatures_);

		// get this device properties
		auto physicalDeviceExtensions = CsySmallVk::Query::deviceExtensionProperties(physicalDevice_);

		// get memory properties
		vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &physicalDeviceMemoryProperties_);

		// Request queue
		auto queueFamilies = CsySmallVk::Query::physicalDeviceQueueFamilyProperties(physicalDevice_);
		// look for queue family indices
		for (uint32_t index = 0; index < queueFamilies.size(); index++)
		{
			// try to search a queue family that contain graphics queue, compute queue, and presentation queue
			// note: queue family index must be unique in the device queue create info
			VkBool32 presentationSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice_, index, surface_, &presentationSupport);
			if (queueFamilies[index].queueCount > 0 && queueFamilies[index].queueFlags & VK_QUEUE_GRAPHICS_BIT && presentationSupport && queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				queueFamilyIndex_ = index;
			}
		}
		if (queueFamilyIndex_ == UINT32_MAX)
		{
			throw std::runtime_error("unable to find a family queue with graphics, presentation, and compute queue");
		}

		const float queuePriorities[3]{ 1, 1, 1 };
		VkDeviceQueueCreateInfo queueCreateInfo = CsySmallVk::deviceQueueCreateInfo();
		queueCreateInfo.queueCount = 3;
		queueCreateInfo.pQueuePriorities = queuePriorities;
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;

		const char* enabledExtensions = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
		VkDeviceCreateInfo deviceCreateInfo = CsySmallVk::deviceCreateInfo();
		deviceCreateInfo.enabledExtensionCount = 1;
		deviceCreateInfo.ppEnabledExtensionNames = &enabledExtensions;
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = nullptr;
		deviceCreateInfo.pEnabledFeatures = nullptr;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		VK_CHECK_RESULT(vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_));

		// Get a compute queue
		vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &graphicsQueue_);
		vkGetDeviceQueue(device_, queueFamilyIndex_, 1, &computeQueue_);
		vkGetDeviceQueue(device_, queueFamilyIndex_, 2, &presentationQueue_);

		CreateSwapchain();

		uint32_t swapchainImageCount;
		vkGetSwapchainImagesKHR(device_, swapchain_, &swapchainImageCount, NULL);
		swapchainImage_.resize(swapchainImageCount);
		vkGetSwapchainImagesKHR(device_, swapchain_, &swapchainImageCount, swapchainImage_.data());
		std::cout << "Successfully get swapchain image" << std::endl;

		CreateSwapchainImageViews();
		CreateRenderPass();
		CreateSwapchainFrameBuffers();
		CreatePipelineCache();
		CreateDescriptorPool();
		CreateBuffers();
		CreateGraphicsPipelineLayout();
		CreateGraphicsPipeline();
		CreateGraphicsCommandPool();
		CreateGraphicsCommandBuffers();
		CreateSemaphores();
		CreateComputeDescriptorSetLayout();
		UpdateComputeDescriptorSets();
		CreateComputePipelineLayout();
		CreateComputePipelines();
		CreateComputeCommandPool();
		CreateComputeCommandBuffer();
		SetInitialParticleData();

		//// Compute command pool
		//VkCommandPoolCreateInfo cmdPoolInfo = {};
		//cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		//cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		//VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));

		///*
		//	Prepare storage buffers
		//*/
		//std::vector<uint32_t> computeInput(BUFFER_ELEMENTS);
		//std::vector<uint32_t> computeOutput(BUFFER_ELEMENTS);

		//// Fill input data
		//uint32_t n = 0;
		//std::generate(computeInput.begin(), computeInput.end(), [&n] { return n++; });

		//const VkDeviceSize bufferSize = BUFFER_ELEMENTS * sizeof(uint32_t);

		//VkBuffer deviceBuffer, hostBuffer;
		//VkDeviceMemory deviceMemory, hostMemory;

		//// Copy input data to VRAM using a staging buffer
		//{
		//	createBuffer(
		//		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		//		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		//		&hostBuffer,
		//		&hostMemory,
		//		bufferSize,
		//		computeInput.data());

		//	// Flush writes to host visible buffer
		//	void* mapped;
		//	vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
		//	VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
		//	mappedRange.memory = hostMemory;
		//	mappedRange.offset = 0;
		//	mappedRange.size = VK_WHOLE_SIZE;
		//	vkFlushMappedMemoryRanges(device, 1, &mappedRange);
		//	vkUnmapMemory(device, hostMemory);

		//	createBuffer(
		//		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		//		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		//		&deviceBuffer,
		//		&deviceMemory,
		//		bufferSize);

		//	// Copy to staging buffer
		//	VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		//	VkCommandBuffer copyCmd;
		//	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));
		//	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		//	VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));

		//	VkBufferCopy copyRegion = {};
		//	copyRegion.size = bufferSize;
		//	vkCmdCopyBuffer(copyCmd, hostBuffer, deviceBuffer, 1, &copyRegion);
		//	VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

		//	VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		//	submitInfo.commandBufferCount = 1;
		//	submitInfo.pCommandBuffers = &copyCmd;
		//	VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		//	VkFence fence;
		//	VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence));

		//	// Submit to the queue
		//	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		//	VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		//	vkDestroyFence(device, fence, nullptr);
		//	vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);
		//}

		///*
		//	Prepare compute pipeline
		//*/
		//{
		//	std::vector<VkDescriptorPoolSize> poolSizes = {
		//		vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
		//	};

		//	VkDescriptorPoolCreateInfo descriptorPoolInfo =
		//		vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), 1);
		//	VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		//	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
		//		vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
		//	};
		//	VkDescriptorSetLayoutCreateInfo descriptorLayout =
		//		vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		//	VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		//	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
		//		vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		//	VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));

		//	VkDescriptorSetAllocateInfo allocInfo =
		//		vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
		//	VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

		//	VkDescriptorBufferInfo bufferDescriptor = { deviceBuffer, 0, VK_WHOLE_SIZE };
		//	std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
		//		vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bufferDescriptor),
		//	};
		//	vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

		//	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		//	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		//	VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));

		//	// Create pipeline
		//	VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);

		//	// Pass SSBO size via specialization constant
		//	struct SpecializationData {
		//		uint32_t BUFFER_ELEMENT_COUNT = BUFFER_ELEMENTS;
		//	} specializationData;
		//	VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		//	VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(SpecializationData), &specializationData);

		//	std::string shaderDir = "glsl";
		//	if (commandLineParser.isSet("shaders")) {
		//		shaderDir = commandLineParser.getValueAsString("shaders", "glsl");
		//	}
		//	const std::string shadersPath = getShaderBasePath() + shaderDir + "/computeheadless/";

		//	VkPipelineShaderStageCreateInfo shaderStage = {};
		//	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		//	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		//	shaderStage.module = vks::tools::loadShader((shadersPath + "headless.comp.spv").c_str(), device);
		//	shaderStage.pName = "main";
		//	shaderStage.pSpecializationInfo = &specializationInfo;
		//	shaderModule = shaderStage.module;

		//	assert(shaderStage.module != VK_NULL_HANDLE);
		//	computePipelineCreateInfo.stage = shaderStage;
		//	VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));

		//	// Create a command buffer for compute operations
		//	VkCommandBufferAllocateInfo cmdBufAllocateInfo =
		//		vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		//	VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

		//	// Fence for compute CB sync
		//	VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		//	VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		//}

		///*
		//	Command buffer creation (for compute work submission)
		//*/
		//{
		//	VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		//	VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

		//	// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
		//	VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		//	bufferBarrier.buffer = deviceBuffer;
		//	bufferBarrier.size = VK_WHOLE_SIZE;
		//	bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
		//	bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		//	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//	vkCmdPipelineBarrier(
		//		commandBuffer,
		//		VK_PIPELINE_STAGE_HOST_BIT,
		//		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		//		VK_FLAGS_NONE,
		//		0, nullptr,
		//		1, &bufferBarrier,
		//		0, nullptr);

		//	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		//	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

		//	vkCmdDispatch(commandBuffer, BUFFER_ELEMENTS, 1, 1);

		//	// Barrier to ensure that shader writes are finished before buffer is read back from GPU
		//	bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		//	bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		//	bufferBarrier.buffer = deviceBuffer;
		//	bufferBarrier.size = VK_WHOLE_SIZE;
		//	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//	vkCmdPipelineBarrier(
		//		commandBuffer,
		//		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		//		VK_PIPELINE_STAGE_TRANSFER_BIT,
		//		VK_FLAGS_NONE,
		//		0, nullptr,
		//		1, &bufferBarrier,
		//		0, nullptr);

		//	// Read back to host visible buffer
		//	VkBufferCopy copyRegion = {};
		//	copyRegion.size = bufferSize;
		//	vkCmdCopyBuffer(commandBuffer, deviceBuffer, hostBuffer, 1, &copyRegion);

		//	// Barrier to ensure that buffer copy is finished before host reading from it
		//	bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		//	bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		//	bufferBarrier.buffer = hostBuffer;
		//	bufferBarrier.size = VK_WHOLE_SIZE;
		//	bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		//	bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//	vkCmdPipelineBarrier(
		//		commandBuffer,
		//		VK_PIPELINE_STAGE_TRANSFER_BIT,
		//		VK_PIPELINE_STAGE_HOST_BIT,
		//		VK_FLAGS_NONE,
		//		0, nullptr,
		//		1, &bufferBarrier,
		//		0, nullptr);

		//	VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

		//	// Submit compute work
		//	vkResetFences(device, 1, &fence);
		//	const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		//	VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		//	computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		//	computeSubmitInfo.commandBufferCount = 1;
		//	computeSubmitInfo.pCommandBuffers = &commandBuffer;
		//	VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, fence));
		//	VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		//	// Make device writes visible to the host
		//	void* mapped;
		//	vkMapMemory(device, hostMemory, 0, VK_WHOLE_SIZE, 0, &mapped);
		//	VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
		//	mappedRange.memory = hostMemory;
		//	mappedRange.offset = 0;
		//	mappedRange.size = VK_WHOLE_SIZE;
		//	vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);

		//	// Copy to output
		//	memcpy(computeOutput.data(), mapped, bufferSize);
		//	vkUnmapMemory(device, hostMemory);
		//}

		//vkQueueWaitIdle(queue);

		//// Output buffer contents
		//LOG("Compute input:\n");
		//for (auto v : computeInput) {
		//	LOG("%d \t", v);
		//}
		//std::cout << std::endl;

		//LOG("Compute output:\n");
		//for (auto v : computeOutput) {
		//	LOG("%d \t", v);
		//}
		//std::cout << std::endl;

		//// Clean up
		//vkDestroyBuffer(device, deviceBuffer, nullptr);
		//vkFreeMemory(device, deviceMemory, nullptr);
		//vkDestroyBuffer(device, hostBuffer, nullptr);
		//vkFreeMemory(device, hostMemory, nullptr);
	}

	~VulkanExample()
	{
		destroyVulkan();
		destroyWindow();
//		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
//		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
//		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
//		vkDestroyPipeline(device, pipeline, nullptr);
//		vkDestroyPipelineCache(device, pipelineCache, nullptr);
//		vkDestroyFence(device, fence, nullptr);
//		vkDestroyCommandPool(device, commandPool, nullptr);
//		vkDestroyShaderModule(device, shaderModule, nullptr);
//		vkDestroyDevice(device, nullptr);
//#if DEBUG
//		if (debugReportCallback) {
//			PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
//			assert(vkDestroyDebugReportCallback);
//			vkDestroyDebugReportCallback(instance, debugReportCallback, nullptr);
//		}
//#endif
//		vkDestroyInstance(instance, nullptr);
	}
};

int main(int argc, char* argv[]) {
	VulkanExample* vulkanExample = new VulkanExample();
	vulkanExample->Run();
	delete(vulkanExample);
	//system("pause");
	return 0;
}