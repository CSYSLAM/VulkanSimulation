#include "application.h"
#include "vkcsy.h"
#include <cmath>
#include <cstring>
#include <string>
#include <algorithm>
#include <exception>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <thread>

namespace SPH
{
	Application::Application()
	{
		InitializeWindow();
		InitializeVulkan();
	}

	Application::~Application()
	{

		destroyVulkan();
		destroyWindow();
	}

	void Application::destroyWindow()
	{
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void Application::destroyVulkan()
	{
		vkDestroySwapchainKHR(logicalDeviceHandle, swapchainHandle, NULL);
		vkDestroySurfaceKHR(instanceHandle, surfaceHandle, NULL);
		vkDestroyDevice(logicalDeviceHandle, NULL);
		vkDestroyInstance(instanceHandle, NULL);
	}

	void Application::InitializeWindow()
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
		window = glfwCreateWindow(1000, 1000, "", NULL, NULL);
		if (!window)
		{
			glfwTerminate();
			throw std::runtime_error("window creation failed");
		}
		glfwMakeContextCurrent(window);
		glfwSwapInterval(0);

		// pass Application pointer to the callback using GLFW user pointer
		glfwSetWindowUserPointer(window, reinterpret_cast<void*>(this));

		// set key callback
		auto key_callback = [](GLFWwindow* window, int key, int, int action, int)
		{
			auto app_ptr = reinterpret_cast<SPH::Application*>(glfwGetWindowUserPointer(window));
			if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
			{
				app_ptr->paused = !app_ptr->paused;
			}
			if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			{
				glfwSetWindowShouldClose(window, GLFW_TRUE);
			}
		};

		glfwSetKeyCallback(window, key_callback);
	}

	void Application::InitializeVulkan()
	{
		CreateInstance();
		CreateSurface();
		SelectPhysicalDevice();
		CreateLogicalDevice();
		GetDeviceQueues();
		CreateSwapchain();
		GetSwapchainImages();
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
	}

	void Application::CreateInstance()
	{
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
		//std::vector<VkLayerProperties> availableInstanceLayers = CsySmallVk::Query::instanceLayerProperties();
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

		if (vkCreateInstance(&instanceCreateInfo, NULL, &instanceHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("vulkan instance creation failed");
		}

	}

	void Application::CreateSurface()
	{
		if (glfwCreateWindowSurface(instanceHandle, window, NULL, &surfaceHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("surface creation failed");
		}
	}

	void Application::SelectPhysicalDevice()
	{
		auto physicalDevices = CsySmallVk::Query::physicalDevices(instanceHandle);
		// select first device and set it as the device used throughout the program
		physicalDeviceHandle = physicalDevices[0];

		// get this device properties
		vkGetPhysicalDeviceProperties(physicalDeviceHandle, &physicalDeviceProperties);
		// print device properties info
		std::cout << "[INFO] selected device name: " << physicalDeviceProperties.deviceName << std::endl
			<< "[INFO] selected device type: ";
		switch (physicalDeviceProperties.deviceType)
		{
		case VK_PHYSICAL_DEVICE_TYPE_OTHER:
			std::cout << "VK_PHYSICAL_DEVICE_TYPE_OTHER";
			break;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			std::cout << "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU";
			break;
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			std::cout << "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU";
			break;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			std::cout << "VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU";
			break;
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			std::cout << "VK_PHYSICAL_DEVICE_TYPE_CPU";
			break;
		default:
			;
		}
		std::cout << " (" << physicalDeviceProperties.deviceType << ")" << std::endl
			<< "[INFO] selected device driver version: "
			<< VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion) << "."
			<< VK_VERSION_MINOR(physicalDeviceProperties.driverVersion) << "."
			<< VK_VERSION_PATCH(physicalDeviceProperties.driverVersion) << std::endl
			<< "[INFO] selected device vulkan api version: "
			<< VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion) << "."
			<< VK_VERSION_MINOR(physicalDeviceProperties.apiVersion) << "."
			<< VK_VERSION_PATCH(physicalDeviceProperties.apiVersion) << std::endl;

		// get this device features
		 vkGetPhysicalDeviceFeatures(physicalDeviceHandle, &physicalDeviceFeatures);

		// get this device properties
		auto physicalDeviceExtensions = CsySmallVk::Query::deviceExtensionProperties(physicalDeviceHandle);
		std::cout << "[INFO] selected device available extensions:" << std::endl;
		for (const auto& extension : physicalDeviceExtensions)
		{
			std::cout << "[INFO]     name: " << extension.extensionName << " spec_ver: "
				<< VK_VERSION_MAJOR(extension.specVersion) << "."
				<< VK_VERSION_MINOR(extension.specVersion) << "."
				<< VK_VERSION_PATCH(extension.specVersion) << std::endl;
		}

		// get memory properties
		vkGetPhysicalDeviceMemoryProperties(physicalDeviceHandle, &physicalDeviceMemoryProperties);
	}

	void Application::CreateLogicalDevice()
	{
		auto queueFamilies = CsySmallVk::Query::physicalDeviceQueueFamilyProperties(physicalDeviceHandle);
		std::cout << "[INFO] available queue families:" << std::endl;
		// look for queue family indices
		for (uint32_t index = 0; index < queueFamilies.size(); index++)
		{
			std::cout << "[INFO]     flags: ";
			if (queueFamilies[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				std::cout << "VK_QUEUE_GRAPHICS_BIT ";
			}
			if (queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				std::cout << "VK_QUEUE_COMPUTE_BIT ";
			}
			if (queueFamilies[index].queueFlags & VK_QUEUE_TRANSFER_BIT)
			{
				std::cout << "VK_QUEUE_TRANSFER_BIT ";
			}
			if (queueFamilies[index].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
			{
				std::cout << "VK_QUEUE_SPARSE_BINDING_BIT ";
			}
			std::cout << "(" << queueFamilies[index].queueFlags << ") count: " << queueFamilies[index].queueCount << std::endl;

			// try to search a queue family that contain graphics queue, compute queue, and presentation queue
			// note: queue family index must be unique in the device queue create info
			VkBool32 presentationSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(physicalDeviceHandle, index, surfaceHandle, &presentationSupport);
			if (queueFamilies[index].queueCount > 0 && queueFamilies[index].queueFlags & VK_QUEUE_GRAPHICS_BIT && presentationSupport && queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				graphicsPresentationComputeQueueFamilyIndex = index;
			}
		}
		if (graphicsPresentationComputeQueueFamilyIndex == UINT32_MAX)
		{
			throw std::runtime_error("unable to find a family queue with graphics, presentation, and compute queue");
		}
		const float queuePriorities[3]{ 1, 1, 1 };
		VkDeviceQueueCreateInfo queueCreateInfo = CsySmallVk::deviceQueueCreateInfo();
		queueCreateInfo.queueCount = 3;
		queueCreateInfo.pQueuePriorities = queuePriorities;
		queueCreateInfo.queueFamilyIndex = graphicsPresentationComputeQueueFamilyIndex;
	
		const char* enabledExtensions = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
		VkDeviceCreateInfo deviceCreateInfo = CsySmallVk::deviceCreateInfo();
		deviceCreateInfo.enabledExtensionCount = 1;
		deviceCreateInfo.ppEnabledExtensionNames = &enabledExtensions;
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = nullptr;
		deviceCreateInfo.pEnabledFeatures = nullptr;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		if (vkCreateDevice(physicalDeviceHandle, &deviceCreateInfo, NULL, &logicalDeviceHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("logical device creation failed");
		}
	}

	void Application::GetDeviceQueues()
	{
		vkGetDeviceQueue(logicalDeviceHandle, graphicsPresentationComputeQueueFamilyIndex, 0, &graphicsQueueHandle);
		vkGetDeviceQueue(logicalDeviceHandle, graphicsPresentationComputeQueueFamilyIndex, 1, &computeQueueHandle);
		vkGetDeviceQueue(logicalDeviceHandle, graphicsPresentationComputeQueueFamilyIndex, 2, &presentationQueueHandle);
	}

	void Application::CreateSwapchain()
	{
		VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
		VkExtent2D extent = { windowWidth, windowHeight };

		// Query the surface capabilities and select the swapchain's extent (width, height).
		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		{
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDeviceHandle, surfaceHandle, &surfaceCapabilities);
			if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
				extent = surfaceCapabilities.currentExtent;
			}
		}

		// Select a surface format.
		{
			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDeviceHandle, surfaceHandle, &formatCount, NULL);
			std::vector<VkSurfaceFormatKHR> surfaceFormats;
			surfaceFormats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDeviceHandle, surfaceHandle, &formatCount, surfaceFormats.data());
			
			for (VkSurfaceFormatKHR entry : surfaceFormats) {
				if ((entry.format == VK_FORMAT_B8G8R8A8_SRGB) && (entry.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
					surfaceFormat = entry;
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
			create_info.surface = surfaceHandle;
			create_info.minImageCount = imageCount;
			create_info.imageFormat = surfaceFormat.format;
			create_info.imageColorSpace = surfaceFormat.colorSpace;
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

		if (vkCreateSwapchainKHR(logicalDeviceHandle, &create_info, nullptr, &swapchainHandle) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		std::cout << "Successfully created swapchain" << std::endl;
	}

	void Application::GetSwapchainImages()
	{
		uint32_t swapchainImageCount;
		vkGetSwapchainImagesKHR(logicalDeviceHandle, swapchainHandle, &swapchainImageCount, NULL);
		swapchainImageHandles.resize(swapchainImageCount);
		vkGetSwapchainImagesKHR(logicalDeviceHandle, swapchainHandle, &swapchainImageCount, swapchainImageHandles.data());
		std::cout << "Successfully get swapchain image" << std::endl;
	}

	void Application::CreateSwapchainImageViews()
	{
		swapchainImageViewHandles.resize(swapchainImageHandles.size());
		for (uint32_t i = 0; i < swapchainImageViewHandles.size(); i++)
		{
			VkImageViewCreateInfo imageViewCreateInfo
			{
				VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				NULL,
				0,
				swapchainImageHandles[i],
				VK_IMAGE_VIEW_TYPE_2D,
				surfaceFormat.format,
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
			if (vkCreateImageView(logicalDeviceHandle, &imageViewCreateInfo, NULL, &swapchainImageViewHandles[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("image views creation failed");
			}
		}
		std::cout << "Successfully create swapchain ImageViews" << std::endl;
	}

	void Application::CreateRenderPass()
	{
		VkAttachmentDescription attachmentDescription
		{
			0,
			surfaceFormat.format,
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
		if (vkCreateRenderPass(logicalDeviceHandle, &renderPassCreateInfo, NULL, &renderPassHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("render pass creation failed");
		}
		std::cout << "Successfully create render pass" << std::endl;
	}

	void Application::CreateSwapchainFrameBuffers()
	{
		swapchainFrameBufferHandles.resize(swapchainImageViewHandles.size());
		for (size_t index = 0; index < swapchainImageViewHandles.size(); index++)
		{
			VkFramebufferCreateInfo framebufferCreateInfo
			{
				VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
				NULL,
				0,
				renderPassHandle,
				1,
				& swapchainImageViewHandles[index],
				windowWidth,
				windowHeight,
				1
			};
			if (vkCreateFramebuffer(logicalDeviceHandle, &framebufferCreateInfo, NULL, &swapchainFrameBufferHandles[index]) != VK_SUCCESS)
			{
				throw std::runtime_error("frame buffer creation failed");
			}
		}
		std::cout << "Successfully create swapchain framebuffers" << std::endl;
	}

	void Application::CreatePipelineCache()
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo
		{
			VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
			NULL,
			0,
			0,
			NULL
		};
		if (vkCreatePipelineCache(logicalDeviceHandle, &pipelineCacheCreateInfo, NULL, &globalPipelineCacheHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("pipeline cache creation failed");
		}
		std::cout << "Successfully create pipelineCache" << std::endl;
	}

	void Application::CreateDescriptorPool()
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
			& descriptorPoolSize
		};
		if (vkCreateDescriptorPool(logicalDeviceHandle, &descriptorPoolCreateInfo, NULL, &globalDescriptorPoolHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("descriptor pool creation failed");
		}
		std::cout << "Successfully create descriptor pool" << std::endl;
	}

	uint32_t Application::findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties = CsySmallVk::Query::physicalDeviceMemoryProperties(physicalDeviceHandle);
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

	void Application::CreateBuffers()
	{
		VkBufferCreateInfo particlesBufferCreateInfo = CsySmallVk::bufferCreateInfo();
		particlesBufferCreateInfo.size = packedBufferSize;
		particlesBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		particlesBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		particlesBufferCreateInfo.queueFamilyIndexCount = 0;
		particlesBufferCreateInfo.pQueueFamilyIndices = nullptr;
		vkCreateBuffer(logicalDeviceHandle, &particlesBufferCreateInfo, NULL, &packedParticlesBufferHandle);
		
		VkMemoryRequirements positionBufferMemoryRequirements = CsySmallVk::Query::memoryRequirements(logicalDeviceHandle, packedParticlesBufferHandle);
		VkMemoryAllocateInfo particleBufferMemoryAllocationInfo = CsySmallVk::memoryAllocateInfo();
		particleBufferMemoryAllocationInfo.allocationSize = positionBufferMemoryRequirements.size;
		particleBufferMemoryAllocationInfo.memoryTypeIndex = findMemoryType(positionBufferMemoryRequirements,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(logicalDeviceHandle, &particleBufferMemoryAllocationInfo, NULL, &packedParticlesMemoryHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("memory allocation failed");
		}
		// bind the memory to the buffer object
		vkBindBufferMemory(logicalDeviceHandle, packedParticlesBufferHandle, packedParticlesMemoryHandle, 0);
		std::cout << "Successfully create buffers" << std::endl;
	}

	void Application::CreateGraphicsPipelineLayout()
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
		if (vkCreatePipelineLayout(logicalDeviceHandle, &pipelineLayoutCreateInfo, NULL, &graphicsPipelineLayoutHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("pipeline layout creation failed");
		}
		std::cout << "Successfully create graphics pipeline layout" << std::endl;
	}

	VkShaderModule Application::CreateShaderModule(const std::vector<char>& code)
	{
		VkShaderModule shaderModule;
		VkShaderModuleCreateInfo createInfo = CsySmallVk::shaderModuleCreateInfo();
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		if (vkCreateShaderModule(logicalDeviceHandle, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("fail to create shader module");
		return shaderModule;
	}

	void Application::CreateGraphicsPipeline()
	{
		std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;
		// create shader stage infos
		auto vertexShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "particle.vert.spv");
		VkShaderModule vertexShaderModule =  CreateShaderModule(vertexShaderCode);
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
			& vertexInputBindingDescription,
			1,
			& vertexInputAttributeDescription
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
			static_cast<float>(windowWidth),
			static_cast<float>(windowHeight),
			0,
			1
		};

		VkRect2D scissor
		{
			{ 0, 0 },
			{ windowWidth, windowHeight }
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
			& colorBlendAttachment,
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
			graphicsPipelineLayoutHandle,
			renderPassHandle,
			0,
			VK_NULL_HANDLE,
			-1
		};

		if (vkCreateGraphicsPipelines(logicalDeviceHandle, globalPipelineCacheHandle, 1, &graphicsPipelineCreateInfo, NULL, &graphicsPipelineHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("graphics pipeline creation failed");
		}
		std::cout << "Successfully create graphics pipeline" << std::endl;
	}

	void Application::CreateGraphicsCommandPool()
	{
		VkCommandPoolCreateInfo graphicsCommandPoolCreateInfo = CsySmallVk::commandPoolCreateInfo();
		graphicsCommandPoolCreateInfo.queueFamilyIndex = graphicsPresentationComputeQueueFamilyIndex;
		if (vkCreateCommandPool(logicalDeviceHandle, &graphicsCommandPoolCreateInfo, NULL, &graphicsCommandPoolHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("command pool creation failed");
		}
		std::cout << "Successfully create graphics command pool" << std::endl;
	}

	void Application::CreateGraphicsCommandBuffers()
	{
		graphicsCommandBufferHandles.resize(swapchainFrameBufferHandles.size());
		VkCommandBufferAllocateInfo graphicsCommandBufferAllocationInfo
		{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			NULL,
			graphicsCommandPoolHandle,
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			static_cast<uint32_t>(graphicsCommandBufferHandles.size())
		};
		if (vkAllocateCommandBuffers(logicalDeviceHandle, &graphicsCommandBufferAllocationInfo, graphicsCommandBufferHandles.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffers allocation failed");
		}
		for (size_t i = 0; i < graphicsCommandBufferHandles.size(); i++)
		{
			VkCommandBufferBeginInfo commandBufferBeginInfo
			{
				VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				NULL,
				VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
				NULL
			};
			vkBeginCommandBuffer(graphicsCommandBufferHandles[i], &commandBufferBeginInfo);
			VkClearValue clear_value{ 0.92f, 0.92f, 0.92f, 1.0f };
			VkRenderPassBeginInfo renderPassBeginInfo
			{
				VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				NULL,
				renderPassHandle,
				swapchainFrameBufferHandles[i],
				{
					{ 0, 0 },
					{ windowWidth, windowHeight }
				},
				1,
				&clear_value
			};
			vkCmdBeginRenderPass(graphicsCommandBufferHandles[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			VkViewport viewport
			{
				0,
				0,
				static_cast<float>(windowWidth),
				static_cast<float>(windowHeight),
				0,
				1
			};

			VkRect2D scissor
			{
				{ 0, 0 },
				{ windowWidth, windowHeight }
			};

			vkCmdSetViewport(graphicsCommandBufferHandles[i], 0, 1, &viewport);
			vkCmdSetScissor(graphicsCommandBufferHandles[i], 0, 1, &scissor);
			vkCmdBindPipeline(graphicsCommandBufferHandles[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineHandle);
			
			VkDeviceSize offsets = 0;
			vkCmdBindVertexBuffers(graphicsCommandBufferHandles[i], 0, 1, &packedParticlesBufferHandle, &offsets);
			vkCmdDraw(graphicsCommandBufferHandles[i], SPH_NUM_PARTICLES, 1, 0, 0);
			vkCmdEndRenderPass(graphicsCommandBufferHandles[i]);

			if (vkEndCommandBuffer(graphicsCommandBufferHandles[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("command buffer creation failed");
			}
		}
		std::cout << "Successfully create graphics command buffers" << std::endl;
	}

	void Application::CreateSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo
		{
			VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
			NULL,
			0
		};
		if (vkCreateSemaphore(logicalDeviceHandle, &semaphoreCreateInfo, NULL, &imageAvailableSemaphoreHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("semaphore creation failed");
		}
		if (vkCreateSemaphore(logicalDeviceHandle, &semaphoreCreateInfo, NULL, &renderFinishedSemaphoreHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("semaphore creation failed");
		}
		std::cout << "Successfully create semaphores" << std::endl;
	}

	void Application::CreateComputeDescriptorSetLayout()
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
		if (vkCreateDescriptorSetLayout(logicalDeviceHandle, &descriptorSetLayoutCreateInfo, NULL, &computeDescriptorSetLayoutHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("compute descriptor layout creation failed");
		}
		std::cout << "Successfully create compute descriptorSet layout" << std::endl;
	}

	void Application::UpdateComputeDescriptorSets()
	{
		// allocate descriptor sets
		VkDescriptorSetAllocateInfo allocInfo = CsySmallVk::descriptorSetAllocateInfo();
		allocInfo.descriptorPool = globalDescriptorPoolHandle;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &computeDescriptorSetLayoutHandle;
		if (vkAllocateDescriptorSets(logicalDeviceHandle, &allocInfo, &computeDescriptorSetHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("compute descriptor set allocation failed");
		}

		VkDescriptorBufferInfo descriptorBufferInfos[5];
		descriptorBufferInfos[0].buffer = packedParticlesBufferHandle;
		descriptorBufferInfos[0].offset = positionSsboOffset;
		descriptorBufferInfos[0].range = positionSsboSize;
		descriptorBufferInfos[1].buffer = packedParticlesBufferHandle;
		descriptorBufferInfos[1].offset = velocitySsboOffset;
		descriptorBufferInfos[1].range = velocitySsboSize;
		descriptorBufferInfos[2].buffer = packedParticlesBufferHandle;
		descriptorBufferInfos[2].offset = forceSsboOffset;
		descriptorBufferInfos[2].range = forceSsboSize;
		descriptorBufferInfos[3].buffer = packedParticlesBufferHandle;
		descriptorBufferInfos[3].offset = densitySsboOffset;
		descriptorBufferInfos[3].range = densitySsboSize;
		descriptorBufferInfos[4].buffer = packedParticlesBufferHandle;
		descriptorBufferInfos[4].offset = pressureSsboOffset;
		descriptorBufferInfos[4].range = pressureSsboSize;

		// write descriptor sets
		VkWriteDescriptorSet writeDescriptorSets[5];
		for (int index = 0; index < 5; index++)
		{
			VkWriteDescriptorSet write = CsySmallVk::writeDescriptorSet();
			write.descriptorCount = 1;
			write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			write.dstBinding = index;
			write.dstArrayElement = 0;
			write.dstSet = computeDescriptorSetHandle;
			write.pBufferInfo = &descriptorBufferInfos[index];
			writeDescriptorSets[index] = write;
		}

		vkUpdateDescriptorSets(logicalDeviceHandle, 5, writeDescriptorSets, 0, NULL);
		std::cout << "Successfully update compute descriptorsets" << std::endl;
	}

	void Application::CreateComputePipelineLayout()
	{
		VkPipelineLayoutCreateInfo layoutCreateInfo = CsySmallVk::pipelineLayoutCreateInfo();
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &computeDescriptorSetLayoutHandle;
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(logicalDeviceHandle, &layoutCreateInfo, nullptr, &computePipelineLayoutHandle)!= VK_SUCCESS)
			throw std::runtime_error("failed to create pipeline layout!");
		std::cout << "Successfully create compute pipeline layout" << std::endl;
	}

	void Application::CreateComputePipelines()
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
		createInfo.layout = computePipelineLayoutHandle; 
		if (vkCreateComputePipelines(logicalDeviceHandle, globalPipelineCacheHandle, 1, &createInfo, NULL, &computePipelineHandles[0]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}
		
		//second
		auto computeForceShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "compute_force.comp.spv");
		VkShaderModule computeForceShaderModule = CreateShaderModule(computeForceShaderCode);
		shaderStageCreateInfo.module = computeForceShaderModule;
		createInfo.stage = shaderStageCreateInfo;
		if (vkCreateComputePipelines(logicalDeviceHandle, globalPipelineCacheHandle, 1, &createInfo, NULL, &computePipelineHandles[1]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}

		//third
		auto integrateShaderCode = CsySmallVk::readFile(MU_SHADER_PATH "integrate.comp.spv");
		VkShaderModule integrateShaderModule = CreateShaderModule(integrateShaderCode);
		shaderStageCreateInfo.module = integrateShaderModule;
		createInfo.stage = shaderStageCreateInfo;
		if (vkCreateComputePipelines(logicalDeviceHandle, globalPipelineCacheHandle, 1, &createInfo, NULL, &computePipelineHandles[2]) != VK_SUCCESS)
		{
			throw std::runtime_error("first compute pipeline creation failed");
		}
		std::cout << "Successfully create compute pipelines" << std::endl;
	}

	void Application::CreateComputeCommandPool()
	{
		VkCommandPoolCreateInfo createInfo = CsySmallVk::commandPoolCreateInfo();
		createInfo.queueFamilyIndex = graphicsPresentationComputeQueueFamilyIndex;
		createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		if (vkCreateCommandPool(logicalDeviceHandle, &createInfo, NULL, &computeCommandPoolHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("command pool creation failed");
		}
		
		std::cout << "Successfully create compute command pool" << std::endl;
	}

	void Application::CreateComputeCommandBuffer()
	{
		VkCommandBufferAllocateInfo allocInfo = CsySmallVk::commandBufferAllocateInfo();
		allocInfo.commandBufferCount = 1;
		allocInfo.commandPool = computeCommandPoolHandle;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		if (vkAllocateCommandBuffers(logicalDeviceHandle, &allocInfo, &computeCommandBufferHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("buffer allocation failed");
		}
		VkCommandBufferBeginInfo beginInfo = CsySmallVk::commandBufferBeginInfo();
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		if (vkBeginCommandBuffer(computeCommandBufferHandle, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer begin failed");
		}
		vkCmdBindDescriptorSets(computeCommandBufferHandle, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayoutHandle, 0, 1, &computeDescriptorSetHandle, 0, NULL);
		// First dispatch
		vkCmdBindPipeline(computeCommandBufferHandle, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineHandles[0]);
		vkCmdDispatch(computeCommandBufferHandle, SPH_NUM_WORK_GROUPS, 1, 1);

		// Barrier: compute to compute dependencies
		// First dispatch writes to a storage buffer, second dispatch reads from that storage buffer
		vkCmdPipelineBarrier(computeCommandBufferHandle, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
	
		// Second dispatch
		vkCmdBindPipeline(computeCommandBufferHandle, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineHandles[1]);
		vkCmdDispatch(computeCommandBufferHandle, SPH_NUM_WORK_GROUPS, 1, 1);

		// Barrier: compute to compute dependencies
		// Second dispatch writes to a storage buffer, third dispatch reads from that storage buffer
		vkCmdPipelineBarrier(computeCommandBufferHandle, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
	
		// Third dispatch
		// Third dispatch writes to the storage buffer. Later, vkCmdDraw reads that buffer as a vertex buffer with vkCmdBindVertexBuffers.
		vkCmdBindPipeline(computeCommandBufferHandle, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineHandles[2]);
		vkCmdDispatch(computeCommandBufferHandle, SPH_NUM_WORK_GROUPS, 1, 1);
	
		vkCmdPipelineBarrier(computeCommandBufferHandle, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 0, NULL);
		vkEndCommandBuffer(computeCommandBufferHandle);
		std::cout << "Successfully create compute command buffer" << std::endl;
	}

	void Application::SetInitialParticleData()
	{
		// staging buffer
		VkBuffer stagingBufferHandle = VK_NULL_HANDLE;
		VkDeviceMemory stagingBufferMemoryDeviceHandle = VK_NULL_HANDLE;
		VkBufferCreateInfo stagingBufferCreateInfo = CsySmallVk::bufferCreateInfo();
		stagingBufferCreateInfo.size = packedBufferSize;
		stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		stagingBufferCreateInfo.queueFamilyIndexCount = 0;
		stagingBufferCreateInfo.pQueueFamilyIndices = nullptr;

		vkCreateBuffer(logicalDeviceHandle, &stagingBufferCreateInfo, NULL, &stagingBufferHandle);

		VkMemoryRequirements stagingBufferMemoryRequirements;
		vkGetBufferMemoryRequirements(logicalDeviceHandle, stagingBufferHandle, &stagingBufferMemoryRequirements);
		
		VkMemoryAllocateInfo allocInfo = CsySmallVk::memoryAllocateInfo();
		allocInfo.allocationSize = stagingBufferMemoryRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(stagingBufferMemoryRequirements,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		if (vkAllocateMemory(logicalDeviceHandle, &allocInfo, NULL, &stagingBufferMemoryDeviceHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("memory allocation failed");
		}

		// bind the memory to the buffer object
		vkBindBufferMemory(logicalDeviceHandle, stagingBufferHandle, stagingBufferMemoryDeviceHandle, 0);
		
		void* mappedMemory = NULL;
		vkMapMemory(logicalDeviceHandle, stagingBufferMemoryDeviceHandle, 0, stagingBufferMemoryRequirements.size, 0, &mappedMemory);
		
		// set the initial particles data
		std::vector<glm::vec2> initialParticlePosition(SPH_NUM_PARTICLES);
		for (auto i = 0, x = 0, y = 0; i < SPH_NUM_PARTICLES; i++)
		{
			initialParticlePosition[i].x = -0.625f + SPH_PARTICLE_RADIUS * 2 * x;
			initialParticlePosition[i].y = -1 + SPH_PARTICLE_RADIUS * 2 * y;
			x++;
			if (x >= 125)
			{
				x = 0;
				y++;
			}
		}
		// zero all 
		std::memset(mappedMemory, 0, packedBufferSize);
		std::memcpy(mappedMemory, initialParticlePosition.data(), positionSsboSize);
		vkUnmapMemory(logicalDeviceHandle, stagingBufferMemoryDeviceHandle);

		// submit a command buffer to copy staging buffer to the particle buffer 
		VkCommandBuffer copyCommandBufferHandle;
		VkCommandBufferAllocateInfo copyCommandBufferAllocationInfo = CsySmallVk::commandBufferAllocateInfo();
		copyCommandBufferAllocationInfo.commandBufferCount = 1;
		copyCommandBufferAllocationInfo.commandPool = computeCommandPoolHandle;
		copyCommandBufferAllocationInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(logicalDeviceHandle, &copyCommandBufferAllocationInfo, &copyCommandBufferHandle) != VK_SUCCESS)
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

		vkCmdCopyBuffer(copyCommandBufferHandle, stagingBufferHandle, packedParticlesBufferHandle, 1, &bufferCopyRegion);
		if (vkEndCommandBuffer(copyCommandBufferHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer end failed");
		}
		VkSubmitInfo submitInfo = CsySmallVk::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCommandBufferHandle;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.signalSemaphoreCount = 0;
		if (vkQueueSubmit(computeQueueHandle, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("command buffer submission failed");
		}
		if (vkQueueWaitIdle(computeQueueHandle) != VK_SUCCESS)
		{
			throw std::runtime_error("vkQueueWaitIdle failed");
		}
		vkFreeCommandBuffers(logicalDeviceHandle, computeCommandPoolHandle, 1, &copyCommandBufferHandle);
		vkFreeMemory(logicalDeviceHandle, stagingBufferMemoryDeviceHandle, NULL);
		vkDestroyBuffer(logicalDeviceHandle, stagingBufferHandle, NULL);
		std::cout << "Successfully set initial particle data" << std::endl;
	}

	void Application::RunSimulation()
	{
		if (vkQueueSubmit(computeQueueHandle, 1, &computeSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("compute queue submission failed");
		}
	}

	void Application::Render()
	{
		// submit graphics command buffer
		vkAcquireNextImageKHR(logicalDeviceHandle, swapchainHandle, UINT64_MAX, imageAvailableSemaphoreHandle, VK_NULL_HANDLE, &imageIndex);
		graphicsSubmitInfo.pCommandBuffers = graphicsCommandBufferHandles.data() + imageIndex;
		if (vkQueueSubmit(graphicsQueueHandle, 1, &graphicsSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("graphics queue submission failed");
		}
		// queue the image for presentation
		vkQueuePresentKHR(presentationQueueHandle, &presentInfo);

		vkQueueWaitIdle(presentationQueueHandle);
	}

	void Application::MainLoop()
	{
		static std::chrono::high_resolution_clock::time_point frame_start;
		static std::chrono::high_resolution_clock::time_point frame_end;
		static int64_t total_frame_time_ns;

		frame_start = std::chrono::high_resolution_clock::now();

		// process user inputs
		glfwPollEvents();

		// step through the simulation if not paused
		if (!paused)
		{
			RunSimulation();
			frameNumber++;
		}

		Render();
		frame_end = std::chrono::high_resolution_clock::now();
		// measure performance
		total_frame_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - frame_start).count();
		std::stringstream title;
		title.precision(3);
		title.setf(std::ios_base::fixed, std::ios_base::floatfield);
		title << "SPH (Vulkan) | "
			<< SPH_NUM_PARTICLES << " particles | "
			"frame #" << frameNumber << " | "
			"render latency: " << 1e-6 * total_frame_time_ns << " ms | "
			"FPS: " << 1.0 / (1e-9 * total_frame_time_ns);
		glfwSetWindowTitle(window, title.str().c_str());
	}

	void Application::Run()
	{
		// to measure performance
		std::thread
		(
			[this]()
			{
				std::this_thread::sleep_for(std::chrono::seconds(20));
				std::cout << "[INFO] frame count after 20 seconds after setup (do not pause or move the window): " << frameNumber << std::endl;
			}
		).detach();

		while (!glfwWindowShouldClose(window))
		{
			MainLoop();
		}
	}
}