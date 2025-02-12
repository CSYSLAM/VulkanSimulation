#pragma once
#include "vulkan/vulkan.h"

#include <vector>
#include <string>

namespace CsyVk 
{
	class VkContext;

	class VkSystem {

	public:
		static VkSystem* instance();

		VkContext* currentContext() { return ctx; }

		bool initialize(bool enableValidation = false);

		VkInstance instanceHandle() { return vkInstance; }

		void enableMemoryPool(bool enableMemPool = false) {
			useMemoryPool = enableMemPool;
		}

		VkPhysicalDeviceProperties getDeviceProperties() {
			return deviceProperties;
		}

		VkPhysicalDevice getPhysicalDevice() {
			return physicalDevice;
		}

	private:
		VkSystem();
		~VkSystem();

		VkResult createVulkanInstance();
		VkContext* ctx = nullptr;

		bool validation;
		bool useMemoryPool = true;
		std::string name = "Vulkan";
		uint32_t apiVersion = VK_API_VERSION_1_2;
		VkInstance vkInstance;
		VkPhysicalDevice physicalDevice;
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		VkPhysicalDeviceFeatures enabledFeatures{};
		void* deviceCreatepNextChain = nullptr;

		VkDebugUtilsMessengerEXT debugUtilsMessenger = nullptr;
				
	public:
		std::vector<const char*> enabledDeviceExtensions;
		std::vector<const char*> enabledInstanceExtensions;

	};
}