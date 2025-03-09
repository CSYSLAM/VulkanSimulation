#pragma once
#include "vulkan/vulkan.h"

#include <vector>
#include <string>
#include <optional>

struct VkData {
	VkInstance instance_;
	VkPhysicalDevice physicalDevice_;
	std::vector<uint32_t> queueFamilyIndex_;
	VkDevice device_;
	std::vector<VkQueue> queue_;
	bool useUniQueue_ = true;
};

namespace CsyVk 
{
	class VkContext;

	class VkSystem {

	public:
		static VkSystem* instance();

		VkContext* currentContext() { return ctx; }

		bool initializeWithInstance(VkData vkData);

		VkInstance instanceHandle() { return vkInstance; }

		void enableMemoryPool(bool enableMemPool = false) {
			useMemoryPool = enableMemPool;
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
	};
}