#include "VkSystem.h"
#include "VkTools.h"
#include "VkContext.h"

#include <iostream>

namespace CsyVk {

	VkSystem* VkSystem::instance()
	{
		static VkSystem gInstance;
		return &gInstance;
	}

	VkSystem::VkSystem()
	{
	}

	VkSystem::~VkSystem()
	{
		if (ctx != nullptr) {
			delete ctx;
		}

		vkDestroyInstance(vkInstance, nullptr);
	}

	bool VkSystem::initializeWithInstance(VkData vkData)
	{
		vkInstance = vkData.instance_;
		physicalDevice = vkData.physicalDevice_;

		ctx = new VkContext(physicalDevice);
		VkResult res = ctx->setLogicalDevice(vkData.device_, vkData.queue_, vkData.queueFamilyIndex_, vkData.useUniQueue_);
		if (res != VK_SUCCESS) {
			// Could not create Vulkan device
			return false;
		}

		if (useMemoryPool) {
			res = ctx->createMemoryPool(vkInstance, apiVersion);
			if (res != VK_SUCCESS) {
				// Could not create Vulkan memory pool
				return false;
			}
		}

		return true;
	}
}