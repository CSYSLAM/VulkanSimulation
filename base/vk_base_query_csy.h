#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#include <set>

namespace CsySmallVk
{
	struct Query
	{
		_NODISCARD
			static std::vector<VkExtensionProperties> instanceExtensionProperties()
		{
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> ext(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, ext.data());
			return ext;
		}

		_NODISCARD
			static std::vector<VkPhysicalDevice> physicalDevices(VkInstance instance)
		{
			uint32_t deviceCount;
			vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
			return devices;
		}
		_NODISCARD
			static std::vector<VkQueueFamilyProperties> physicalDeviceQueueFamilyProperties(VkPhysicalDevice device)
		{
			uint32_t propertyCount;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &propertyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(propertyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &propertyCount, queueFamilies.data());
			return queueFamilies;
		}
		_NODISCARD
			static std::vector<VkExtensionProperties> deviceExtensionProperties(VkPhysicalDevice device)
		{
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
			return availableExtensions;
		}
		_NODISCARD
			static VkPhysicalDeviceProperties physicalDeviceProperties(VkPhysicalDevice device)
		{
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(device, &properties);
			return properties;
		}
		_NODISCARD
			static VkMemoryRequirements memoryRequirements(VkDevice device, VkBuffer buffer)
		{
			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
			return memoryRequirements;
		}
		_NODISCARD
			static VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties(VkPhysicalDevice device)
		{
			VkPhysicalDeviceMemoryProperties memoryProperties;
			vkGetPhysicalDeviceMemoryProperties(device, &memoryProperties);
			return memoryProperties;
		}
	};
}