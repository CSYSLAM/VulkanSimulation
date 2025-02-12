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
		if (validation && debugUtilsMessenger != nullptr)
		{
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(vkInstance, debugUtilsMessenger, nullptr);
			}
		}

		if (ctx != nullptr) {
			delete ctx;
		}

		vkDestroyInstance(vkInstance, nullptr);
	}

	bool VkSystem::initialize(bool enableValidation)
	{
		validation = enableValidation;
		VkResult err;

		// Vulkan instance
		err = createVulkanInstance();
		if (err) {
			return false;
		}

		// If requested, we enable the default validation layers for debugging
		if (validation) {}

		// Physical device
		uint32_t gpuCount = 0;
		// Get number of available physical devices
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(vkInstance, &gpuCount, nullptr));
		assert(gpuCount > 0);
		// Enumerate devices
		std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
		err = vkEnumeratePhysicalDevices(vkInstance, &gpuCount, physicalDevices.data());
		if (err) {
			// Could not enumerate physical devices
			return false;
		}

		// GPU selection

		// Select physical device to be used for the Vulkan example
		// Defaults to the first device unless specified by command line
		uint32_t selectedDevice = 0;

		physicalDevice = physicalDevices[selectedDevice];

		// Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);

		ctx = new VkContext(physicalDevice);
		VkResult res = ctx->createLogicalDevice(enabledFeatures, enabledDeviceExtensions, deviceCreatepNextChain);
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

	VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData)
	{
		return VK_FALSE;
	}

	VkResult VkSystem::createVulkanInstance()
	{
		// Validation can also be forced via a define
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = name.c_str();
		appInfo.pEngineName = name.c_str();
		appInfo.apiVersion = apiVersion;

		std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

		// Enable surface extensions depending on os
#if defined(_WIN32)
		instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
		instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
		instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
		instanceExtensions.push_back(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
		instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
		instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
		instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
		instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#endif

		// Get extensions supported by the instance and store for later use
		uint32_t extCount = 0;
		std::vector<std::string> supportedInstanceExtensions;

		vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
		if (extCount > 0)
		{
			std::vector<VkExtensionProperties> extensions(extCount);
			if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, &extensions.front()) == VK_SUCCESS)
			{
				for (VkExtensionProperties extension : extensions)
				{
					supportedInstanceExtensions.push_back(extension.extensionName);
				}
			}
		}

		// Enabled requested instance extensions
		if (enabledInstanceExtensions.size() > 0)
		{
			for (const char * enabledExtension : enabledInstanceExtensions)
			{
				// Output message if requested extension is not available
				if (std::find(supportedInstanceExtensions.begin(), supportedInstanceExtensions.end(), enabledExtension) == supportedInstanceExtensions.end())
				{
					// Enabled instance extension enabledExtension is not present at instance level;
				}
				instanceExtensions.push_back(enabledExtension);
			}
		}

		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pNext = NULL;
		instanceCreateInfo.pApplicationInfo = &appInfo;
				
		if (instanceExtensions.size() > 0)
		{
			if (validation)
			{
				instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}
			instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
			instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
		}


		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugCreateInfo.messageSeverity = 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = 
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugUtilsMessengerCallback;

		// also enable shader debug print
		VkValidationFeatureEnableEXT enabled[] = { VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT };
		VkValidationFeaturesEXT validationFeatures{};
		validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
		validationFeatures.enabledValidationFeatureCount = 1;
		validationFeatures.pEnabledValidationFeatures = enabled;
		validationFeatures.disabledValidationFeatureCount = 0;
		validationFeatures.pDisabledValidationFeatures = nullptr;

		debugCreateInfo.pNext = &validationFeatures;

		if (validation)
		{
			const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
			uint32_t instanceLayerCount;
			vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
			std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
			vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
			bool validationLayerPresent = false;
			for (VkLayerProperties layer : instanceLayerProperties) {
				if (strcmp(layer.layerName, validationLayerName) == 0) {
					validationLayerPresent = true;
					break;
				}
			}
			if (validationLayerPresent) {
				instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
				instanceCreateInfo.enabledLayerCount = 1;
				instanceCreateInfo.pNext = &debugCreateInfo;
			}
			else {
				// Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled;
				instanceCreateInfo.enabledLayerCount = 0;
				instanceCreateInfo.pNext = nullptr;
			}			
		}
		VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &vkInstance);

		if (result == VK_SUCCESS && validation) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(vkInstance, "vkCreateDebugUtilsMessengerEXT");

			if (func != nullptr) {
				VkResult r = func(vkInstance, &debugCreateInfo, nullptr, &debugUtilsMessenger);

				if (r != VK_SUCCESS) {
					// Failed to create VkDebugUtilsMessengerEXT;
				}
			}
		}
		return result;
	}

}