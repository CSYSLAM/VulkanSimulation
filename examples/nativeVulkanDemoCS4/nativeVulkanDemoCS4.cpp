#pragma comment(linker, "/subsystem:console")
#include "VkSystem.h"
#include "Array.h"
#include "VkProgram.h"
#include "VkBuffer.h"
#include <optional>
#include <array>
#include <cstring>
#include  "SceneGraph.h"
#include  "ComputeDemoNode.h"
#include "SceneGraphFactory.h"

#include "u_vk_csy.h"

using namespace CsyVk;

std::string shaderDir = "C:\\temp\\CG\\Engine\\VulkanSimulation\\shaders\\glsl\\nativeVulkanDemoCS4\\VecAdd.comp.spv";
std::array<float, 100> inputData;
std::array<float, 100> outputData;
constexpr VkDeviceSize inputDataSize() { return sizeof(inputData); }

VkData vkData;
VkInstance instance;
VkPhysicalDevice physicalDevice;
std::optional<uint32_t> queueFamilyIndex;
VkDevice device;
VkQueue queue;

// Helper function to find suitable memory type
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

int main(int argc, char* argv[])
{
	for (int i = 0; i < 100; i++)
	{
		inputData[i] = 33.0f;
		outputData[i] = 0.0f;
	}

	VkApplicationInfo appInfo = CsySmallVk::applicationInfo();
	appInfo.pApplicationName = "Csy Compute Shader";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 1);
	appInfo.pEngineName = "None";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instanceCreateInfo = CsySmallVk::instanceCreateInfo();
	instanceCreateInfo.pApplicationInfo = &appInfo;

	//get extension properties
	auto extensionProperties = CsySmallVk::Query::instanceExtensionProperties();

	VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
	if (result != VkResult::VK_SUCCESS)
		throw std::runtime_error("failed to create instance");

	std::cout << "successfully!" << std::endl;

	auto physicalDevices = CsySmallVk::Query::physicalDevices(instance);
	for (const auto device : physicalDevices)
	{
		auto queueFamilies = CsySmallVk::Query::physicalDeviceQueueFamilyProperties(device);
		for (size_t i = 0; i < queueFamilies.size(); ++i)
		{
			if (queueFamilies[i].queueFlags & (VK_QUEUE_COMPUTE_BIT))
			{
				queueFamilyIndex = i;
				physicalDevice = device;
				break;
			}
		}
		if (queueFamilyIndex.has_value()) break;
	}
	if (!queueFamilyIndex.has_value())
		throw std::runtime_error("can't find a family that contains compute queue!");
	else
	{
		std::cout << "Select Physical Device:" << physicalDevice << std::endl;
		auto extensions = CsySmallVk::Query::deviceExtensionProperties(physicalDevice);
		std::cout << "Select Queue Index:" << queueFamilyIndex.value() << std::endl;
	}
	auto p = CsySmallVk::Query::physicalDeviceProperties(physicalDevice);

	VkDeviceCreateInfo createInfo = CsySmallVk::deviceCreateInfo();
	createInfo.enabledExtensionCount = 0;
	createInfo.ppEnabledExtensionNames = nullptr;
	createInfo.enabledLayerCount = 0;
	createInfo.ppEnabledLayerNames = nullptr;
	createInfo.pEnabledFeatures = nullptr;

	float priority = 1.0f; //default
	VkDeviceQueueCreateInfo queueCreateInfo = CsySmallVk::deviceQueueCreateInfo();
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &priority;
	queueCreateInfo.queueFamilyIndex = queueFamilyIndex.value();

	createInfo.queueCreateInfoCount = 1;
	createInfo.pQueueCreateInfos = &queueCreateInfo;
	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create logical device");
	}
	vkGetDeviceQueue(device, queueFamilyIndex.value(), 0, &queue);

	vkData.instance_ = instance;
	vkData.physicalDevice_ = physicalDevice;
	vkData.device_ = device;
	vkData.queue_ = queue;
	vkData.queueFamilyIndex_ = queueFamilyIndex.value();
	VkSystem::instance()->initializeWithInstance(vkData);
	DArray<float> dA(100);

	VkCommandPool commandPool;
	VkCommandPoolCreateInfo createInfo11 = CsySmallVk::commandPoolCreateInfo();
	createInfo11.queueFamilyIndex = queueFamilyIndex.value();
	if (vkCreateCommandPool(device, &createInfo11, nullptr, &commandPool)
		!= VK_SUCCESS)
		throw std::runtime_error("failed to create command pool!");

	// Create staging buffer and allocate memory
	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	VkBufferCreateInfo stagingBufferCreateInfo = {};
	stagingBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferCreateInfo.size = inputDataSize();
	stagingBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	stagingBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(device, &stagingBufferCreateInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to create staging buffer!");
	}

	VkMemoryRequirements stagingMemRequirements;
	vkGetBufferMemoryRequirements(device, stagingBuffer, &stagingMemRequirements);

	VkMemoryAllocateInfo stagingAllocInfo = {};
	stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	stagingAllocInfo.allocationSize = stagingMemRequirements.size;
	stagingAllocInfo.memoryTypeIndex = findMemoryType(stagingMemRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	if (vkAllocateMemory(device, &stagingAllocInfo, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate staging buffer memory!");
	}

	vkBindBufferMemory(device, stagingBuffer, stagingBufferMemory, 0);

	// Copy inputData to staging buffer
	void* stagingData;
	vkMapMemory(device, stagingBufferMemory, 0, stagingBufferCreateInfo.size, 0, &stagingData);
	memcpy(stagingData, inputData.data(), (size_t) stagingBufferCreateInfo.size);
	vkUnmapMemory(device, stagingBufferMemory);

	// Copy data from staging buffer to storage buffer
	VkCommandBufferAllocateInfo allocInfoCmd = {};
	allocInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfoCmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfoCmd.commandPool = commandPool; // You need to create or have a commandPool
	allocInfoCmd.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &allocInfoCmd, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = inputDataSize();
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, dA.mData.buffer->buffer, 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
	vkDestroyBuffer(device, stagingBuffer, nullptr);

	std::cout << "-----------------------------------------------" << std::endl;

	uint num = 100;
	
	CArray<float> hB(num);
	CArray<float> hC(num);

	for (int i = 0; i < num; i++)
	{
		hB[i] = 1.0;
		hC[i] = 1.0;
	}

	auto scene = std::make_shared<CsyVk::SceneGraph>();
	auto computeDemo = scene->addNode(std::make_shared<CsyVk::ComputeDemo>());
	computeDemo->loadData(hB, hC);
	CsyVk::SceneGraphFactory::instance()->pushScene(scene);
	auto activeScene = CsyVk::SceneGraphFactory::instance()->active();
	activeScene->reset();

	while (true) {
		activeScene->takeOneFrame();
		// activeScene->updateGraphicsContext();
	}

	// system("pause");

	return 0;
}
