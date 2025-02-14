#pragma comment(linker, "/subsystem:console")
#include "VkSystem.h"
#include "Array.h"
#include "VkProgram.h"
#include <optional>
#include <array>

#include "u_vk_csy.h"

using namespace CsyVk;

std::string shaderDir = "C:/temp/CG/Code/VulkanSimulation/shaders/glsl/nativeVulkanDemoCS/VecAdd.comp.spv";
std::array<float, 1024> inputData;
constexpr VkDeviceSize inputDataSize() { return sizeof(inputData); }

VkData vkData;
VkInstance instance;
VkPhysicalDevice physicalDevice;
std::optional<uint32_t> queueFamilyIndex;
VkDevice device;
VkQueue queue;
VkBuffer storageBuffer;
VkDeviceMemory storageBufferMemory;

uint32_t findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties = CsySmallVk::Query::physicalDeviceMemoryProperties(physicalDevice);
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

int main(int argc, char* argv[])
{
	for (int i = 0; i < 100; i++)
	{
		inputData[i] = float(i);
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

	//VkBufferCreateInfo createInfo1;
	//createInfo1.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	//createInfo1.size = inputDataSize();
	//createInfo1.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	//createInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	//createInfo1.queueFamilyIndexCount = 0;
	//createInfo1.pQueueFamilyIndices = nullptr;
	//if (vkCreateBuffer(device, &createInfo1, nullptr, &storageBuffer) != VK_SUCCESS)
	//	throw std::runtime_error("failed to create storage buffer!");
	//VkMemoryRequirements requirements = CsySmallVk::Query::memoryRequirements(device, storageBuffer);

	//VkMemoryAllocateInfo allocInfo = CsySmallVk::memoryAllocateInfo();
	//allocInfo.allocationSize = requirements.size;
	//allocInfo.memoryTypeIndex = findMemoryType(requirements,
	//	VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	//if (vkAllocateMemory(device, &allocInfo, nullptr, &storageBufferMemory) != VK_SUCCESS)
	//	throw std::runtime_error("failed to allocate storage buffer memory");
	//vkBindBufferMemory(device, storageBuffer, storageBufferMemory, 0);

	//void* data;
	//if (vkMapMemory(device, storageBufferMemory, 0, inputDataSize(), 0, &data) != VK_SUCCESS)
	//	throw std::runtime_error("failed to map memory");
	//memcpy(data, inputData.data(), inputDataSize());
	//vkUnmapMemory(device, storageBufferMemory);

	vkData.instance_ = instance;
	vkData.physicalDevice_ = physicalDevice;
	vkData.device_ = device;
	vkData.queue_ = queue;
	vkData.queueFamilyIndex_ = queueFamilyIndex.value();

	VkSystem::instance()->initializeWithInstance(vkData);

	//Initialize all buffers
	uint num = 100;

	DArray<float> dA(num);
	DArray<float> dB(num);
	DArray<float> dC(num);

	CArray<float> hA(num);
	CArray<float> hB(num);
	CArray<float> hC(num);

	for (int i = 0; i < num; i++)
	{
		hA[i] = float(i);
		hB[i] = float(i);
	}


	dA.assign(hA);
	dB.assign(hB);

	//Declare a kernel
	auto kernel = std::make_shared<VkProgram>(
		BUFFER(float),		//Array A
		BUFFER(float),		//Array B
		BUFFER(float),		//Array C
		CONSTANT(uint));
	kernel->load(shaderDir);

	//Execuate the kernel
	VkConstant<uint> N(num);
	kernel->flush(
		vkDispatchSize(num, 128),
		dA.handle(),
		dB.handle(),
		dC.handle(),
		&N);

	//Copy results back to the host and print out
	hC.assign(dC);
	for (int i = 0; i < num; i++)
	{
		printf("%f \n", hC[i]);
	}
	system("pause");
	return 0;
}
