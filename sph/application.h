#pragma once
#include <vulkan/vulkan.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <chrono>
#include <cstdint>
#include <vector>
#include <atomic>

#ifndef MU_SHADER_PATH
#define MU_SHADER_PATH "../shader/"
#endif
#define SPH_NUM_PARTICLES 20000
#define SPH_PARTICLE_RADIUS 0.005f
#define SPH_WORK_GROUP_SIZE 128
// work group count is the ceiling of particle count divided by work group size
#define SPH_NUM_WORK_GROUPS ((SPH_NUM_PARTICLES + SPH_WORK_GROUP_SIZE - 1) / SPH_WORK_GROUP_SIZE)

namespace SPH
{
	class Application
	{
	public:
		Application();
		Application(const Application&) = delete;
		~Application();
		void Run();

	private:
		void InitializeWindow();
		void InitializeVulkan();

		void destroyWindow();
		void destroyVulkan();

		void CreateInstance();
		void CreateSurface();
		void SelectPhysicalDevice();
		void CreateLogicalDevice();
		void GetDeviceQueues();
		void CreateSwapchain();
		void GetSwapchainImages();
		void CreateSwapchainImageViews();
		void CreateRenderPass();
		void CreateSwapchainFrameBuffers();
		void CreatePipelineCache();
		void CreateDescriptorPool();
		void CreateBuffers();

		void CreateGraphicsPipelineLayout();
		void CreateGraphicsPipeline();
		void CreateGraphicsCommandPool();
		void CreateGraphicsCommandBuffers();
		void CreateSemaphores();
		void CreateComputeDescriptorSetLayout();
		void UpdateComputeDescriptorSets();
		void CreateComputePipelineLayout();
		void CreateComputePipelines();
		void CreateComputeCommandPool();
		void CreateComputeCommandBuffer();

		void SetInitialParticleData();
		void RunSimulation();
		void Render();
		void MainLoop();

		// helper functions
		VkShaderModule CreateShaderModule(const std::vector<char>& code);
		uint32_t findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties);
		
		GLFWwindow* window = NULL;
		uint32_t windowHeight = 1000;
		uint32_t windowWidth = 1000;

		bool paused = false;
		std::atomic_uint64_t frameNumber = 1;

		uint32_t graphicsPresentationComputeQueueFamilyIndex = UINT32_MAX;

		VkQueue presentationQueueHandle = VK_NULL_HANDLE;
		VkQueue graphicsQueueHandle = VK_NULL_HANDLE;
		VkQueue computeQueueHandle = VK_NULL_HANDLE;

		VkSurfaceCapabilitiesKHR surfaceCapabilities;
		VkSurfaceFormatKHR surfaceFormat;
		std::vector<VkImage> swapchainImageHandles;
		VkSwapchainKHR swapchainHandle;
		std::vector<VkImageView> swapchainImageViewHandles;

		VkRenderPass renderPassHandle = VK_NULL_HANDLE;
		std::vector<VkFramebuffer> swapchainFrameBufferHandles;

		VkInstance instanceHandle = VK_NULL_HANDLE;
		VkSurfaceKHR surfaceHandle = VK_NULL_HANDLE;
		VkPhysicalDevice physicalDeviceHandle = VK_NULL_HANDLE;
		VkDevice logicalDeviceHandle = VK_NULL_HANDLE;
		VkPhysicalDeviceProperties physicalDeviceProperties;
		VkPhysicalDeviceFeatures physicalDeviceFeatures;
		VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

		VkPipelineCache globalPipelineCacheHandle = VK_NULL_HANDLE;
		VkDescriptorPool globalDescriptorPoolHandle = VK_NULL_HANDLE;

		VkBuffer packedParticlesBufferHandle = VK_NULL_HANDLE;
		VkDeviceMemory packedParticlesMemoryHandle = VK_NULL_HANDLE;
		VkPipelineLayout graphicsPipelineLayoutHandle = VK_NULL_HANDLE;
		VkPipeline graphicsPipelineHandle = VK_NULL_HANDLE;
		VkCommandPool graphicsCommandPoolHandle = VK_NULL_HANDLE;

		VkCommandPool computeCommandPoolHandle = VK_NULL_HANDLE;

		std::vector<VkCommandBuffer> graphicsCommandBufferHandles;
		VkDescriptorSetLayout computeDescriptorSetLayoutHandle = VK_NULL_HANDLE;
		VkPipeline computePipelineHandles[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
		// synchronization
		VkSemaphore imageAvailableSemaphoreHandle = VK_NULL_HANDLE;
		VkSemaphore renderFinishedSemaphoreHandle = VK_NULL_HANDLE;

		VkDescriptorSet computeDescriptorSetHandle = VK_NULL_HANDLE;
		VkPipelineLayout computePipelineLayoutHandle = VK_NULL_HANDLE;
		VkCommandBuffer computeCommandBufferHandle = VK_NULL_HANDLE;

		VkSubmitInfo computeSubmitInfo
		{
			VK_STRUCTURE_TYPE_SUBMIT_INFO,
			NULL,
			0,
			NULL,
			0,
			1,
			&computeCommandBufferHandle,
			0,
			NULL
		};

		uint32_t imageIndex;
		VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo graphicsSubmitInfo
		{
			VK_STRUCTURE_TYPE_SUBMIT_INFO,
			NULL,
			1,
			&imageAvailableSemaphoreHandle,
			&wait_dst_stage_mask,
			1,
			VK_NULL_HANDLE,
			1,
			& renderFinishedSemaphoreHandle
		};

		VkPresentInfoKHR presentInfo
		{
			VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			NULL,
			1,
			& renderFinishedSemaphoreHandle,
			1,
			&swapchainHandle,
			&imageIndex,
			NULL
		};

		// ssbo sizes
		const uint64_t positionSsboSize = sizeof(glm::vec2) * SPH_NUM_PARTICLES;
		const uint64_t velocitySsboSize = sizeof(glm::vec2) * SPH_NUM_PARTICLES;
		const uint64_t forceSsboSize = sizeof(glm::vec2) * SPH_NUM_PARTICLES;
		const uint64_t densitySsboSize = sizeof(float) * SPH_NUM_PARTICLES;
		const uint64_t pressureSsboSize = sizeof(float) * SPH_NUM_PARTICLES;

		const uint64_t packedBufferSize = positionSsboSize + velocitySsboSize + forceSsboSize + densitySsboSize + pressureSsboSize;
		// ssbo offsets
		const uint64_t positionSsboOffset = 0;
		const uint64_t velocitySsboOffset = positionSsboSize;
		const uint64_t forceSsboOffset = velocitySsboOffset + velocitySsboSize;
		const uint64_t densitySsboOffset = forceSsboOffset + forceSsboSize;
		const uint64_t pressureSsboOffset = densitySsboOffset + densitySsboSize;
		
	};
}