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
#include "CommandLineParser.hpp"
#include "vk_base_bufferUtils_csy.h"

#include "vkcsy.h"
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <chrono>
#include <cstdint>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>

#define NUM_PARTICLES 100000

struct Vertex {
	glm::vec2 pos;
	glm::vec2 vel;
};

struct PushConstants {
	glm::vec2 attractor;
	float attractor_strength;
	float delta_time;
};

#define WORK_GROUP_SIZE 128
#define NUM_WORK_GROUPS ((NUM_PARTICLES + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE)

#define LOG(...) printf(__VA_ARGS__)

std::string shaderDir = "glsl/simpleParticles/";
const std::string shadersPath = getShaderBasePath() + shaderDir;

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
	uint32_t windowWidth_ = 800;
	uint32_t windowHeight_ = 600;
	bool paused_ = false;
	std::atomic_uint64_t frameNumber_ = 1;
	std::chrono::steady_clock::time_point startTime_;
	std::chrono::steady_clock::time_point lastFrameTime_;

	VkSurfaceKHR surface_;
	VkPhysicalDeviceFeatures physicalDeviceFeatures_;
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties_;
	InstanceCsy* instance;
	DeviceCsy* device;
	SwapChainCsy* swapChain;
	VkSwapchainKHR swapchain_;
	std::vector<VkImageView> swapchainImageView_;
	VkRenderPass renderPass_;
	std::vector<VkFramebuffer> swapchainFrameBuffer_;
	VkPipelineCache globalPipelineCache_;
	VkDescriptorPool globalDescriptorPool_;
	VkBuffer particlesBuffer_;
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
	VkPipeline computePipeline_;
	VkCommandPool computeCommandPool_;
	VkCommandBuffer computeCommandBuffer_;
	uint32_t imageIndex_;
	VkPipelineStageFlags wait_dst_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

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

	const uint64_t bufferSize_ = NUM_PARTICLES * sizeof(Vertex);
	const uint64_t bufferOffset_ = 0;

	// Assuming these are defined elsewhere in your C++ code

	VkDebugReportCallbackEXT debugReportCallback{};

	void CreateSwapchainImageViews()
	{
		std::vector<VkImage>swapchainImage = swapChain->GetVkImages();
		swapchainImageView_.resize(swapchainImage.size());
		for (uint32_t i = 0; i < swapchainImageView_.size(); i++)
		{
			VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
			view.image = VK_NULL_HANDLE;
			view.viewType = VK_IMAGE_VIEW_TYPE_2D;
			view.format = swapChain->GetSurfaceFormat().format;
			view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			view.image = swapchainImage[i];
			VK_CHECK_RESULT(vkCreateImageView(device->GetVkDevice(), &view, NULL, &swapchainImageView_[i]));
		}
		std::cout << "Successfully create swapchain ImageViews" << std::endl;
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = swapChain->GetSurfaceFormat().format;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;	// Clear depth at beginning of the render pass
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	// We will read from depth, so it's important to store the depth attachment results
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;					// We don't care about initial layout of the attachment
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentReference = {};
		colorAttachmentReference.attachment = 0;
		colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.pColorAttachments = &colorAttachmentReference;
		subpassDescription.colorAttachmentCount = 1;

		VkRenderPassCreateInfo renderPassCreateInfo = vks::initializers::renderPassCreateInfo();
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpassDescription;
		VK_CHECK_RESULT(vkCreateRenderPass(device->GetVkDevice(), &renderPassCreateInfo, nullptr, &renderPass_));
		std::cout << "Successfully create render pass" << std::endl;
	}

	void CreateSwapchainFrameBuffers()
	{
		swapchainFrameBuffer_.resize(swapchainImageView_.size());
		for (size_t index = 0; index < swapchainImageView_.size(); index++)
		{
			VkFramebufferCreateInfo fbufCreateInfo = vks::initializers::framebufferCreateInfo();
			fbufCreateInfo.renderPass = renderPass_;
			fbufCreateInfo.attachmentCount = 1;
			fbufCreateInfo.pAttachments = &swapchainImageView_[index];
			fbufCreateInfo.width = windowWidth_;
			fbufCreateInfo.height = windowHeight_;
			fbufCreateInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device->GetVkDevice(), &fbufCreateInfo, nullptr, &swapchainFrameBuffer_[index]));
		}
		std::cout << "Successfully create swapchain framebuffers" << std::endl;
	}

	void CreatePipelineCache()
	{
		VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(device->GetVkDevice(), &pipelineCacheCreateInfo, NULL, &globalPipelineCache_));
		std::cout << "Successfully create pipelineCache" << std::endl;
	}

	void CreateDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device->GetVkDevice(), &descriptorPoolInfo, NULL, &globalDescriptorPool_));
		std::cout << "Successfully create descriptor pool" << std::endl;
	}

	uint32_t findMemoryType(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties = CsySmallVk::Query::physicalDeviceMemoryProperties(instance->GetPhysicalDevice());
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
		std::vector<Vertex> initialParticlePosition;
		initialParticlePosition.reserve(NUM_PARTICLES);

		for (int i = 0; i < NUM_PARTICLES; ++i) {
			float f = static_cast<float>(i) / (NUM_PARTICLES / 10);
			Vertex vertex;
			vertex.pos = glm::vec2(2.0f * glm::fract(f) - 1.0f, 0.2f * glm::floor(f) - 1.0f);
			vertex.vel = glm::vec2(0.0f);
			initialParticlePosition.push_back(vertex);
		}

		BufferUtilsCsy::CreateBufferFromData(device, computeCommandPool_, initialParticlePosition.data(), NUM_PARTICLES * sizeof(Vertex),
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
			particlesBuffer_, packedParticlesMemory_);
	}

	void CreateGraphicsPipelineLayout()
	{
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(nullptr, 0);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device->GetVkDevice(), &pipelineLayoutCreateInfo, NULL, &graphicsPipelineLayout_));
		std::cout << "Successfully create graphics pipeline layout" << std::endl;
	}

	void CreateGraphicsPipeline()
	{
		std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

		VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		vertexShaderStageCreateInfo.module = vks::tools::loadShader((shadersPath + "particle.vert.spv").c_str(), device->GetVkDevice());
		vertexShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertexShaderStageCreateInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		fragmentShaderStageCreateInfo.module = vks::tools::loadShader((shadersPath + "particle.frag.spv").c_str(), device->GetVkDevice());
		fragmentShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragmentShaderStageCreateInfo.pName = "main";

		shaderStageCreateInfos.push_back(vertexShaderStageCreateInfo);
		shaderStageCreateInfos.push_back(fragmentShaderStageCreateInfo);


		VkVertexInputBindingDescription vertexInputBindingDescription =
			vks::initializers::vertexInputBindingDescription(0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX);

		//layout(location = 0) in vec2 pos;
		//layout(location = 1) in vec2 vel;
		std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributeDescriptions = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)),
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, vel))
		};

		VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
		vertexInputStateCreateInfo.pVertexBindingDescriptions = &vertexInputBindingDescription;
		vertexInputStateCreateInfo.vertexAttributeDescriptionCount = 2;
		vertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexInputAttributeDescriptions.data();;

		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo =
				vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);

		VkViewport viewport = vks::initializers::viewport(static_cast<float>(windowWidth_), static_cast<float>(windowHeight_), 0.0f, 1.0f);
		VkRect2D scissor = vks::initializers::rect2D(windowWidth_, windowHeight_, 0, 0);

		VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {};
		viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportStateCreateInfo.viewportCount = 1;
		viewportStateCreateInfo.pViewports = &viewport;
		viewportStateCreateInfo.scissorCount = 1;
		viewportStateCreateInfo.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo =
			vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo =
				vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		VkPipelineColorBlendAttachmentState colorBlendAttachment = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo = vks::initializers::pipelineColorBlendStateCreateInfo(1, &colorBlendAttachment);
		
		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(graphicsPipelineLayout_, renderPass_);
		pipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
		pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
		pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
		pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
		pipelineCreateInfo.pViewportState = nullptr;
		pipelineCreateInfo.pDepthStencilState = nullptr;
		pipelineCreateInfo.pDynamicState = nullptr;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStageCreateInfos.size());
		pipelineCreateInfo.pStages = shaderStageCreateInfos.data();
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device->GetVkDevice(), globalPipelineCache_, 1, &pipelineCreateInfo, NULL, &graphicsPipeline_));
		std::cout << "Successfully create graphics pipeline" << std::endl;
	}

	void CreateGraphicsCommandPool()
	{
		VkCommandPoolCreateInfo graphicsCommandPoolCreateInfo = CsySmallVk::commandPoolCreateInfo();
		graphicsCommandPoolCreateInfo.queueFamilyIndex = device->GetQueueIndex(QueueFlags::Graphics);
		VK_CHECK_RESULT(vkCreateCommandPool(device->GetVkDevice(), &graphicsCommandPoolCreateInfo, NULL, &graphicsCommandPool_));
		std::cout << "Successfully create graphics command pool" << std::endl;
	}

	void CreateGraphicsCommandBuffers()
	{
		graphicsCommandBuffer_.resize(swapchainFrameBuffer_.size());
		VkCommandBufferAllocateInfo graphicsCommandBufferAllocationInfo =
				vks::initializers::commandBufferAllocateInfo(graphicsCommandPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, static_cast<uint32_t>(graphicsCommandBuffer_.size()));
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device->GetVkDevice(), &graphicsCommandBufferAllocationInfo, graphicsCommandBuffer_.data()));
		
		VkClearValue clearValues{ 0.0f, 0.0f, 0.0f, 1.0f };
		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass_;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = windowWidth_;
		renderPassBeginInfo.renderArea.extent.height = windowHeight_;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = &clearValues;

		for (size_t i = 0; i < graphicsCommandBuffer_.size(); i++)
		{
			VkCommandBufferBeginInfo commandBufferBeginInfo = vks::initializers::commandBufferBeginInfo();
			vkBeginCommandBuffer(graphicsCommandBuffer_[i], &commandBufferBeginInfo);

			renderPassBeginInfo.framebuffer = swapchainFrameBuffer_[i];
			vkCmdBeginRenderPass(graphicsCommandBuffer_[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)windowWidth_, (float)windowHeight_, 0.0f, 1.0f);
			vkCmdSetViewport(graphicsCommandBuffer_[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(windowWidth_, windowHeight_, 0, 0);
			vkCmdSetScissor(graphicsCommandBuffer_[i], 0, 1, &scissor);

			vkCmdBindPipeline(graphicsCommandBuffer_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);

			VkDeviceSize offsets = 0;
			vkCmdBindVertexBuffers(graphicsCommandBuffer_[i], 0, 1, &particlesBuffer_, &offsets);
			vkCmdDraw(graphicsCommandBuffer_[i], NUM_PARTICLES, 1, 0, 0);
			vkCmdEndRenderPass(graphicsCommandBuffer_[i]);
			VK_CHECK_RESULT(vkEndCommandBuffer(graphicsCommandBuffer_[i]));
		}
		std::cout << "Successfully create graphics command buffers" << std::endl;
	}

	void CreateSemaphores()
	{
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device->GetVkDevice(), &semaphoreCreateInfo, NULL, &imageAvailableSemaphore_));
		VK_CHECK_RESULT(vkCreateSemaphore(device->GetVkDevice(), &semaphoreCreateInfo, NULL, &renderFinishedSemaphore_));
		std::cout << "Successfully create semaphores" << std::endl;
	}

	void CreateComputeDescriptorSetLayout()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0)
		};
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->GetVkDevice(), &descriptorSetLayoutCreateInfo, NULL, &computeDescriptorSetLayout_));
		std::cout << "Successfully create compute descriptorSet layout" << std::endl;
	}

	void UpdateComputeDescriptorSets()
	{
		// allocate descriptor sets
		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(globalDescriptorPool_, &computeDescriptorSetLayout_, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device->GetVkDevice(), &allocInfo, &computeDescriptorSet_));

		VkDescriptorBufferInfo descriptorBufferInfos;
		descriptorBufferInfos.buffer = particlesBuffer_;
		descriptorBufferInfos.offset = bufferOffset_;
		descriptorBufferInfos.range = bufferSize_;

		// write descriptor sets
		VkWriteDescriptorSet writeDescriptorSets =
			vks::initializers::writeDescriptorSet(computeDescriptorSet_, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &descriptorBufferInfos);
		vkUpdateDescriptorSets(device->GetVkDevice(), 1, &writeDescriptorSets, 0, NULL);
		std::cout << "Successfully update compute descriptorsets" << std::endl;
	}

	void CreateComputePipelineLayout()
	{
		VkPipelineLayoutCreateInfo layoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&computeDescriptorSetLayout_, 1);

		// Push constants used to pass some parameters
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(PushConstants), 0);
		layoutCreateInfo.pushConstantRangeCount = 1;
		layoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device->GetVkDevice(), &layoutCreateInfo, nullptr, &computePipelineLayout_));
		std::cout << "Successfully create compute pipeline layout" << std::endl;
	}

	void CreateComputePipelines()
	{
		VkPipelineShaderStageCreateInfo shaderStageCreateInfo = CsySmallVk::pipelineShaderStageCreateInfo();
		shaderStageCreateInfo.module = vks::tools::loadShader((shadersPath + "particle.comp.spv").c_str(), device->GetVkDevice());
		shaderStageCreateInfo.pName = "main";

		VkComputePipelineCreateInfo createInfo = vks::initializers::computePipelineCreateInfo(computePipelineLayout_, 0);
		createInfo.stage = shaderStageCreateInfo;
		VK_CHECK_RESULT(vkCreateComputePipelines(device->GetVkDevice(), globalPipelineCache_, 1, &createInfo, NULL, &computePipeline_));
		std::cout << "Successfully create compute pipelines" << std::endl;
	}

	void CreateComputeCommandPool()
	{
		VkCommandPoolCreateInfo createInfo = CsySmallVk::commandPoolCreateInfo();
		createInfo.queueFamilyIndex = device->GetQueueIndex(QueueFlags::Compute);
		createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device->GetVkDevice(), &createInfo, NULL, &computeCommandPool_));
		std::cout << "Successfully create compute command pool" << std::endl;
	}

	void RecordComputeCommandBuffer(const PushConstants& push_constants)
	{
		// 重置命令缓冲区以重新记录
		VK_CHECK_RESULT(vkResetCommandBuffer(computeCommandBuffer_, 0));
		// 开始记录命令缓冲区
		VkCommandBufferBeginInfo beginInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(computeCommandBuffer_, &beginInfo));
		// 绑定计算管线和描述符集
		vkCmdBindPipeline(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
		vkCmdBindDescriptorSets(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout_, 0, 1, &computeDescriptorSet_, 0, nullptr);
		// 设置推送常量
		vkCmdPushConstants(computeCommandBuffer_, computePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &push_constants);
		// 调度计算着色器
		vkCmdDispatch(computeCommandBuffer_, NUM_WORK_GROUPS, 1, 1);
		// 添加管线屏障以确保计算着色器完成
		VkMemoryBarrier memoryBarrier = {};
		memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
		vkCmdPipelineBarrier(computeCommandBuffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
			1, &memoryBarrier, 0, nullptr, 0, nullptr);

		// 结束记录命令缓冲区
		VK_CHECK_RESULT(vkEndCommandBuffer(computeCommandBuffer_));
	}

	void RunSimulation()
	{
		// 更新每帧变量
		auto now = std::chrono::steady_clock::now();
		float time = std::chrono::duration<float>(now - startTime_).count();
		float delta_time = std::chrono::duration<float>(now - lastFrameTime_).count();
		lastFrameTime_ = now;

		PushConstants push_constants;
		push_constants.attractor = glm::vec2(0.75f * std::cos(3.0f * time), 0.6f * std::sin(0.75f * time));
		push_constants.attractor_strength = 1.2f * std::cos(2.0f * time);
		push_constants.delta_time = delta_time;

		// 记录计算命令缓冲区
		RecordComputeCommandBuffer(push_constants);

		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &computeCommandBuffer_;
		VK_CHECK_RESULT(vkQueueSubmit(device->GetQueue(QueueFlags::Compute), 1, &submitInfo, VK_NULL_HANDLE));
	}


	void Render()
	{
		// submit graphics command buffer
		vkAcquireNextImageKHR(device->GetVkDevice(), swapchain_, UINT64_MAX, imageAvailableSemaphore_, VK_NULL_HANDLE, &imageIndex_);
		graphicsSubmitInfo_.pCommandBuffers = graphicsCommandBuffer_.data() + imageIndex_;
		VK_CHECK_RESULT(vkQueueSubmit(device->GetQueue(QueueFlags::Graphics), 1, &graphicsSubmitInfo_, VK_NULL_HANDLE));
		// queue the image for presentation
		vkQueuePresentKHR(device->GetQueue(QueueFlags::Present), &presentInfo_);

		vkQueueWaitIdle(device->GetQueue(QueueFlags::Present));
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
		glfwSetWindowTitle(GetGLFWWindow(), title.str().c_str());
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
		startTime_ = std::chrono::steady_clock::now();
		lastFrameTime_ = startTime_;
		while (!ShouldQuit())
		{
			MainLoop();
		}
	}

	void destroyVulkan()
	{
		vkDestroySwapchainKHR(device->GetVkDevice(), swapchain_, NULL);
		vkDestroySurfaceKHR(instance->GetVkInstance(), surface_, NULL);
		vkDestroyDevice(device->GetVkDevice(), NULL);
		vkDestroyInstance(instance->GetVkInstance(), NULL);
	}

	VulkanExample()
	{
		LOG("Running headless compute example\n");
		static constexpr char* applicationName = "Vulkan Example";
		InitializeWindow((int)windowWidth_, (int)windowHeight_, applicationName);

		unsigned int glfwExtensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		instance = new InstanceCsy(applicationName, glfwExtensionCount, glfwExtensions);

		// CreateSurface
		VK_CHECK_RESULT(glfwCreateWindowSurface(instance->GetVkInstance(), GetGLFWWindow(), NULL, &surface_));
		instance->PickPhysicalDevice({ VK_KHR_SWAPCHAIN_EXTENSION_NAME },
			QueueFlagBit::GraphicsBit | QueueFlagBit::TransferBit | QueueFlagBit::ComputeBit | QueueFlagBit::PresentBit, surface_);

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.tessellationShader = VK_TRUE;
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.fillModeNonSolid = VK_TRUE;
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		device = instance->CreateDevice(QueueFlagBit::GraphicsBit | QueueFlagBit::TransferBit | QueueFlagBit::ComputeBit | QueueFlagBit::PresentBit, deviceFeatures);

		swapChain = device->CreateSwapChain(surface_, 1);
		swapchain_ = swapChain->GetVkSwapChain();

		CreateSwapchainImageViews();
		CreateRenderPass();
		CreateSwapchainFrameBuffers();
		CreatePipelineCache();
		CreateDescriptorPool();
		CreateComputeCommandPool();
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

		VkCommandBufferAllocateInfo allocInfo =
			vks::initializers::commandBufferAllocateInfo(computeCommandPool_, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device->GetVkDevice(), &allocInfo, &computeCommandBuffer_));
		std::cout << "Successfully allocated compute command buffer" << std::endl;
	}

	~VulkanExample()
	{
		destroyVulkan();
		DestroyWindow();
	}
};

int main(int argc, char* argv[]) {
	VulkanExample* vulkanExample = new VulkanExample();
	vulkanExample->Run();
	delete(vulkanExample);
	return 0;
}