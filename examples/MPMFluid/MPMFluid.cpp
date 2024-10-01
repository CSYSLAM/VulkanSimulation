/*
* Vulkan Example - Attraction based compute shader particle system
*
* Updated compute shader by Lukas Bergdoll (https://github.com/Voultapher)
*
* Copyright (C) 2016-2024 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"

#define PARTICLE_RADIUS 0.005f
#define NUM_PARTICLES 20000
#define GRID_RESOLUTION 64
#define NUM_CELLS (GRID_RESOLUTION * GRID_RESOLUTION)
#define ELASTIC_LAMBDA 10.0f
#define ELASTIC_MU 20.0f
#define DT 0.1f

class VulkanExample : public VulkanExampleBase
{
public:
	struct Particle {
		glm::vec2 pos;
		glm::vec2 vel;
		float mass;
		glm::mat2 C;
		float volume_0;
	};

	struct  Cell {
		glm::vec2 vel;
		float mass;
	};
	// We use a shader storage buffer object to store the particlces
	// This is updated by the compute pipeline and displayed as a vertex buffer by the graphics pipeline
	struct StorageBuffers {
		vks::Buffer particles;
		vks::Buffer grids;
		vks::Buffer c_mat;
	} storageBuffers;

	//vks::Buffer storageBuffer;

	// Resources for the graphics part of the example
	struct Graphics {
		uint32_t queueFamilyIndex;					// Used to check if compute and graphics queue families differ and require additional barriers
		VkDescriptorSetLayout descriptorSetLayout;	// Particle system rendering shader binding layout
		VkDescriptorSet descriptorSet;				// Particle system rendering shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the graphics pipeline
		VkPipeline pipeline;						// Particle rendering pipeline
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
	} graphics;

	// Resources for the compute part of the example
	struct Compute {
		uint32_t queueFamilyIndex;					// Used to check if compute and graphics queue families differ and require additional barriers
		VkQueue queue;								// Separate queue for compute commands (queue family may differ from the one used for graphics)
		VkCommandPool commandPool;					// Use a separate command pool (queue family may differ from the one used for graphics)
		VkCommandBuffer commandBuffer;				// Command buffer storing the dispatch commands and barriers
		VkSemaphore semaphore;                      // Execution dependency between compute & graphic submission
		VkDescriptorSetLayout descriptorSetLayout;	// Compute shader binding layout
		VkDescriptorSet descriptorSet;				// Compute shader bindings
		VkPipelineLayout pipelineLayout;			// Layout of the compute pipeline
		VkPipeline pipeline0;						// Compute pipeline for compyting density pressure
		VkPipeline pipeline1;						// Compute pipeline for compyting force
		VkPipeline pipeline2;						// Compute pipeline for integrating
		VkPipeline pipeline3;						// Compute pipeline for integrating
	} compute;

	VulkanExample() : VulkanExampleBase()
	{
		title = "Compute shader particle system";
		width = 1280;
		height = 720;
	}

	~VulkanExample()
	{
		if (device) {
			// Graphics
			vkDestroyPipeline(device, graphics.pipeline, nullptr);
			vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
			vkDestroySemaphore(device, graphics.semaphore, nullptr);

			// Compute
			vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
			vkDestroyPipeline(device, compute.pipeline0, nullptr);
			vkDestroyPipeline(device, compute.pipeline1, nullptr);
			vkDestroyPipeline(device, compute.pipeline2, nullptr);
			vkDestroySemaphore(device, compute.semaphore, nullptr);
			vkDestroyCommandPool(device, compute.commandPool, nullptr);

			storageBuffers.particles.destroy();
			storageBuffers.grids.destroy();
			storageBuffers.c_mat.destroy();
		}
	}

	void addComputeToGraphicsBarriers(VkCommandBuffer commandBuffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
	{
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.srcAccessMask = srcAccessMask;
		bufferBarrier.dstAccessMask = dstAccessMask;
		bufferBarrier.srcQueueFamilyIndex = compute.queueFamilyIndex;
		bufferBarrier.dstQueueFamilyIndex = graphics.queueFamilyIndex;
		bufferBarrier.size = VK_WHOLE_SIZE;
		std::vector<VkBufferMemoryBarrier> bufferBarriers;
		bufferBarrier.buffer = storageBuffers.particles.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.grids.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.c_mat.buffer;
		bufferBarriers.push_back(bufferBarrier);
		vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, VK_FLAGS_NONE, 0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);
	}

	void addGraphicsToComputeBarriers(VkCommandBuffer commandBuffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
	{
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.srcAccessMask = srcAccessMask;
		bufferBarrier.dstAccessMask = dstAccessMask;
		bufferBarrier.srcQueueFamilyIndex = graphics.queueFamilyIndex;
		bufferBarrier.dstQueueFamilyIndex = compute.queueFamilyIndex;
		bufferBarrier.size = VK_WHOLE_SIZE;
		std::vector<VkBufferMemoryBarrier> bufferBarriers;
		bufferBarrier.buffer = storageBuffers.particles.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.grids.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.c_mat.buffer;
		bufferBarriers.push_back(bufferBarrier);
		vkCmdPipelineBarrier(commandBuffer, srcStageMask, dstStageMask, VK_FLAGS_NONE, 0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);
	}

	void addComputeToComputeBarriers(VkCommandBuffer commandBuffer)
	{
		VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
		bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		bufferBarrier.size = VK_WHOLE_SIZE;
		std::vector<VkBufferMemoryBarrier> bufferBarriers;
		bufferBarrier.buffer = storageBuffers.particles.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.grids.buffer;
		bufferBarriers.push_back(bufferBarrier);
		bufferBarrier.buffer = storageBuffers.c_mat.buffer;
		bufferBarriers.push_back(bufferBarrier);
		vkCmdPipelineBarrier( commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_FLAGS_NONE,
			0, nullptr, static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(), 0, nullptr);
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Acquire barrier
			if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
			{
				addComputeToGraphicsBarriers(drawCmdBuffers[i], 0, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
			}

			// Draw the particle system using the update vertex buffer
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);

			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &storageBuffers.particles.buffer, offsets);
			vkCmdDraw(drawCmdBuffers[i], NUM_PARTICLES, 1, 0, 0);

			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			// Release barrier
			if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
			{
				addGraphicsToComputeBarriers(drawCmdBuffers[i], VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 0, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
			}

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void buildComputeCommandBuffer()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

		// Compute particle movement

		// Add memory barrier to ensure that the (graphics) vertex shader has fetched attributes before compute starts to write to the buffer
		addGraphicsToComputeBarriers(compute.commandBuffer, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

		// Dispatch the compute job
		// 1st pass
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline0);
		vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
		vkCmdDispatch(compute.commandBuffer, NUM_CELLS / 256, 1, 1);
		addComputeToComputeBarriers(compute.commandBuffer);

		// 2nd pass
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline1);
		vkCmdDispatch(compute.commandBuffer, 1, 1, 1);
		addComputeToComputeBarriers(compute.commandBuffer);

		// 3rd pass
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline2);
		vkCmdDispatch(compute.commandBuffer, NUM_CELLS / 256, 1, 1);
		addComputeToComputeBarriers(compute.commandBuffer);

		// 4th pass
		vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline3);
		vkCmdDispatch(compute.commandBuffer, NUM_PARTICLES / 256, 1, 1);
		addComputeToGraphicsBarriers(compute.commandBuffer, VK_ACCESS_SHADER_WRITE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

		vkEndCommandBuffer(compute.commandBuffer);
	}

	void Job_P2G(const std::vector<Particle>& particleBuffer, const std::vector<glm::mat2>& Fs, std::vector<Cell>& grid) const
	{
		std::vector<glm::vec2> weights(3);
		for (int i = 0; i < NUM_PARTICLES; ++i) {
			const Particle& p = particleBuffer[i];
			glm::mat2 stress = glm::mat2(0, 0, 0, 0);
			// deformation gradient
			const glm::mat2& F = Fs[i];
			float J = glm::determinant(F);
			// MPM course, page 46
			float volume = p.volume_0 * J;
			// useful matrices for Neo-Hookean model
			glm::mat2 F_T = glm::transpose(F);
			glm::mat2 F_inv_T = glm::inverse(F_T);
			glm::mat2 F_minus_F_inv_T = F - F_inv_T;
			// MPM course equation 48
			glm::mat2 P_term_0 = ELASTIC_MU * (F_minus_F_inv_T);
			glm::mat2 P_term_1 = ELASTIC_LAMBDA * glm::log(J) * F_inv_T;
			glm::mat2 P = P_term_0 + P_term_1;
			// cauchy_stress = (1 / det(F)) * P * F_T
			// equation 38, MPM course
			stress = (1.0f / J) * (P * F_T);
			// (M_p)^-1 = 4, see APIC paper and MPM course page 42
			// this term is used in MLS-MPM paper eq. 16. with quadratic weights, Mp = (1/4) * (delta_x)^2.
			// in this simulation, delta_x = 1, because i scale the rendering of the domain rather than the domain itself.
			// we multiply by dt as part of the process of fusing the momentum and force update for MLS-MPM
    			glm::mat2 eq_16_term_0 = -volume * 4 * stress * DT;
			// quadratic interpolation weights
			glm::vec2 cell_idx = glm::ivec2(p.pos);  // uvec2 -> unsigned
			glm::vec2 cell_diff = (p.pos - cell_idx) - 0.5f;
			weights[0] = 0.5f * ((0.5f - cell_diff) * (0.5f - cell_diff));
			weights[1] = 0.75f - (cell_diff * cell_diff);
			weights[2] = 0.5f * ((0.5f + cell_diff) * (0.5f + cell_diff));

			// for all surrounding 9 cells
			for (unsigned int gx = 0; gx < 3; ++gx) {
				for (unsigned int gy = 0; gy < 3; ++gy) {
					float weight = weights[gx].x * weights[gy].y;

					glm::vec2 cell_x = glm::ivec2(cell_idx.x + gx - 1, cell_idx.y + gy - 1);  // uint2
					glm::vec2 cell_dist = (cell_x - p.pos) + 0.5f;                               // cast uvec2 into vec2
					glm::vec2 Q = p.C * cell_dist;

					// scatter mass and momentum to the grid
					int cell_index = std::abs(cell_x.x * GRID_RESOLUTION + cell_x.y);
					Cell& cell = grid[cell_index];

					// MPM course, equation 172
					float weighted_mass = weight * p.mass;
					cell.mass += weighted_mass;

					// APIC P2G momentum contribution
					cell.vel += weighted_mass * (p.vel + Q);

					// fused force/momentum update from MLS-MPM
					// see MLS-MPM paper, equation listed after eqn. 28
					glm::vec2 momentum = (eq_16_term_0 * weight) * cell_dist;
					cell.vel += momentum;
				}
			}
		}
	}

	// Setup and fill the compute shader storage buffers containing the particles
	void prepareStorageBuffers()
	{
		// -------------------------Data Preprocess---------------------------//
		std::vector<glm::mat2> FsBuffer;
		FsBuffer.reserve(NUM_PARTICLES);
		for (auto i = 0; i < NUM_PARTICLES; i++) {
			FsBuffer.push_back(glm::mat2(1.0f));
		}

		std::vector<Particle> particleBuffer;
		particleBuffer.reserve(NUM_PARTICLES);
		for (auto i = 0, x = 0, y = 0; i < NUM_PARTICLES; i++)
		{
			Particle p;
			p.pos = glm::vec2(-0.625f + PARTICLE_RADIUS * 2 * x, PARTICLE_RADIUS * 2 * y);
			p.vel = glm::vec2(0.f);
			p.C = glm::mat2(0.f);
			p.mass = 1.0f;
			p.volume_0 = 1.0f;
			x++;
			if (x >= 125)
			{
				x = 0;
				y++;
			}
			particleBuffer.push_back(p);
		}

		std::vector<Cell> gridBuffer(NUM_CELLS);
		for (int i = 0; i < NUM_CELLS; ++i) {
			gridBuffer[i].vel = glm::vec2(0.f);
		}

		Job_P2G(particleBuffer, FsBuffer, gridBuffer);

		std::vector<glm::vec2> weights(3);
		for (int i = 0; i < NUM_PARTICLES; ++i) {
			Particle& p = particleBuffer[i];

			// quadratic interpolation weights
			glm::vec2 cell_idx = glm::ivec2(p.pos);
			glm::vec2 cell_diff = (p.pos - cell_idx) - 0.5f;
			weights[0] = 0.5f * ((0.5f - cell_diff) * (0.5f - cell_diff));
			weights[1] = 0.75f - (cell_diff * cell_diff);
			weights[2] = 0.5f * ((0.5f + cell_diff) * (0.5f + cell_diff));

			float density = 0.0f;
			// iterate over neighbouring 3x3 cells
			for (int gx = 0; gx < 3; ++gx) {
				for (int gy = 0; gy < 3; ++gy) {
					float weight = weights[gx].x * weights[gy].y;

					// map 2D to 1D index in grid
					int cell_index =std::abs( (cell_idx.x + (gx - 1)) * GRID_RESOLUTION + (cell_idx.y + gy - 1));
					density += gridBuffer[cell_index].mass * weight;
				}
			}
			// per-particle volume estimate has now been computed
			float volume = p.mass / density;
			p.volume_0 = volume;
		}

		// ------------------------Buffer Create----------------------------//

		VkDeviceSize  particleBufferSize = particleBuffer.size() * sizeof(Particle);
		VkDeviceSize  gridBufferSize = gridBuffer.size() * sizeof(Cell);
		VkDeviceSize  FsBufferSize = FsBuffer.size() * sizeof(glm::mat2);

		// Staging
		// SSBO won't be changed on the host after upload so copy to device local memory
		vks::Buffer stagingBuffer;
		// particle buffer
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			particleBufferSize,
			particleBuffer.data());

		vulkanDevice->createBuffer(
			// The SSBO will be used as a storage buffer for the compute pipeline and as a vertex buffer in the graphics pipeline
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&storageBuffers.particles,
			particleBufferSize);

		// Copy from staging buffer to storage buffer
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = particleBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.particles.buffer, 1, &copyRegion);
		// Execute a transfer barrier to the compute queue, if necessary
		if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
		{
			addGraphicsToComputeBarriers(copyCmd, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 0, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
		}
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		// ------------------------------grids buffer create-------------------
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			gridBufferSize,
			gridBuffer.data());

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&storageBuffers.grids,
			gridBufferSize);

		// Copy from staging buffer
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion = {};
		copyRegion.size = gridBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.grids.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();

		//------------------------------fs buffer create ------------------------------------
		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&stagingBuffer,
			FsBufferSize,
			FsBuffer.data());

		vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&storageBuffers.c_mat,
			FsBufferSize);

		// Copy from staging buffer
		copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		copyRegion = {};
		copyRegion.size = FsBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffers.c_mat.buffer, 1, &copyRegion);
		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
		stagingBuffer.destroy();
	}

	// The descriptor pool will be shared between graphics and compute
	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)
		};
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void prepareGraphics()
	{
		prepareStorageBuffers();

		// Pipeline layout
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(nullptr, 0);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
		VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);
		VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		// Vertex Input state
		std::vector<VkVertexInputBindingDescription> inputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(Particle), VK_VERTEX_INPUT_RATE_VERTEX)
		};
		std::vector<VkVertexInputAttributeDescription> inputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Particle, pos))
		};
		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
		vertexInputState.pVertexBindingDescriptions = inputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

		shaderStages[0] = loadShader(getShadersPath() + "MPMFluid/particle.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getShadersPath() + "MPMFluid/particle.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(graphics.pipelineLayout, renderPass, 0);
		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();
		pipelineCreateInfo.renderPass = renderPass;

		// Additive blending
		blendAttachmentState.colorWriteMask = 0xF;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
		blendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;
		blendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &graphics.pipeline));

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &graphics.semaphore));

		// Signal the semaphore
		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &graphics.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
		VK_CHECK_RESULT(vkQueueWaitIdle(queue));
	}

	void prepareCompute()
	{
		// Create a compute capable device queue
		// The VulkanDevice::createLogicalDevice functions finds a compute capable queue and prefers queue families that only support compute
		// Depending on the implementation this may result in different queue family indices for graphics and computes,
		// requiring proper synchronization (see the memory and pipeline barriers)
		vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);

		// Create compute pipeline
		// Compute pipelines are created separate from graphics pipelines even if they use the same queue (family index)

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			// Binding 0 : Particle position storage buffer
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2)
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

		VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));
		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &storageBuffers.particles.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &storageBuffers.grids.descriptor),
			vks::initializers::writeDescriptorSet(compute.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &storageBuffers.c_mat.descriptor),
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

		// Create pipeline
		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));

		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "MPMFluid/clear_grid.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline0));

		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "MPMFluid/particle_to_grid.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline1));

		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "MPMFluid/update_grid.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline2));

		computePipelineCreateInfo.stage = loadShader(getShadersPath() + "MPMFluid/grid_to_particle.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline3));

		// Separate command pool as queue family for compute may be different than graphics
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = compute.queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &compute.commandPool));

		// Create a command buffer for compute operations
		compute.commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, compute.commandPool);

		// Semaphore for compute & graphics sync
		VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &compute.semaphore));

		// Build a single command buffer containing the compute dispatch commands
		buildComputeCommandBuffer();
	}

	void draw()
	{
		// Wait for rendering finished
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

		// Submit compute commands
		VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
		computeSubmitInfo.waitSemaphoreCount = 1;
		computeSubmitInfo.pWaitSemaphores = &graphics.semaphore;
		computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &compute.semaphore;
		VK_CHECK_RESULT(vkQueueSubmit(compute.queue, 1, &computeSubmitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::prepareFrame();

		VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore graphicsWaitSemaphores[] = { compute.semaphore, semaphores.presentComplete };
		VkSemaphore graphicsSignalSemaphores[] = { graphics.semaphore, semaphores.renderComplete };

		// Submit graphics commands
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		submitInfo.waitSemaphoreCount = 2;
		submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
		submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;
		submitInfo.signalSemaphoreCount = 2;
		submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		// We will be using the queue family indices to check if graphics and compute queue families differ
		// If that's the case, we need additional barriers for acquiring and releasing resources
		graphics.queueFamilyIndex = vulkanDevice->queueFamilyIndices.graphics;
		compute.queueFamilyIndex = vulkanDevice->queueFamilyIndices.compute;
		setupDescriptorPool();
		prepareGraphics();
		prepareCompute();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
	{
		if (overlay->header("Settings")) {
		}
	}
};

VULKAN_EXAMPLE_MAIN()
