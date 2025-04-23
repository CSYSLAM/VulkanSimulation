/*
* Vulkan Example - 3D Ball Physics Simulation
*
* Based on examples by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include <chrono>

#define NUM_BALLS 64 // 改为64个球
#define BALLS_PER_ROW 8  // 每行8个球，形成8×8网格

class VulkanExample : public VulkanExampleBase
{
public:
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point lastFrameTime;

    // Ball declaration
    struct Ball {
        glm::vec4 position;  // xyz = position, w = radius
        glm::vec4 velocity;  // xyz = velocity, w = mass
        glm::vec4 color;     // rgb = color, a = opacity
    };

    // Boundary box
    struct BoundaryBox {
        glm::vec3 min;
        glm::vec3 max;
    } boundaryBox;

    // We use a shader storage buffer object to store the balls
    vks::Buffer storageBuffer;

    // Resources for the graphics part of the example
    struct Graphics {
        uint32_t queueFamilyIndex;                // Used to check if compute and graphics queue families differ and require additional barriers
        VkDescriptorSetLayout descriptorSetLayout; // Particle system rendering shader binding layout
        VkDescriptorSet descriptorSet;            // Particle system rendering shader bindings
        VkPipelineLayout pipelineLayout;          // Layout of the graphics pipeline
        VkPipeline pipeline;                      // Particle rendering pipeline
        VkSemaphore semaphore;                    // Execution dependency between compute & graphic submission
        vks::Buffer uniformBuffer;                // Uniform buffer object containing transformation matrices
        struct UniformData {
            glm::mat4 projection;
            glm::mat4 view;
            glm::mat4 model;
        } uniformData;
    } graphics;

    // Resources for the compute part of the example
    struct Compute {
        uint32_t queueFamilyIndex;                // Used to check if compute and graphics queue families differ and require additional barriers
        VkQueue queue;                            // Separate queue for compute commands (queue family may differ from the one used for graphics)
        VkCommandPool commandPool;                // Use a separate command pool (queue family may differ from the one used for graphics)
        VkCommandBuffer commandBuffer;            // Command buffer storing the dispatch commands and barriers
        VkSemaphore semaphore;                    // Execution dependency between compute & graphic submission
        VkDescriptorSetLayout descriptorSetLayout; // Compute shader binding layout
        VkDescriptorSet descriptorSet;            // Compute shader bindings
        VkPipelineLayout pipelineLayout;          // Layout of the compute pipeline
        VkPipeline pipeline;                      // Compute pipeline for updating ball positions
        vks::Buffer uniformBuffer;                // Uniform buffer object containing simulation parameters
        struct UniformData {                      // Compute shader uniform block object
            glm::vec3 boundaryMin;
            float deltaTime;
            glm::vec3 boundaryMax;
            float gravity;
            float restitution;                    // Coefficient of restitution (bounciness)
        } uniformData;
    } compute;

    // 在构造函数中
    VulkanExample() : VulkanExampleBase()
    {
        title = "3D Ball Physics Simulation";
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-30.0f, 45.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -10.0f));  // 拉远相机

        // 设置更大的边界盒
        boundaryBox.min = glm::vec3(-3.0f, -3.0f, -3.0f);
        boundaryBox.max = glm::vec3(3.0f, 3.0f, 3.0f);
    }

    ~VulkanExample()
    {
        if (device) {
            // Graphics
            graphics.uniformBuffer.destroy();
            vkDestroyPipeline(device, graphics.pipeline, nullptr);
            vkDestroyPipelineLayout(device, graphics.pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, graphics.descriptorSetLayout, nullptr);
            vkDestroySemaphore(device, graphics.semaphore, nullptr);

            // Compute
            compute.uniformBuffer.destroy();
            vkDestroyPipelineLayout(device, compute.pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, compute.descriptorSetLayout, nullptr);
            vkDestroyPipeline(device, compute.pipeline, nullptr);
            vkDestroySemaphore(device, compute.semaphore, nullptr);
            vkDestroyCommandPool(device, compute.commandPool, nullptr);

            storageBuffer.destroy();
        }
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
                VkBufferMemoryBarrier buffer_barrier =
                {
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    0,
                    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    compute.queueFamilyIndex,
                    graphics.queueFamilyIndex,
                    storageBuffer.buffer,
                    0,
                    storageBuffer.size
                };

                vkCmdPipelineBarrier(
                    drawCmdBuffers[i],
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    0,
                    0, nullptr,
                    1, &buffer_barrier,
                    0, nullptr);
            }

            // Draw the balls
            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

            VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
            vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

            VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
            vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

            vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
            vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipelineLayout, 0, 1, &graphics.descriptorSet, 0, NULL);

            VkDeviceSize offsets[1] = { 0 };
            vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &storageBuffer.buffer, offsets);
            vkCmdDraw(drawCmdBuffers[i], NUM_BALLS, 1, 0, 0);

            drawUI(drawCmdBuffers[i]);

            vkCmdEndRenderPass(drawCmdBuffers[i]);

            // Release barrier
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
            {
                VkBufferMemoryBarrier buffer_barrier =
                {
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    nullptr,
                    VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                    0,
                    graphics.queueFamilyIndex,
                    compute.queueFamilyIndex,
                    storageBuffer.buffer,
                    0,
                    storageBuffer.size
                };

                vkCmdPipelineBarrier(
                    drawCmdBuffers[i],
                    VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0,
                    0, nullptr,
                    1, &buffer_barrier,
                    0, nullptr);
            }

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }

    void buildComputeCommandBuffer()
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VK_CHECK_RESULT(vkBeginCommandBuffer(compute.commandBuffer, &cmdBufInfo));

        // Add memory barrier to ensure that the (graphics) vertex shader has fetched attributes before compute starts to write to the buffer
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
        {
            VkBufferMemoryBarrier buffer_barrier =
            {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_SHADER_WRITE_BIT,
                graphics.queueFamilyIndex,
                compute.queueFamilyIndex,
                storageBuffer.buffer,
                0,
                storageBuffer.size
            };

            vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
        }

        // Dispatch the compute job
        vkCmdBindPipeline(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
        vkCmdBindDescriptorSets(compute.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipelineLayout, 0, 1, &compute.descriptorSet, 0, 0);
        vkCmdDispatch(compute.commandBuffer, 1, 1, 1);

        // Add barrier to ensure that compute shader has finished writing to the buffer
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
        {
            VkBufferMemoryBarrier buffer_barrier =
            {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_WRITE_BIT,
                0,
                compute.queueFamilyIndex,
                graphics.queueFamilyIndex,
                storageBuffer.buffer,
                0,
                storageBuffer.size
            };

            vkCmdPipelineBarrier(
                compute.commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
        }

        vkEndCommandBuffer(compute.commandBuffer);
    }

    // Setup and fill the compute shader storage buffers containing the balls
    void prepareStorageBuffers()
    {
        // 初始球位置和速度
        std::vector<Ball> ballBuffer(NUM_BALLS);

        // 计算网格布局
        int ballsPerRow = (int)sqrt(NUM_BALLS);
        if (ballsPerRow * ballsPerRow < NUM_BALLS) ballsPerRow++;  // 确保有足够的行
        
        // 计算间距和起始位置
        float spacing = 5.0f / ballsPerRow;  // 根据球体数量动态调整间距
        float startX = -2.5f + spacing/2;
        float startZ = -2.5f + spacing/2;
        
        // 初始化所有球体
        for (int i = 0; i < NUM_BALLS; i++) {
            Ball& ball = ballBuffer[i];
            
            // 计算行和列索引
            int row = i / ballsPerRow;
            int col = i % ballsPerRow;
            
            // 添加一些随机偏移，使球不完全对齐
            float offsetX = ((rand() % 100) / 500.0f) - 0.1f;
            float offsetZ = ((rand() % 100) / 500.0f) - 0.1f;
            
            // 位置 (x和z形成网格, y是高度)
            ball.position = glm::vec4(
                startX + col * spacing + offsetX,  // x 位置
                -2.0f + (rand() % 100) / 200.0f,    // y 位置 (高度)
                startZ + row * spacing + offsetZ,  // z 位置
                0.12f                              // 半径 (减小以适应更多球体)
            );

            // 给每个球不同的初始速度和方向
            float vx = ((rand() % 200) / 100.0f) - 1.0f;  // -1.0到1.0之间的随机速度
            float vy = 1.0f + (rand() % 100) / 100.0f;    // 1.0到2.0之间的随机向上速度
            float vz = ((rand() % 200) / 100.0f) - 1.0f;  // -1.0到1.0之间的随机速度
            
            ball.velocity = glm::vec4(vx, vy, vz, 1.0f);  // 随机初始速度，w 分量是质量
            
            // 给每个球稍微不同的质量
            ball.velocity.w = 0.8f + (rand() % 40) / 100.0f;  // 0.8到1.2之间的随机质量
            
            // 颜色（基于位置生成渐变色）
            float r = 0.5f + 0.5f * sin(row * 0.5f);
            float g = 0.5f + 0.5f * sin(col * 0.5f);
            float b = 0.5f + 0.5f * sin((row + col) * 0.3f);
            
            // 添加一些随机变化
            float colorVar = 0.7f + (rand() % 30) / 100.0f;
            ball.color = glm::vec4(r * colorVar, g * colorVar, b * colorVar, 1.0f);
        }

        VkDeviceSize storageBufferSize = ballBuffer.size() * sizeof(Ball);

        // Staging buffer
        vks::Buffer stagingBuffer;

        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &stagingBuffer,
            storageBufferSize,
            ballBuffer.data());

        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &storageBuffer,
            storageBufferSize);

        // Copy from staging buffer to storage buffer
        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = storageBufferSize;
        vkCmdCopyBuffer(copyCmd, stagingBuffer.buffer, storageBuffer.buffer, 1, &copyRegion);

        // Execute a transfer barrier to the compute queue, if necessary
        if (graphics.queueFamilyIndex != compute.queueFamilyIndex)
        {
            VkBufferMemoryBarrier buffer_barrier =
            {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
                0,
                graphics.queueFamilyIndex,
                compute.queueFamilyIndex,
                storageBuffer.buffer,
                0,
                storageBuffer.size
            };

            vkCmdPipelineBarrier(
                copyCmd,
                VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                1, &buffer_barrier,
                0, nullptr);
        }
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

        stagingBuffer.destroy();
    }

    // The descriptor pool will be shared between graphics and compute
    void setupDescriptorPool()
    {
        std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
    }

    void prepareGraphics()
    {
        prepareStorageBuffers();
        prepareUniformBuffers();

        // Descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Transformation matrices
            vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &graphics.descriptorSetLayout));

        // Descriptor set
        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &graphics.descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &graphics.descriptorSet));

        std::vector<VkWriteDescriptorSet> writeDescriptorSets;
        // Binding 0 : Transformation matrices
        writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(
            graphics.descriptorSet,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            0,
            &graphics.uniformBuffer.descriptor));

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&graphics.descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &graphics.pipelineLayout));

        // Pipeline
        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST, 0, VK_FALSE);
        VkPipelineRasterizationStateCreateInfo rasterizationState = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
        VkPipelineColorBlendAttachmentState blendAttachmentState = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
        VkPipelineColorBlendStateCreateInfo colorBlendState = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);
        VkPipelineDepthStencilStateCreateInfo depthStencilState = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
        VkPipelineViewportStateCreateInfo viewportState = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
        VkPipelineMultisampleStateCreateInfo multisampleState = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
        std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

        // Vertex Input state
        std::vector<VkVertexInputBindingDescription> inputBindings = {
            vks::initializers::vertexInputBindingDescription(0, sizeof(Ball), VK_VERTEX_INPUT_RATE_VERTEX)
        };
        std::vector<VkVertexInputAttributeDescription> inputAttributes = {
            // Location 0 : Position
            vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Ball, position)),
            // Location 1 : Velocity
            vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Ball, velocity)),
            // Location 2 : Color
            vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Ball, color))
        };
        VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
        vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(inputBindings.size());
        vertexInputState.pVertexBindingDescriptions = inputBindings.data();
        vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributes.size());
        vertexInputState.pVertexAttributeDescriptions = inputAttributes.data();

        shaderStages[0] = loadShader(getShadersPath() + "rayquery1/ball.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "rayquery1/ball.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

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
        vkGetDeviceQueue(device, compute.queueFamilyIndex, 0, &compute.queue);

        // Create compute pipeline
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
            // Binding 0 : Ball storage buffer
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0),
            // Binding 1 : Uniform buffer
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                1),
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &compute.descriptorSetLayout));

        VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &compute.descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &compute.descriptorSet));
        std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
            // Binding 0 : Ball storage buffer
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0,
                &storageBuffer.descriptor),
            // Binding 1 : Uniform buffer
            vks::initializers::writeDescriptorSet(
                compute.descriptorSet,
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                1,
                &compute.uniformBuffer.descriptor)
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

        // Create pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));
        VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
        computePipelineCreateInfo.stage = loadShader(getShadersPath() + "rayquery1/ball.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
        VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &compute.pipeline));

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

    // Prepare and initialize uniform buffer containing shader uniforms
    void prepareUniformBuffers()
    {
        // Graphics uniform buffer
        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &graphics.uniformBuffer,
            sizeof(Graphics::UniformData));
        VK_CHECK_RESULT(graphics.uniformBuffer.map());

        // Compute shader uniform buffer
        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &compute.uniformBuffer,
            sizeof(Compute::UniformData));
        VK_CHECK_RESULT(compute.uniformBuffer.map());

        // Initialize compute shader uniform data
        compute.uniformData.boundaryMin = boundaryBox.min;
        compute.uniformData.boundaryMax = boundaryBox.max;
        compute.uniformData.gravity = 0.8f;
        compute.uniformData.restitution = 0.99f;  // 80% energy conservation on bounce

        startTime = std::chrono::steady_clock::now();
        lastFrameTime = startTime;

        updateUniformBuffers();
    }

    void updateGraphicsUBO()
    {
        graphics.uniformData.projection = camera.matrices.perspective;
        graphics.uniformData.view = camera.matrices.view;
        graphics.uniformData.model = glm::mat4(1.0f);
        memcpy(graphics.uniformBuffer.mapped, &graphics.uniformData, sizeof(Graphics::UniformData));
    }

    void updateUniformBuffers()
    {
        auto now = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        // Limit delta time to avoid large jumps
        if (deltaTime > 0.05f) {
            deltaTime = 0.05f;
        }

        compute.uniformData.deltaTime = deltaTime;
        memcpy(compute.uniformBuffer.mapped, &compute.uniformData, sizeof(Compute::UniformData));

        updateGraphicsUBO();
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
        updateUniformBuffers();
    }

    virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
    {
        if (overlay->header("Settings")) {
            if (overlay->sliderFloat("Gravity", &compute.uniformData.gravity, 0.0f, 20.0f)) {
                updateUniformBuffers();
            }
            if (overlay->sliderFloat("Restitution", &compute.uniformData.restitution, 0.0f, 1.0f)) {
                updateUniformBuffers();
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()


