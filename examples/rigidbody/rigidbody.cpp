/*
* Vulkan Example - 3D Ball Physics Simulation
*
* Based on examples by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanexamplebase.h"
#include <chrono>
#include <iostream>

#define NUM_BALLS 128
#define BALLS_PER_ROW 12

class VulkanExample : public VulkanExampleBase
{
public:
    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point lastFrameTime;

    // Ball declaration
    struct Ball {
        glm::vec4 position;  // xyz = position, w = radius
        glm::vec4 velocity;  // xyz = velocity, w = mass
        glm::vec4 color;     // rgb = color, a = opacity
    };

    struct AABB {
        glm::vec3 min;
        glm::vec3 max;
    };
    vks::Buffer aabbsBuffer;
    uint32_t aabbCount{ 0 };

    vks::Buffer instanceBuffer;
    std::vector<Ball> ballBuffer;

    struct AccelerationStructure {
        VkAccelerationStructureKHR handle;
        uint64_t deviceAddress = 0;
        VkDeviceMemory memory;
        VkBuffer buffer;
    };

    struct ScratchBuffer
    {
        uint64_t deviceAddress = 0;
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    AccelerationStructure bottomLevelAS;
    AccelerationStructure topLevelAS;

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

    VulkanExample() : VulkanExampleBase()
    {
        title = "3D Ball Physics Simulation";
        camera.type = Camera::CameraType::lookat;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setRotation(glm::vec3(-30.0f, 45.0f, 0.0f));
        camera.setTranslation(glm::vec3(0.0f, 0.0f, -10.0f));

        boundaryBox.min = glm::vec3(-3.0f, -3.0f, -3.0f);
        boundaryBox.max = glm::vec3(3.0f, 3.0f, 3.0f);

        enabledDeviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
        enabledDeviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
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
            aabbsBuffer.destroy();
            instanceBuffer.destroy();
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

        // 插入屏障：确保计算着色器写入完成
        VkMemoryBarrier memoryBarrier = vks::initializers::memoryBarrier();
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(
            compute.commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

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
        ballBuffer.resize(NUM_BALLS);
        int ballsPerRow = (int)sqrt(NUM_BALLS);
        if (ballsPerRow * ballsPerRow < NUM_BALLS) ballsPerRow++;

        float spacing = 5.0f / ballsPerRow;
        float startX = -2.5f + spacing / 2;
        float startZ = -2.5f + spacing / 2;

        std::vector<AABB> aabbs{};

        for (int i = 0; i < NUM_BALLS; i++) {
            Ball& ball = ballBuffer[i];

            int row = i / ballsPerRow;
            int col = i % ballsPerRow;

            float offsetX = ((rand() % 100) / 500.0f) - 0.1f;
            float offsetZ = ((rand() % 100) / 500.0f) - 0.1f;

            ball.position = glm::vec4(
                startX + col * spacing + offsetX,
                -2.0f + (rand() % 500) / 100.0f,
                startZ + row * spacing + offsetZ,
                0.5f
            );

            float vx = ((rand() % 400) / 100.0f) - 2.0f;
            float vy = 2.0f + (rand() % 200) / 100.0f;
            float vz = ((rand() % 400) / 100.0f) - 2.0f;

            ball.velocity = glm::vec4(vx, vy, vz, 1.0f);
            ball.velocity.w = 0.8f + (rand() % 40) / 100.0f;

            float r = 0.5f + 0.5f * sin(row * 0.5f);
            float g = 0.5f + 0.5f * sin(col * 0.5f);
            float b = 0.5f + 0.5f * sin((row + col) * 0.3f);

            float colorVar = 0.7f + (rand() % 30) / 100.0f;
            ball.color = glm::vec4(r * colorVar, g * colorVar, b * colorVar, 1.0f);

            aabbs.push_back({
                glm::vec3(-1.0f), // min (局部坐标)
                glm::vec3(1.0f)   // max (局部坐标)
                });
        }

        aabbCount = static_cast<uint32_t>(aabbs.size());
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
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
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

        // AABBs
        vks::Buffer stagingBufferAABB{};
        VkBufferUsageFlags usageFlags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBufferAABB, sizeof(AABB) * aabbs.size(), aabbs.data()));
        VK_CHECK_RESULT(vulkanDevice->createBuffer(usageFlags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &aabbsBuffer, sizeof(AABB) * aabbs.size()));
        vulkanDevice->copyBuffer(&stagingBufferAABB, &aabbsBuffer, queue);
        stagingBufferAABB.destroy();
    }

    void createAccelerationStructure(AccelerationStructure& accelerationStructure, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
    {
        // Buffer and memory
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = buildSizeInfo.accelerationStructureSize;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VK_CHECK_RESULT(vkCreateBuffer(vulkanDevice->logicalDevice, &bufferCreateInfo, nullptr, &accelerationStructure.buffer));
        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(vulkanDevice->logicalDevice, accelerationStructure.buffer, &memoryRequirements);
        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        VkMemoryAllocateInfo memoryAllocateInfo{};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice->logicalDevice, &memoryAllocateInfo, nullptr, &accelerationStructure.memory));
        VK_CHECK_RESULT(vkBindBufferMemory(vulkanDevice->logicalDevice, accelerationStructure.buffer, accelerationStructure.memory, 0));
        // Acceleration structure
        VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
        accelerationStructureCreate_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        accelerationStructureCreate_info.buffer = accelerationStructure.buffer;
        accelerationStructureCreate_info.size = buildSizeInfo.accelerationStructureSize;
        accelerationStructureCreate_info.type = type;
        vkCreateAccelerationStructureKHR(vulkanDevice->logicalDevice, &accelerationStructureCreate_info, nullptr, &accelerationStructure.handle);
        // AS device address
        VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
        accelerationDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        accelerationDeviceAddressInfo.accelerationStructure = accelerationStructure.handle;
        accelerationStructure.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(vulkanDevice->logicalDevice, &accelerationDeviceAddressInfo);
    }

    uint64_t getBufferDeviceAddress(VkBuffer buffer)
    {
        VkBufferDeviceAddressInfoKHR bufferDeviceAI{};
        bufferDeviceAI.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAI.buffer = buffer;
        return vkGetBufferDeviceAddressKHR(vulkanDevice->logicalDevice, &bufferDeviceAI);
    }

    void loadRayTracingFunctions() {
        vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
        vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
        vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
        vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
        vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
        vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkBuildAccelerationStructuresKHR"));
        vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
        vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
        vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
        vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
    }

    ScratchBuffer createScratchBuffer(VkDeviceSize size)
    {
        ScratchBuffer scratchBuffer{};
        // Buffer and memory
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = size;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VK_CHECK_RESULT(vkCreateBuffer(vulkanDevice->logicalDevice, &bufferCreateInfo, nullptr, &scratchBuffer.handle));
        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(vulkanDevice->logicalDevice, scratchBuffer.handle, &memoryRequirements);
        VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
        memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
        VkMemoryAllocateInfo memoryAllocateInfo = {};
        memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
        memoryAllocateInfo.allocationSize = memoryRequirements.size;
        memoryAllocateInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice->logicalDevice, &memoryAllocateInfo, nullptr, &scratchBuffer.memory));
        VK_CHECK_RESULT(vkBindBufferMemory(vulkanDevice->logicalDevice, scratchBuffer.handle, scratchBuffer.memory, 0));
        // Buffer device address
        VkBufferDeviceAddressInfoKHR bufferDeviceAddresInfo{};
        bufferDeviceAddresInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        bufferDeviceAddresInfo.buffer = scratchBuffer.handle;
        scratchBuffer.deviceAddress = vkGetBufferDeviceAddressKHR(vulkanDevice->logicalDevice, &bufferDeviceAddresInfo);
        return scratchBuffer;
    }

    void createBottomLevelAccelerationStructure()
    {
        // Build
        VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
        accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        // Instead of providing actual geometry (e.g. triangles), we only provide the axis aligned bounding boxes (AABBs) of the spheres
        // The data for the actual spheres is passed elsewhere as a shader storage buffer object
        accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
        accelerationStructureGeometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        accelerationStructureGeometry.geometry.aabbs.data.deviceAddress = getBufferDeviceAddress(aabbsBuffer.buffer);
        accelerationStructureGeometry.geometry.aabbs.stride = sizeof(AABB);

        // Get size info
        VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
        accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        accelerationStructureBuildGeometryInfo.geometryCount = 1;
        accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

        VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &accelerationStructureBuildGeometryInfo,
            &aabbCount,
            &accelerationStructureBuildSizesInfo);

        createAccelerationStructure(bottomLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, accelerationStructureBuildSizesInfo);

        // Create a small scratch buffer used during build of the bottom level acceleration structure
        ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

        VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
        accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        accelerationBuildGeometryInfo.dstAccelerationStructure = bottomLevelAS.handle;
        accelerationBuildGeometryInfo.geometryCount = 1;
        accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
        accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo.primitiveCount = aabbCount;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

        // Build the acceleration structure on the device via a one-time command buffer submission
        // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
        VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        vkCmdBuildAccelerationStructuresKHR(
            commandBuffer,
            1,
            &accelerationBuildGeometryInfo,
            accelerationBuildStructureRangeInfos.data());
        vulkanDevice->flushCommandBuffer(commandBuffer, queue);

        if (scratchBuffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(vulkanDevice->logicalDevice, scratchBuffer.memory, nullptr);
        }
        if (scratchBuffer.handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkanDevice->logicalDevice, scratchBuffer.handle, nullptr);
        }
    }

    void createTopLevelAccelerationStructure()
    {
        std::vector<VkAccelerationStructureInstanceKHR> instancesData;
        instancesData.resize(NUM_BALLS);
        for (uint32_t i = 0; i < NUM_BALLS; ++i) {
            Ball& ball = ballBuffer[i];
            VkTransformMatrixKHR transform = {
                ball.position.w, 0.0f, 0.0f, ball.position.x,
                0.0f, ball.position.w, 0.0f, ball.position.y,
                0.0f, 0.0f, ball.position.w, ball.position.z
            };
            instancesData[i].transform = transform;
            instancesData[i].instanceCustomIndex = i;
            instancesData[i].mask = 0xFF;
            instancesData[i].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instancesData[i].accelerationStructureReference = bottomLevelAS.deviceAddress;
        }

        vks::Buffer instancesStagingBuffer;
        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &instancesStagingBuffer,
            sizeof(VkAccelerationStructureInstanceKHR) * NUM_BALLS,
            nullptr
        );

        void* data;
        vkMapMemory(device, instancesStagingBuffer.memory, 0, instancesStagingBuffer.size, 0, &data);
        memcpy(data, instancesData.data(), instancesStagingBuffer.size);
        vkUnmapMemory(device, instancesStagingBuffer.memory);

        vulkanDevice->createBuffer(
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &instanceBuffer,
            sizeof(VkAccelerationStructureInstanceKHR) * NUM_BALLS,
            nullptr
        );

        VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        VkBufferCopy copyRegion = {};
        copyRegion.size = sizeof(VkAccelerationStructureInstanceKHR) * NUM_BALLS;
        vkCmdCopyBuffer(copyCmd, instancesStagingBuffer.buffer, instanceBuffer.buffer, 1, &copyRegion);
        vulkanDevice->flushCommandBuffer(copyCmd, queue, true);
        instancesStagingBuffer.destroy();

        VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
        instanceDataDeviceAddress.deviceAddress = getBufferDeviceAddress(instanceBuffer.buffer);

        VkAccelerationStructureGeometryKHR accelerationStructureGeometry = vks::initializers::accelerationStructureGeometryKHR();
        accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
        accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;

        // Get size info
        VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
        accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        accelerationStructureBuildGeometryInfo.geometryCount = 1;
        accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;

        uint32_t primitive_count = NUM_BALLS;

        VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo = vks::initializers::accelerationStructureBuildSizesInfoKHR();
        vkGetAccelerationStructureBuildSizesKHR(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &accelerationStructureBuildGeometryInfo,
            &primitive_count,
            &accelerationStructureBuildSizesInfo);

        createAccelerationStructure(topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

        // Create a small scratch buffer used during build of the top level acceleration structure
        ScratchBuffer scratchBuffer = createScratchBuffer(accelerationStructureBuildSizesInfo.buildScratchSize);

        VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo = vks::initializers::accelerationStructureBuildGeometryInfoKHR();
        accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
        accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        accelerationBuildGeometryInfo.dstAccelerationStructure = topLevelAS.handle;
        accelerationBuildGeometryInfo.geometryCount = 1;
        accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
        accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

        VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
        accelerationStructureBuildRangeInfo.primitiveCount = primitive_count;
        accelerationStructureBuildRangeInfo.primitiveOffset = 0;
        accelerationStructureBuildRangeInfo.firstVertex = 0;
        accelerationStructureBuildRangeInfo.transformOffset = 0;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

        // Build the acceleration structure on the device via a one-time command buffer submission
        // Some implementations may support acceleration structure building on the host (VkPhysicalDeviceAccelerationStructureFeaturesKHR->accelerationStructureHostCommands), but we prefer device builds
        VkCommandBuffer commandBuffer = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
        vkCmdBuildAccelerationStructuresKHR(
            commandBuffer,
            1,
            &accelerationBuildGeometryInfo,
            accelerationBuildStructureRangeInfos.data());
        vulkanDevice->flushCommandBuffer(commandBuffer, queue);

        if (scratchBuffer.memory != VK_NULL_HANDLE) {
            vkFreeMemory(vulkanDevice->logicalDevice, scratchBuffer.memory, nullptr);
        }
        if (scratchBuffer.handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkanDevice->logicalDevice, scratchBuffer.handle, nullptr);
        }
    }

    void deleteAccelerationStructure(AccelerationStructure& accelerationStructure)
    {
        vkFreeMemory(device, accelerationStructure.memory, nullptr);
        vkDestroyBuffer(device, accelerationStructure.buffer, nullptr);
        vkDestroyAccelerationStructureKHR(device, accelerationStructure.handle, nullptr);
    }

    // The descriptor pool will be shared between graphics and compute
    void setupDescriptorPool()
    {
        std::vector<VkDescriptorPoolSize> poolSizes = {
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3),
            vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1)
        };
        VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 3);
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

        shaderStages[0] = loadShader(getShadersPath() + "rigidbody/rigidbody.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
        shaderStages[1] = loadShader(getShadersPath() + "rigidbody/rigidbody.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);

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
            // Binding 2 : Acceleration structure
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                VK_SHADER_STAGE_COMPUTE_BIT,
                2),
            // Binding 3 : Instance buffer (for transform updates)
            vks::initializers::descriptorSetLayoutBinding(
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                VK_SHADER_STAGE_COMPUTE_BIT,
                3),
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
                &compute.uniformBuffer.descriptor),
        };
        // Add acceleration structure descriptor
        VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo = vks::initializers::writeDescriptorSetAccelerationStructureKHR();
        descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
        descriptorAccelerationStructureInfo.pAccelerationStructures = &topLevelAS.handle;

        VkWriteDescriptorSet accelerationStructureWrite{};
        accelerationStructureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        accelerationStructureWrite.pNext = &descriptorAccelerationStructureInfo;
        accelerationStructureWrite.dstSet = compute.descriptorSet;
        accelerationStructureWrite.dstBinding = 2;
        accelerationStructureWrite.descriptorCount = 1;
        accelerationStructureWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        computeWriteDescriptorSets.push_back(accelerationStructureWrite);

        // 确保实例缓冲区绑定到描述符集
        VkWriteDescriptorSet instanceBufferWrite = vks::initializers::writeDescriptorSet(
            compute.descriptorSet,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            3,
            &instanceBuffer.descriptor);
        computeWriteDescriptorSets.push_back(instanceBufferWrite);

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

        // Create pipeline
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&compute.descriptorSetLayout, 1);
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &compute.pipelineLayout));
        VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(compute.pipelineLayout, 0);
        computePipelineCreateInfo.stage = loadShader(getShadersPath() + "rigidbody/rigidbody.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);
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
        compute.uniformData.restitution = 0.8f;

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
            deltaTime = 0.016f;
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
        loadRayTracingFunctions();
        setupDescriptorPool();
        prepareGraphics();
        // Create acceleration structures for collision detection
        createBottomLevelAccelerationStructure();
        createTopLevelAccelerationStructure();
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