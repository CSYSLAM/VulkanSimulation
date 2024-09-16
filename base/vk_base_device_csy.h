#pragma once

#include <array>
#include <vulkan/vulkan.h>
#include "vk_base_queueFlags_csy.h"
#include "vk_base_swapChain_csy.h"

class SwapChainCsy;
class DeviceCsy {
    friend class InstanceCsy;

public:
    SwapChainCsy* CreateSwapChain(VkSurfaceKHR surface, unsigned int numBuffers);
    InstanceCsy* GetInstance();
    VkDevice GetVkDevice();
    VkQueue GetQueue(QueueFlags flag);
    unsigned int GetQueueIndex(QueueFlags flag);
    ~DeviceCsy();

private:
    using Queues = std::array<VkQueue, sizeof(QueueFlags)>;
    
    DeviceCsy() = delete;
    DeviceCsy(InstanceCsy* instance, VkDevice vkDevice, Queues queues);

    InstanceCsy* instance;
    VkDevice vkDevice;
    Queues queues;
};
