#pragma once

#include <vector>
#include "vk_base_device_csy.h"

class DeviceCsy;
class SwapChainCsy {
    friend class DeviceCsy;

public:
    VkSwapchainKHR GetVkSwapChain() const;
    VkSurfaceFormatKHR GetSurfaceFormat() const;
    VkFormat GetVkImageFormat() const;
    VkExtent2D GetVkExtent() const;
    uint32_t GetIndex() const;
    uint32_t GetCount() const;
    VkImage GetVkImage(uint32_t index) const;
    std::vector<VkImage> SwapChainCsy::GetVkImages() const;
    VkSemaphore GetImageAvailableVkSemaphore() const;
    VkSemaphore GetRenderFinishedVkSemaphore() const;
    
    void Recreate();
    bool Acquire();
    bool Present();
    ~SwapChainCsy();

private:
    SwapChainCsy(DeviceCsy* device, VkSurfaceKHR vkSurface, unsigned int numBuffers);
    void Create();
    void Destroy();

    DeviceCsy* device;
    VkSurfaceKHR vkSurface;
    unsigned int numBuffers;
    VkSwapchainKHR vkSwapChain;
    VkSurfaceFormatKHR surfaceFormat;
    std::vector<VkImage> vkSwapChainImages;
    VkFormat vkSwapChainImageFormat;
    VkExtent2D vkSwapChainExtent;
    uint32_t imageIndex = 0;

    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
};
