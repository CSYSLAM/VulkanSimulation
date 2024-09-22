#include "vk_base_device_csy.h"
#include "vk_base_instance_csy.h"

DeviceCsy::DeviceCsy(InstanceCsy* instance, VkDevice vkDevice, Queues queues)
  : instance(instance), vkDevice(vkDevice), queues(queues) {
}


InstanceCsy* DeviceCsy::GetInstance() {
    return instance;
}


VkDevice DeviceCsy::GetVkDevice() {
    return vkDevice;
}


VkQueue DeviceCsy::GetQueue(QueueFlags flag) {
    return queues[flag];
}


unsigned int DeviceCsy::GetQueueIndex(QueueFlags flag) {
    return GetInstance()->GetQueueFamilyIndices()[flag];
}


SwapChainCsy* DeviceCsy::CreateSwapChain(VkSurfaceKHR surface, unsigned int numBuffers) {
    return new SwapChainCsy(this, surface, numBuffers);
}


DeviceCsy::~DeviceCsy() {
    vkDestroyDevice(vkDevice, nullptr);
}
