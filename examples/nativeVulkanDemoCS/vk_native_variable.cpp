#include "vk_native_variable.h"
#include "vk_native_system.h"

namespace CsyVkN {

	VkVariable::VkVariable()
	{
	    buffer = std::make_shared<CsyVkN::Buffer>();
		ctx = VkSystem::instance()->currentContext();
	}

	VkVariable::~VkVariable()
	{
		// TODO: sovle other issue while destroy buffer here.
		// buffer.destroy();
	}

	VkDescriptorType VkVariable::descriptorType(const VariableType varType)
	{
		switch (varType)
		{
		case DeviceBuffer:
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		case Uniform:
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		default:
			break;
		}

		return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}

}