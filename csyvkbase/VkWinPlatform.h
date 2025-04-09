#pragma once
#define PERIDYNO_VERSION 0.0.1
#define PERIDYNO_VERSION_MAJOR 0
#define PERIDYNO_VERSION_MINOR 0
#define PERIDYNO_VERSION_PATCH 1

#define PERIDYNO_EXPORT
#define PERIDYNO_IMPORT

#define PERIDYNO_API PERIDYNO_IMPORT

#define VK_BACKEND
#define DYN_FUNC
#define GPU_FUNC 
#define CPU_FUNC 

enum DeviceType
{
	CPU,
	GPU,
	UNDEFINED
};

#define PRECISION_FLOAT

#include "Typedef.inl"
const inline std::string getAssetPath() {
	return "C:/temp/CG/Code/VulkanSimulation/";
}

const inline std::string getPluginPath() {
	return "";
}
