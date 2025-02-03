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

#include "VkSystem.h"
#include "Array.h"
#include "VkProgram.h"

#include"VulkanTools.h"

using namespace CsyVk;

std::string shaderDir = getShaderBasePath() + "glsl/simpleDemoForCsyVk/VecAdd.comp.spv";

/**
 * This example demonstrates how to use the wrapped api for vulkan to ease the programming
 */

int main(int argc, char* argv[])
{
	VkSystem::instance()->initialize();

	//Initialize all buffers
	uint num = 100;

	DArray<float> dA(num);
	DArray<float> dB(num);
	DArray<float> dC(num);

	CArray<float> hA(num);
	CArray<float> hB(num);
	CArray<float> hC(num);

	for (int i = 0; i < num; i++)
	{
		hA[i] = float(i);
		hB[i] = float(i);
	}

	dA.assign(hA);
	dB.assign(hB);

	//Declare a kernel
	auto kernel = std::make_shared<VkProgram>(
		BUFFER(float),		//Array A
		BUFFER(float),		//Array B
		BUFFER(float),		//Array C
		CONSTANT(uint));
	kernel->load(shaderDir);

	//Execuate the kernel
	VkConstant<uint> N(num);
	kernel->flush(
		vkDispatchSize(num, 128),
		dA.handle(),
		dB.handle(),
		dC.handle(),
		&N);

	//Copy results back to the host and print out
	hC.assign(dC);
	for (int i = 0; i < num; i++)
	{
		printf("%f \n", hC[i]);
	}
	system("pause");
	return 0;
}
