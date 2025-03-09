#pragma once
#include "ComputeModule.h"
#include "VkUniform.h"
#include "ComputeDemoNode.h"

namespace CsyVk {

	class ComputeDemoMoudle : public ComputeModule
	{
	
	public:
		ComputeDemoMoudle();
		~ComputeDemoMoudle() override;

		void compute() override;
		
	public:

        DEF_VAR_IN(float, TimeStep, "Time Step");

		DEF_ARRAY_IN(float, InputArrayA, DeviceType::GPU, "first input array");

		DEF_ARRAY_IN(float, InputArrayB, DeviceType::GPU, "second input array");

        // DEF_VAR_IN(uint32_t, Frame, "frame index");

	private:

		void Init();

		void Step();

        DArray<float> InputArrayC;

		float kValue = 0.0;

        uint32_t frameNum = 0;
	};



}