#pragma once

#include <string>
#include <vector>
#include <utility>
#include "VkUniform.h"
#include "Node.h"
#include "Array.h"
namespace CsyVk
{

	class ComputeDemo : public Node
	{
	
	public:
	

		typedef typename float Real;

		typedef struct {
			DArray<float> inputArrayA;
            DArray<float> inputArrayB;
		}SimulationData;

		ComputeDemo(std::string name = "default");
		virtual ~ComputeDemo();

		void resetStates() override;
        void loadData(CArray<float> A, CArray<float> B);

	protected:

		DEF_ARRAY_STATE(float, InputArrayA, DeviceType::GPU, "first input array");

		DEF_ARRAY_STATE(float, InputArrayB, DeviceType::GPU, "second input array");

        // DEF_VAR_STATE(uint32_t, Frame, "frame index");

		SimulationData simulationData;
	};
}