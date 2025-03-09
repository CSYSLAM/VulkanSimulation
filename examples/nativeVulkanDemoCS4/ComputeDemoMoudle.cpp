#include "ComputeDemoMoudle.h"
#include "Node.h"
#define WORKGROUP_SIZE 128

struct Push1 {
	float value_; 
	Push1(float value) : value_(value) {};
	Push1() = default;
};

std::string shaderDir111 = "C:\\temp\\CG\\Engine\\VulkanSimulation\\shaders\\glsl\\nativeVulkanDemoCS4\\VecAdd.comp.spv";

namespace CsyVk
{
	ComputeDemoMoudle::ComputeDemoMoudle() : ComputeModule()
	{
		this->addKernel(
			"VecAdd",
			std::make_shared<VkProgram>(
				BUFFER(float),
				BUFFER(float),
				BUFFER(float),
				CONSTANT(Push1)
			)
		);
		kernel("VecAdd")->load(shaderDir111);

		Log::sendMessage(Log::Info, "Compute Demo Moudle created \n");
	}

	ComputeDemoMoudle::~ComputeDemoMoudle()
	{	
		Log::sendMessage(Log::Info, "Compute Demo Moudle release \n");
	}

	void ComputeDemoMoudle::compute()
	{

		Log::sendMessage(Log::Info, "Compute Demo Moudle compute \n");

        if (frameNum == 0) {
            this->Init();
            
        }
	
		this->Step();

        std::cout << "kCSY 111" << std::endl;
	}

	void ComputeDemoMoudle::Init()
	{
		// //init volume
		// if (frameNum == 0);
		// {
        //     frameNum++;
		// }
	}

	void ComputeDemoMoudle::Step()
	{
        float dt_ = this->inTimeStep()->getData();
        frameNum++;
        kValue += 1.5;
		//adopt forward-Euler time integration
		auto vNum = this->inInputArrayA()->getData().size();
        InputArrayC.resize(vNum);

		VkConstant<Push1> p;
        p.setValue(Push1(kValue));

		kernel("VecAdd")->flush(
			vkDispatchSize(vNum, WORKGROUP_SIZE),
			this->inInputArrayA()->getData().handle(),
			this->inInputArrayB()->getData().handle(),
            InputArrayC.handle(),
			&p);

         CArray<float> hC(vNum);
		 hC.assign(InputArrayC);
		 for (int i = 0; i < vNum; i++)
		 {
		 	printf("%f \n", hC[i]);
		 }
         hC.clear();
	}
}