
#include "ComputeDemoNode.h"
#include "ComputeDemoMoudle.h"
#include <cstdlib>


namespace CsyVk
{
	ComputeDemo::ComputeDemo(std::string name) : Node()
	{
        setName(name);
		this->stateInputArrayA()->allocate();
        this->stateInputArrayB()->allocate();

		auto computeDemoMoudle = std::make_shared<ComputeDemoMoudle>();
        this->stateFrameNumber()->connect(computeDemoMoudle->inFrameNumber());
		this->stateInputArrayA()->connect(computeDemoMoudle->inInputArrayA());
		this->stateInputArrayB()->connect(computeDemoMoudle->inInputArrayB());
		this->animationPipeline()->pushModule(computeDemoMoudle);
	}


	ComputeDemo::~ComputeDemo()
	{
		Log::sendMessage(Log::Info, "ComputeDemo released \n");
		this->simulationData.inputArrayA.clear();
		this->simulationData.inputArrayB.clear();
	}

	void ComputeDemo::resetStates()
	{
		Log::sendMessage(Log::Info, "ComputeDemo reset state \n");
		
		uint vNum1 = this->simulationData.inputArrayA.size();
		uint vNum2 = this->simulationData.inputArrayB.size();

		//Assign the state to vertex size.
		if (this->stateInputArrayA()->isEmpty() || this->stateInputArrayA()->getData().size() != vNum1) {
			this->stateInputArrayA()->allocate();
            this->stateInputArrayB()->allocate();
			printf("vNum: %u\n", vNum1);

			this->stateInputArrayA()->getData().assign(this->simulationData.inputArrayA);
			this->stateInputArrayB()->getData().assign(this->simulationData.inputArrayB);
		}
		Log::sendMessage(Log::Info, "ComputeDemo reset state finished \n");
	}

	void ComputeDemo::loadData(CArray<float> A, CArray<float> B)
	{
		this->simulationData.inputArrayA.assign(A);
		this->simulationData.inputArrayB.assign(B);
	}

}