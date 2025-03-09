#include "ComputeModule.h"

namespace CsyVk
{
	ComputeModule::ComputeModule()
	{
	}

	ComputeModule::~ComputeModule()
	{
	}

	void ComputeModule::updateImpl()
	{
		std::cout << "ComputeModule is updated" << std::endl;
		this->compute();
	}
}