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
		this->compute();
	}
}