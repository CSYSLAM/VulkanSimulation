#include "OutputModule.h"

namespace CsyVk
{
	OutputModule::OutputModule()
		: Module()
	{
		this->varStride()->setRange(1, 1024);
		this->inFrameNumber()->tagOptional(true);
	}

	OutputModule::~OutputModule()
	{
	}

	void OutputModule::updateImpl()
	{
		uint startFrame = this->varStartFrame()->getValue();
		uint endFrame = this->varEndFrame()->getValue();

		uint frame = this->inFrameNumber()->getValue();

		uint stride = this->varStride()->getValue();

		if (frame >= startFrame && frame <= endFrame)
		{
			if ((frame - startFrame) % stride == 0)
			{
				//OutputFile
				this->output();
			}
		}
	}

}