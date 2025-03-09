#include "VisualModule.h"
#include "Node.h"

namespace CsyVk
{
	VisualModule::VisualModule()
		: Module()
	{
	}

	VisualModule::~VisualModule()
	{
	}

	void VisualModule::setVisible(bool bVisible)
	{
		this->varVisible()->setValue(bVisible);
	}
}