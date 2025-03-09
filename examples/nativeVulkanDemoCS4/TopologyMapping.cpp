#pragma once
#include "TopologyMapping.h"

namespace CsyVk
{
	TopologyMapping::TopologyMapping()
		: Module()
	{

	}

	TopologyMapping::~TopologyMapping()
	{

	}

	void TopologyMapping::updateImpl()
	{
		this->apply();
	}

}