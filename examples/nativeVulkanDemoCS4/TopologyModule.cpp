#include "TopologyModule.h"
#include "Node.h"

namespace CsyVk
{
IMPLEMENT_CLASS(TopologyModule)

TopologyModule::TopologyModule()
	: OBase()
	, m_topologyChanged(true)
{

}

TopologyModule::~TopologyModule()
{
}

void TopologyModule::update()
{
	this->updateTopology();
}

}