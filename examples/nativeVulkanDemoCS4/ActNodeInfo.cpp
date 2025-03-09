#include "ActNodeInfo.h"

namespace CsyVk
{
	
	NodeInfoAct::NodeInfoAct()
	{

	}

	NodeInfoAct::~NodeInfoAct()
	{

	}

	void NodeInfoAct::process(Node* node)
	{
		std::cout << node->getClassInfo()->getClassName() << std::endl;
	}

}