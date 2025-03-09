#include "ConstraintModule.h"
#include "Node.h"

namespace CsyVk
{
ConstraintModule::ConstraintModule()
	: Module()
{
}

ConstraintModule::~ConstraintModule()
{
}

void ConstraintModule::updateImpl()
{
	this->constrain();
}

}