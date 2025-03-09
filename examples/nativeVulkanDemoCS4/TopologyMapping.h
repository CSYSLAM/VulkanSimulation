#pragma once
#include "Module.h"

namespace CsyVk
{
	class TopologyMapping : public Module
	{
	public:
		TopologyMapping();
		virtual ~TopologyMapping();

	protected:
		virtual bool apply() = 0;

	private:
		void updateImpl() override;
	};
}