#pragma once
#include "Action.h"

namespace CsyVk
{
	class PostProcessing : public Action
	{
	public:
		PostProcessing();
		virtual ~PostProcessing();

	private:
		void process(Node* node) override;
	};
}
