#pragma once

#include <windows.h>

#include "VkTypeDefine.h"

#include <iostream>

namespace CsyVk 
{
	class CTimer
	{
	public:
		CTimer();
		~CTimer();
		void start();
		void stop();

		/**
		 * @brief return the elapsed time in (ms)
		 */
		double getElapsedTime();
		void outputString(char* str);
	protected:
		LARGE_INTEGER timer_frequency_;
		LARGE_INTEGER start_count_, stop_count_;
	};
} //end of namespace CsyVk
