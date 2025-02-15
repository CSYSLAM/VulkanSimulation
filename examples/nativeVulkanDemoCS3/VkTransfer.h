#pragma once
#include <assert.h>
#include "Array.h"
#include "VkContext.h"
#include "VkHostArray.h"
#include "VkHostArray2D.h"
#include "VkDeviceArray.h"
#include "VkDeviceArray2D.h"
#include "VkDeviceArray3D.h"

namespace CsyVk
{
	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);
		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkHostArray<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size()*sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size());

		vkTransfer(vkHostSrc, src);

		memcpy(dst.data(), vkHostSrc.mapped(), sizeof(T)*src.size());

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size(), src.data());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, vkHostSrc.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		vkHostSrc.clear();

		return true;
	}


	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, const VkDeviceArray<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, uint64_t dstOffset, const VkDeviceArray<T>& src, uint64_t srcOffset, uint64_t copySize)
	{
		if (copySize <= 0)
			return false;

		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() >= dstOffset + copySize);
		assert(src.size() >= srcOffset + copySize);

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.dstOffset = dstOffset * sizeof(T);
		copyRegion.srcOffset = srcOffset * sizeof(T);
		copyRegion.size = src.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size());

		vkTransfer(vkHostSrc, src);

		memcpy(dst.data(), vkHostSrc.mapped(), sizeof(T)*src.size());

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size(), src.data());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, vkHostSrc.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray2D<T>& dst, const VkDeviceArray2D<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const std::vector<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size(), src.data());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, vkHostSrc.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray3D<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(VkHostArray<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.currentContext() == src.currentContext());
		assert(dst.size() == src.size());

		// Copy from staging buffer
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src.bufferHandle(), dst.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		return true;
	}

	template<typename T>
	bool vkTransfer(std::vector<T>& dst, const VkDeviceArray3D<T>& src)
	{
		VkContext* ctx = src.currentContext();

		assert(ctx != nullptr);
		assert(dst.size() == src.size());

		VkHostArray<T> vkHostSrc;
		vkHostSrc.resize(src.size());

		vkTransfer(vkHostSrc, src);

		memcpy(dst.data(), vkHostSrc.mapped(), sizeof(T)*src.size());

		vkHostSrc.clear();

		return true;
	}

	template<typename T>
	bool vkTransfer(VkDeviceArray<T>& dst, VkBuffer src)
	{
		VkContext* ctx = dst.currentContext();

		assert(ctx != nullptr);
		assert(dst.bufferHandle() != VK_NULL_HANDLE);

		// Create a temporary buffer to hold the data during transfer
		VkHostArray<T> tmpHostArray;
		tmpHostArray.resize(dst.size());

		// Copy data from existing VkBuffer to the temporary host array
		VkCommandBuffer copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};
		copyRegion.size = dst.size() * sizeof(T);
		vkCmdCopyBuffer(copyCmd, src, tmpHostArray.bufferHandle(), 1, &copyRegion);

		ctx->flushCommandBuffer(copyCmd, ctx->graphicsQueueHandle(), true);

		// Transfer data from the temporary host array to the DArray's device buffer
		vkTransfer(dst, tmpHostArray);
		return true;
	}
}