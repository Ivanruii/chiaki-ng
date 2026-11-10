// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "framegen/cuda_dyn.h"

#include <mutex>
#include <stdio.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace framegen {

namespace {

void *openLibrary()
{
#ifdef _WIN32
	return reinterpret_cast<void *>(LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#else
	void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!h)
		h = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
	return h;
#endif
}

void *symbol(void *lib, const char *name)
{
#ifdef _WIN32
	return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
	return dlsym(lib, name);
#endif
}

CudaApi g_api = {};
const CudaApi *g_api_ptr = nullptr;
const char *g_api_error = nullptr;
std::once_flag g_api_once;

template<typename T>
bool load(void *lib, T &fn, const char *name)
{
	fn = reinterpret_cast<T>(symbol(lib, name));
	return fn != nullptr;
}

void loadOnce()
{
	void *lib = openLibrary();
	if (!lib) {
		g_api_error = "NVIDIA CUDA driver library not found";
		return;
	}
	bool ok = true;
#define LOAD(field, name) ok = load(lib, g_api.field, name) && ok
	LOAD(cuInit, "cuInit");
	LOAD(cuDriverGetVersion, "cuDriverGetVersion");
	LOAD(cuDeviceGetCount, "cuDeviceGetCount");
	LOAD(cuDeviceGet, "cuDeviceGet");
	LOAD(cuDeviceGetName, "cuDeviceGetName");
	LOAD(cuDeviceGetAttribute, "cuDeviceGetAttribute");
	LOAD(cuDeviceGetUuid, "cuDeviceGetUuid");
	LOAD(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
	LOAD(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
	LOAD(cuCtxPushCurrent, "cuCtxPushCurrent_v2");
	LOAD(cuCtxPopCurrent, "cuCtxPopCurrent_v2");
	LOAD(cuCtxGetDevice, "cuCtxGetDevice");
	LOAD(cuStreamCreate, "cuStreamCreate");
	LOAD(cuStreamDestroy, "cuStreamDestroy_v2");
	LOAD(cuStreamSynchronize, "cuStreamSynchronize");
	LOAD(cuMemAlloc, "cuMemAlloc_v2");
	LOAD(cuMemAllocPitch, "cuMemAllocPitch_v2");
	LOAD(cuMemFree, "cuMemFree_v2");
	LOAD(cuMemcpy2DAsync, "cuMemcpy2DAsync_v2");
	LOAD(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
	LOAD(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
	LOAD(cuMemsetD32Async, "cuMemsetD32Async");
	LOAD(cuModuleLoadData, "cuModuleLoadData");
	LOAD(cuModuleUnload, "cuModuleUnload");
	LOAD(cuModuleGetFunction, "cuModuleGetFunction");
	LOAD(cuLaunchKernel, "cuLaunchKernel");
	LOAD(cuEventCreate, "cuEventCreate");
	LOAD(cuEventDestroy, "cuEventDestroy_v2");
	LOAD(cuEventRecord, "cuEventRecord");
	LOAD(cuEventElapsedTime, "cuEventElapsedTime");
	LOAD(cuGetErrorString, "cuGetErrorString");
#undef LOAD
	if (!ok) {
		g_api_error = "NVIDIA CUDA driver is too old (missing driver API entry points)";
		return;
	}
	if (g_api.cuInit(0) != CHIAKI_CUDA_SUCCESS) {
		g_api_error = "cuInit failed (no usable NVIDIA GPU or driver)";
		return;
	}
	g_api_ptr = &g_api;
}

} // namespace

const CudaApi *cudaApi(const char **error)
{
	std::call_once(g_api_once, loadOnce);
	if (!g_api_ptr && error)
		*error = g_api_error;
	return g_api_ptr;
}

const char *cudaErrorName(CUresult r)
{
	thread_local char buf[160];
	const char *str = nullptr;
	if (g_api_ptr && g_api_ptr->cuGetErrorString)
		g_api_ptr->cuGetErrorString(r, &str);
	snprintf(buf, sizeof(buf), "%s (%d)", str ? str : "CUDA error", r);
	return buf;
}

CudaContextScope::CudaContextScope(CUcontext ctx)
{
	const CudaApi *cu = cudaApi();
	if (cu && ctx)
		pushed = cu->cuCtxPushCurrent(ctx) == CHIAKI_CUDA_SUCCESS;
}

CudaContextScope::~CudaContextScope()
{
	if (pushed) {
		CUcontext dummy = nullptr;
		cudaApi()->cuCtxPopCurrent(&dummy);
	}
}

} // namespace framegen
