// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// Minimal, dynamically loaded subset of the CUDA driver API used by frame generation.
// The driver library (nvcuda.dll / libcuda.so.1) ships with the NVIDIA display driver,
// so no CUDA toolkit is needed to build or run chiaki-ng.
//
// The type names match <cuda.h> so FFmpeg's hwcontext_cuda.h can be included after this
// header (it skips <cuda.h> when CUDA_VERSION is defined).

#ifndef CHIAKI_FRAMEGEN_CUDA_DYN_H
#define CHIAKI_FRAMEGEN_CUDA_DYN_H

#include <stddef.h>
#include <stdint.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#define CHIAKI_FRAMEGEN_OWNS_CUDA_TYPES 1
#endif

#ifdef _WIN32
#define CHIAKI_CUDAAPI __stdcall
#else
#define CHIAKI_CUDAAPI
#endif

#ifdef CHIAKI_FRAMEGEN_OWNS_CUDA_TYPES
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUarray_st *CUarray;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUevent_st *CUevent;
typedef struct CUuuid_st { char bytes[16]; } CUuuid;

typedef enum CUmemorytype_enum {
	CU_MEMORYTYPE_HOST = 1,
	CU_MEMORYTYPE_DEVICE = 2,
	CU_MEMORYTYPE_ARRAY = 3,
	CU_MEMORYTYPE_UNIFIED = 4
} CUmemorytype;

typedef struct CUDA_MEMCPY2D_st {
	size_t srcXInBytes;
	size_t srcY;
	CUmemorytype srcMemoryType;
	const void *srcHost;
	CUdeviceptr srcDevice;
	CUarray srcArray;
	size_t srcPitch;
	size_t dstXInBytes;
	size_t dstY;
	CUmemorytype dstMemoryType;
	void *dstHost;
	CUdeviceptr dstDevice;
	CUarray dstArray;
	size_t dstPitch;
	size_t WidthInBytes;
	size_t Height;
} CUDA_MEMCPY2D;
#endif

#define CHIAKI_CUDA_SUCCESS 0
#define CHIAKI_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CHIAKI_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CHIAKI_CU_STREAM_NON_BLOCKING 0x1
#define CHIAKI_CU_EVENT_DEFAULT 0x0

namespace framegen {

struct CudaApi
{
	CUresult (CHIAKI_CUDAAPI *cuInit)(unsigned int flags);
	CUresult (CHIAKI_CUDAAPI *cuDriverGetVersion)(int *version);
	CUresult (CHIAKI_CUDAAPI *cuDeviceGetCount)(int *count);
	CUresult (CHIAKI_CUDAAPI *cuDeviceGet)(CUdevice *device, int ordinal);
	CUresult (CHIAKI_CUDAAPI *cuDeviceGetName)(char *name, int len, CUdevice dev);
	CUresult (CHIAKI_CUDAAPI *cuDeviceGetAttribute)(int *value, int attrib, CUdevice dev);
	CUresult (CHIAKI_CUDAAPI *cuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);
	CUresult (CHIAKI_CUDAAPI *cuDevicePrimaryCtxRetain)(CUcontext *ctx, CUdevice dev);
	CUresult (CHIAKI_CUDAAPI *cuDevicePrimaryCtxRelease)(CUdevice dev);
	CUresult (CHIAKI_CUDAAPI *cuCtxPushCurrent)(CUcontext ctx);
	CUresult (CHIAKI_CUDAAPI *cuCtxPopCurrent)(CUcontext *ctx);
	CUresult (CHIAKI_CUDAAPI *cuCtxGetDevice)(CUdevice *device);
	CUresult (CHIAKI_CUDAAPI *cuStreamCreate)(CUstream *stream, unsigned int flags);
	CUresult (CHIAKI_CUDAAPI *cuStreamDestroy)(CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuStreamSynchronize)(CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuMemAlloc)(CUdeviceptr *dptr, size_t bytes);
	CUresult (CHIAKI_CUDAAPI *cuMemAllocPitch)(CUdeviceptr *dptr, size_t *pitch, size_t width_bytes, size_t height, unsigned int element_size);
	CUresult (CHIAKI_CUDAAPI *cuMemFree)(CUdeviceptr dptr);
	CUresult (CHIAKI_CUDAAPI *cuMemcpy2DAsync)(const CUDA_MEMCPY2D *copy, CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuMemcpyHtoD)(CUdeviceptr dst, const void *src, size_t bytes);
	CUresult (CHIAKI_CUDAAPI *cuMemcpyDtoHAsync)(void *dst, CUdeviceptr src, size_t bytes, CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuMemsetD32Async)(CUdeviceptr dst, unsigned int value, size_t count, CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuModuleLoadData)(CUmodule *module, const void *image);
	CUresult (CHIAKI_CUDAAPI *cuModuleUnload)(CUmodule module);
	CUresult (CHIAKI_CUDAAPI *cuModuleGetFunction)(CUfunction *fn, CUmodule module, const char *name);
	CUresult (CHIAKI_CUDAAPI *cuLaunchKernel)(CUfunction f,
			unsigned int gx, unsigned int gy, unsigned int gz,
			unsigned int bx, unsigned int by, unsigned int bz,
			unsigned int shared_mem, CUstream stream, void **params, void **extra);
	CUresult (CHIAKI_CUDAAPI *cuEventCreate)(CUevent *event, unsigned int flags);
	CUresult (CHIAKI_CUDAAPI *cuEventDestroy)(CUevent event);
	CUresult (CHIAKI_CUDAAPI *cuEventRecord)(CUevent event, CUstream stream);
	CUresult (CHIAKI_CUDAAPI *cuEventElapsedTime)(float *ms, CUevent start, CUevent end);
	CUresult (CHIAKI_CUDAAPI *cuGetErrorString)(CUresult error, const char **str);
};

// Loads the driver library once per process. Returns nullptr (and fills *error) when the
// NVIDIA CUDA driver is unavailable. The returned table stays valid until process exit.
const CudaApi *cudaApi(const char **error = nullptr);

// Human readable "name (code)" for a CUresult.
const char *cudaErrorName(CUresult r);

// RAII helper that makes a context current on the calling thread.
class CudaContextScope
{
public:
	explicit CudaContextScope(CUcontext ctx);
	~CudaContextScope();
	bool ok() const { return pushed; }
private:
	bool pushed = false;
};

} // namespace framegen

#endif // CHIAKI_FRAMEGEN_CUDA_DYN_H
