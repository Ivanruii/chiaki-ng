// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "framegen/ofa_interpolator.h"
#include "framegen/nvof/nvOpticalFlowCuda.h"
#include "framegen_kernels_ptx.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <vector>

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

struct FlowF { float x, y; };

typedef NV_OF_STATUS(NVOFAPI *PFN_NvOFAPICreateInstanceCuda)(uint32_t, NV_OF_CUDA_API_FUNCTION_LIST *);

NV_OF_CUDA_API_FUNCTION_LIST g_of = {};
bool g_of_ok = false;
std::string g_of_error;
std::once_flag g_of_once;

void loadNvofOnce()
{
#ifdef _WIN32
	HMODULE lib = LoadLibraryExA("nvofapi64.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	auto create = lib ? reinterpret_cast<PFN_NvOFAPICreateInstanceCuda>(GetProcAddress(lib, "NvOFAPICreateInstanceCuda")) : nullptr;
	const char *libname = "nvofapi64.dll";
#else
	void *lib = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
	auto create = lib ? reinterpret_cast<PFN_NvOFAPICreateInstanceCuda>(dlsym(lib, "NvOFAPICreateInstanceCuda")) : nullptr;
	const char *libname = "libnvidia-opticalflow.so.1";
#endif
	if (!lib) {
		g_of_error = std::string("NVIDIA optical flow driver library (") + libname + ") not found";
		return;
	}
	if (!create) {
		g_of_error = "NvOFAPICreateInstanceCuda missing from the NVIDIA driver";
		return;
	}
	NV_OF_STATUS st = create(NV_OF_API_VERSION, &g_of);
	if (st != NV_OF_SUCCESS) {
		g_of_error = "NVIDIA driver does not support optical flow API 5.0 (status " + std::to_string(st) + "); update the driver";
		return;
	}
	g_of_ok = true;
}

const NV_OF_CUDA_API_FUNCTION_LIST *nvofApi(std::string *error)
{
	std::call_once(g_of_once, loadNvofOnce);
	if (!g_of_ok) {
		if (error)
			*error = g_of_error;
		return nullptr;
	}
	return &g_of;
}

std::string ofError(NvOFHandle h, const char *what, NV_OF_STATUS st)
{
	std::string msg = std::string(what) + " failed (NV_OF_STATUS " + std::to_string(st) + ")";
	if (h && g_of.nvOFGetLastError) {
		char buf[256] = {};
		uint32_t size = sizeof(buf);
		if (g_of.nvOFGetLastError(h, buf, &size) == NV_OF_SUCCESS && buf[0])
			msg += std::string(": ") + buf;
	}
	return msg;
}

std::string cuError(const char *what, CUresult r)
{
	return std::string(what) + " failed: " + cudaErrorName(r);
}

unsigned int divUp(unsigned int a, unsigned int b)
{
	return (a + b - 1) / b;
}

} // namespace

struct OfaInterpolator::Impl
{
	const CudaApi *cu = nullptr;
	const NV_OF_CUDA_API_FUNCTION_LIST *of = nullptr;
	CUmodule module = nullptr;
	CUfunction k_unpack = nullptr, k_median = nullptr, k_consistency = nullptr;
	CUfunction k_luma8 = nullptr, k_chroma8 = nullptr, k_luma16 = nullptr, k_chroma16 = nullptr, k_l16to8 = nullptr;
	CUfunction k_mask8 = nullptr, k_mask16 = nullptr;

	NvOFGPUBufferHandle gray[2] = {};
	CUdeviceptr gray_ptr[2] = {};
	uint32_t gray_pitch[2] = {};
	NvOFGPUBufferHandle out_fwd = nullptr, out_bwd = nullptr;
	CUdeviceptr fwd_ptr = 0, bwd_ptr = 0;
	uint32_t fwd_pitch = 0, bwd_pitch = 0;

	CUdeviceptr hist_luma[2] = {}, hist_chroma[2] = {};
	size_t hist_luma_pitch[2] = {}, hist_chroma_pitch[2] = {};

	CUdeviceptr raw01 = 0, raw10 = 0, f01 = 0, f10 = 0, vis0 = 0, vis1 = 0, counters = 0;
	CUdeviceptr zones = 0;
	int nzones = 0;
	CUdeviceptr mask = 0;   // protection weight per luma pixel (0..255), pitch = width
	bool mask_valid = false;
	CUevent ev_begin = nullptr, ev_end = nullptr;

	int grid = 4;
	int bw = 0, bh = 0;
	int cur = -1;       // history slot of the newest frame
	int frames = 0;     // frames in history (0..2)
	bool hints_valid = false;
	bool pair_valid = false;
};

OfaInterpolator::OfaInterpolator()
	: d(new Impl)
{
}

OfaInterpolator::~OfaInterpolator()
{
	destroy();
	delete d;
}

static bool loadKernels(const CudaApi *cu, CUmodule *module, std::string *error)
{
	CUresult r = cu->cuModuleLoadData(module, framegen_kernels_ptx);
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("Loading the frame generation CUDA kernels (PTX JIT)", r);
		return false;
	}
	return true;
}

OfaRuntimeInfo OfaInterpolator::probe(CUcontext ctx)
{
	OfaRuntimeInfo info;
	const char *cu_err = nullptr;
	const CudaApi *cu = cudaApi(&cu_err);
	if (!cu) {
		info.reason = cu_err ? cu_err : "CUDA unavailable";
		return info;
	}
	cu->cuDriverGetVersion(&info.driver_version);

	CUdevice dev = 0;
	bool own_ctx = false;
	if (!ctx) {
		int count = 0;
		if (cu->cuDeviceGetCount(&count) != CHIAKI_CUDA_SUCCESS || count <= 0) {
			info.reason = "No CUDA device found";
			return info;
		}
		cu->cuDeviceGet(&dev, 0);
		if (cu->cuDevicePrimaryCtxRetain(&ctx, dev) != CHIAKI_CUDA_SUCCESS) {
			info.reason = "Could not create a CUDA context";
			return info;
		}
		own_ctx = true;
	}
	{
		CudaContextScope scope(ctx);
		if (!scope.ok()) {
			info.reason = "Could not activate the CUDA context";
		} else {
			cu->cuCtxGetDevice(&dev);
			char name[256] = {};
			cu->cuDeviceGetName(name, sizeof(name), dev);
			info.device_name = name;
			cu->cuDeviceGetAttribute(&info.cc_major, CHIAKI_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
			cu->cuDeviceGetAttribute(&info.cc_minor, CHIAKI_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

			std::string of_err;
			const NV_OF_CUDA_API_FUNCTION_LIST *of = nvofApi(&of_err);
			if (info.cc_major < 7 || (info.cc_major == 7 && info.cc_minor < 5)) {
				info.reason = "GPU has no optical flow accelerator (Turing / RTX 20 or newer required)";
			} else if (!of) {
				info.reason = of_err;
			} else {
				NvOFHandle h = nullptr;
				NV_OF_STATUS st = of->nvCreateOpticalFlowCuda(ctx, &h);
				if (st != NV_OF_SUCCESS) {
					info.reason = "Optical flow engine unavailable on this GPU (NV_OF_STATUS " + std::to_string(st) + ")";
				} else {
					uint32_t size = 0;
					if (of->nvOFGetCaps(h, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size) == NV_OF_SUCCESS && size > 0) {
						std::vector<uint32_t> vals(size);
						if (of->nvOFGetCaps(h, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, vals.data(), &size) == NV_OF_SUCCESS)
							for (uint32_t v : vals)
								info.grid_sizes |= v;
					}
					of->nvOFDestroy(h);
					CUmodule module = nullptr;
					std::string kerr;
					if (!loadKernels(cu, &module, &kerr)) {
						info.reason = kerr;
					} else {
						cu->cuModuleUnload(module);
						info.available = true;
					}
				}
			}
		}
	}
	if (own_ctx)
		cu->cuDevicePrimaryCtxRelease(dev);
	return info;
}

bool OfaInterpolator::init(CUcontext context, int width, int height, PixelLayout layout, std::string *error)
{
	destroy();
	const char *cu_err = nullptr;
	d->cu = cudaApi(&cu_err);
	if (!d->cu) {
		if (error)
			*error = cu_err ? cu_err : "CUDA unavailable";
		return false;
	}
	d->of = nvofApi(error);
	if (!d->of)
		return false;
	if (width < 64 || height < 64 || (width & 1) || (height & 1)) {
		if (error)
			*error = "Unsupported frame size for interpolation";
		return false;
	}
	const CudaApi *cu = d->cu;
	const NV_OF_CUDA_API_FUNCTION_LIST *of = d->of;
	ctx = context;
	w = width;
	h = height;
	fmt = layout;
	CudaContextScope scope(ctx);
	if (!scope.ok()) {
		if (error)
			*error = "Could not activate the CUDA context";
		return false;
	}

	auto fail = [&](const std::string &msg) {
		if (error)
			*error = msg;
		destroy();
		return false;
	};

	// Blocking stream (flags 0): it implicitly orders with the legacy default stream that
	// FFmpeg's NVDEC output copies and CUDA -> Vulkan transfers use.
	CUresult r = cu->cuStreamCreate(&cu_stream, 0);
	if (r != CHIAKI_CUDA_SUCCESS)
		return fail(cuError("cuStreamCreate", r));
	if (!loadKernels(cu, &d->module, error)) {
		destroy();
		return false;
	}
	struct { CUfunction *fn; const char *name; } kernels[] = {
		{ &d->k_unpack, "fg_unpack_flow" },
		{ &d->k_median, "fg_median_flow" },
		{ &d->k_consistency, "fg_consistency" },
		{ &d->k_luma8, "fg_synth_luma8" },
		{ &d->k_chroma8, "fg_synth_chroma8" },
		{ &d->k_luma16, "fg_synth_luma16" },
		{ &d->k_chroma16, "fg_synth_chroma16" },
		{ &d->k_l16to8, "fg_luma16_to_gray8" },
		{ &d->k_mask8, "fg_zone_mask8" },
		{ &d->k_mask16, "fg_zone_mask16" },
	};
	for (auto &k : kernels) {
		r = cu->cuModuleGetFunction(k.fn, d->module, k.name);
		if (r != CHIAKI_CUDA_SUCCESS)
			return fail(cuError(k.name, r));
	}

	NvOFHandle hof = nullptr;
	NV_OF_STATUS st = of->nvCreateOpticalFlowCuda(ctx, &hof);
	if (st != NV_OF_SUCCESS)
		return fail(ofError(nullptr, "nvCreateOpticalFlowCuda", st));
	of_handle = hof;

	uint32_t grid_mask = 0;
	{
		uint32_t size = 0;
		if (of->nvOFGetCaps(hof, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, nullptr, &size) == NV_OF_SUCCESS && size > 0) {
			std::vector<uint32_t> vals(size);
			if (of->nvOFGetCaps(hof, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, vals.data(), &size) == NV_OF_SUCCESS)
				for (uint32_t v : vals)
					grid_mask |= v;
		}
	}
	// 4x4 blocks: what NVIDIA's own FRUC library uses, and cheapest to post-process.
	d->grid = (grid_mask & NV_OF_OUTPUT_VECTOR_GRID_SIZE_4) || !grid_mask ? 4
	        : (grid_mask & NV_OF_OUTPUT_VECTOR_GRID_SIZE_2) ? 2 : 1;
	d->bw = static_cast<int>(divUp(w, d->grid));
	d->bh = static_cast<int>(divUp(h, d->grid));

	NV_OF_INIT_PARAMS ip = {};
	ip.width = static_cast<uint32_t>(w);
	ip.height = static_cast<uint32_t>(h);
	ip.outGridSize = static_cast<NV_OF_OUTPUT_VECTOR_GRID_SIZE>(d->grid);
	ip.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_UNDEFINED;
	ip.mode = NV_OF_MODE_OPTICALFLOW;
	ip.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
	ip.enableExternalHints = NV_OF_FALSE;
	ip.enableOutputCost = NV_OF_FALSE;
	ip.hPrivData = nullptr;
	ip.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
	ip.enableRoi = NV_OF_FALSE;
	ip.predDirection = NV_OF_PRED_DIRECTION_BOTH;
	ip.enableGlobalFlow = NV_OF_FALSE;
	ip.inputBufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
	st = of->nvOFInit(hof, &ip);
	if (st != NV_OF_SUCCESS)
		return fail(ofError(hof, "nvOFInit", st));
	st = of->nvOFSetIOCudaStreams(hof, cu_stream, cu_stream);
	if (st != NV_OF_SUCCESS)
		return fail(ofError(hof, "nvOFSetIOCudaStreams", st));

	auto createBuffer = [&](uint32_t bwidth, uint32_t bheight, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT format,
	                        NvOFGPUBufferHandle *handle, CUdeviceptr *ptr, uint32_t *pitch) -> bool {
		NV_OF_BUFFER_DESCRIPTOR desc = {};
		desc.width = bwidth;
		desc.height = bheight;
		desc.bufferUsage = usage;
		desc.bufferFormat = format;
		NV_OF_STATUS s = of->nvOFCreateGPUBufferCuda(hof, &desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, handle);
		if (s != NV_OF_SUCCESS) {
			if (error)
				*error = ofError(hof, "nvOFCreateGPUBufferCuda", s);
			return false;
		}
		*ptr = of->nvOFGPUBufferGetCUdeviceptr(*handle);
		NV_OF_CUDA_BUFFER_STRIDE_INFO stride = {};
		s = of->nvOFGPUBufferGetStrideInfo(*handle, &stride);
		if (s != NV_OF_SUCCESS || !*ptr) {
			if (error)
				*error = ofError(hof, "nvOFGPUBufferGetStrideInfo", s);
			return false;
		}
		*pitch = stride.strideInfo[0].strideXInBytes;
		return true;
	};
	for (int i = 0; i < 2; i++)
		if (!createBuffer(w, h, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, &d->gray[i], &d->gray_ptr[i], &d->gray_pitch[i])) {
			destroy();
			return false;
		}
	if (!createBuffer(d->bw, d->bh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &d->out_fwd, &d->fwd_ptr, &d->fwd_pitch)
	    || !createBuffer(d->bw, d->bh, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, &d->out_bwd, &d->bwd_ptr, &d->bwd_pitch)) {
		destroy();
		return false;
	}

	const size_t bpp = fmt == PixelLayout::P010 ? 2 : 1;
	for (int i = 0; i < 2; i++) {
		r = cu->cuMemAllocPitch(&d->hist_luma[i], &d->hist_luma_pitch[i], w * bpp, h, 16);
		if (r == CHIAKI_CUDA_SUCCESS)
			r = cu->cuMemAllocPitch(&d->hist_chroma[i], &d->hist_chroma_pitch[i], w * bpp, h / 2, 16);
		if (r != CHIAKI_CUDA_SUCCESS)
			return fail(cuError("cuMemAllocPitch", r));
	}
	const size_t flow_bytes = sizeof(FlowF) * d->bw * d->bh;
	const size_t vis_bytes = sizeof(float) * d->bw * d->bh;
	CUdeviceptr *flow_bufs[] = { &d->raw01, &d->raw10, &d->f01, &d->f10 };
	for (CUdeviceptr *p : flow_bufs)
		if ((r = cu->cuMemAlloc(p, flow_bytes)) != CHIAKI_CUDA_SUCCESS)
			return fail(cuError("cuMemAlloc", r));
	if ((r = cu->cuMemAlloc(&d->vis0, vis_bytes)) != CHIAKI_CUDA_SUCCESS
	    || (r = cu->cuMemAlloc(&d->vis1, vis_bytes)) != CHIAKI_CUDA_SUCCESS
	    || (r = cu->cuMemAlloc(&d->counters, 4 * sizeof(uint32_t))) != CHIAKI_CUDA_SUCCESS
	    || (r = cu->cuMemAlloc(&d->zones, kMaxProtectedZones * 4 * sizeof(float))) != CHIAKI_CUDA_SUCCESS
	    || (r = cu->cuMemAlloc(&d->mask, static_cast<size_t>(w) * h)) != CHIAKI_CUDA_SUCCESS)
		return fail(cuError("cuMemAlloc", r));
	cu->cuEventCreate(&d->ev_begin, CHIAKI_CU_EVENT_DEFAULT);
	cu->cuEventCreate(&d->ev_end, CHIAKI_CU_EVENT_DEFAULT);
	reset();
	return true;
}

void OfaInterpolator::destroy()
{
	if (!d->cu || !ctx) {
		of_handle = nullptr;
		cu_stream = nullptr;
		return;
	}
	const CudaApi *cu = d->cu;
	CudaContextScope scope(ctx);
	if (cu_stream)
		cu->cuStreamSynchronize(cu_stream);
	if (d->of) {
		for (auto &g : d->gray)
			if (g)
				d->of->nvOFDestroyGPUBufferCuda(g);
		if (d->out_fwd)
			d->of->nvOFDestroyGPUBufferCuda(d->out_fwd);
		if (d->out_bwd)
			d->of->nvOFDestroyGPUBufferCuda(d->out_bwd);
		if (of_handle)
			d->of->nvOFDestroy(static_cast<NvOFHandle>(of_handle));
	}
	for (int i = 0; i < 2; i++) {
		if (d->hist_luma[i])
			cu->cuMemFree(d->hist_luma[i]);
		if (d->hist_chroma[i])
			cu->cuMemFree(d->hist_chroma[i]);
	}
	CUdeviceptr bufs[] = { d->raw01, d->raw10, d->f01, d->f10, d->vis0, d->vis1, d->counters, d->zones, d->mask };
	for (CUdeviceptr p : bufs)
		if (p)
			cu->cuMemFree(p);
	if (d->ev_begin)
		cu->cuEventDestroy(d->ev_begin);
	if (d->ev_end)
		cu->cuEventDestroy(d->ev_end);
	if (d->module)
		cu->cuModuleUnload(d->module);
	if (cu_stream)
		cu->cuStreamDestroy(cu_stream);
	const CudaApi *keep_cu = d->cu;
	*d = Impl();
	d->cu = keep_cu;
	of_handle = nullptr;
	cu_stream = nullptr;
	ctx = nullptr;
}

void OfaInterpolator::reset()
{
	d->cur = -1;
	d->frames = 0;
	d->hints_valid = false;
	d->pair_valid = false;
}

PairAnalysis OfaInterpolator::pushFrame(const GpuFrame &src, std::string *error)
{
	PairAnalysis res;
	if (!initialized()) {
		if (error)
			*error = "Interpolator not initialized";
		return res;
	}
	const CudaApi *cu = d->cu;
	const NV_OF_CUDA_API_FUNCTION_LIST *of = d->of;
	CudaContextScope scope(ctx);
	const int slot = d->cur < 0 ? 0 : (d->cur ^ 1);
	const size_t bpp = fmt == PixelLayout::P010 ? 2 : 1;

	auto copy2d = [&](CUdeviceptr dst, size_t dst_pitch, CUdeviceptr srcp, size_t src_pitch, size_t row_bytes, size_t rows) {
		CUDA_MEMCPY2D c = {};
		c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		c.srcDevice = srcp;
		c.srcPitch = src_pitch;
		c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		c.dstDevice = dst;
		c.dstPitch = dst_pitch;
		c.WidthInBytes = row_bytes;
		c.Height = rows;
		return cu->cuMemcpy2DAsync(&c, cu_stream);
	};

	CUresult r = copy2d(d->hist_luma[slot], d->hist_luma_pitch[slot], src.luma, src.luma_pitch, w * bpp, h);
	if (r == CHIAKI_CUDA_SUCCESS)
		r = copy2d(d->hist_chroma[slot], d->hist_chroma_pitch[slot], src.chroma, src.chroma_pitch, w * bpp, h / 2);
	if (r == CHIAKI_CUDA_SUCCESS) {
		if (fmt == PixelLayout::NV12) {
			r = copy2d(d->gray_ptr[slot], d->gray_pitch[slot], src.luma, src.luma_pitch, w, h);
		} else {
			CUdeviceptr s = src.luma;
			unsigned int sp = static_cast<unsigned int>(src.luma_pitch);
			CUdeviceptr g = d->gray_ptr[slot];
			unsigned int gp = d->gray_pitch[slot];
			int ww = w, hh = h;
			void *args[] = { &s, &sp, &g, &gp, &ww, &hh };
			r = cu->cuLaunchKernel(d->k_l16to8, divUp(w, 32), divUp(h, 8), 1, 32, 8, 1, 0, cu_stream, args, nullptr);
		}
	}
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("Copying the decoded frame", r);
		reset();
		return res;
	}
	const int prev = d->cur;
	d->cur = slot;
	d->mask_valid = false;
	d->frames = std::min(d->frames + 1, 2);
	d->pair_valid = false;
	if (d->frames < 2)
		return res;

	cu->cuEventRecord(d->ev_begin, cu_stream);
	NV_OF_EXECUTE_INPUT_PARAMS in = {};
	in.inputFrame = d->gray[prev];
	in.referenceFrame = d->gray[slot];
	in.disableTemporalHints = d->hints_valid ? NV_OF_FALSE : NV_OF_TRUE;
	NV_OF_EXECUTE_OUTPUT_PARAMS out = {};
	out.outputBuffer = d->out_fwd;
	out.bwdOutputBuffer = d->out_bwd;
	NV_OF_STATUS st = of->nvOFExecute(static_cast<NvOFHandle>(of_handle), &in, &out);
	if (st != NV_OF_SUCCESS) {
		if (error)
			*error = ofError(static_cast<NvOFHandle>(of_handle), "nvOFExecute", st);
		reset();
		return res;
	}

	int bw = d->bw, bh = d->bh;
	float grid = static_cast<float>(d->grid);
	const unsigned gx = divUp(bw, 16), gy = divUp(bh, 8);
	{
		CUdeviceptr fwd = d->fwd_ptr, bwd = d->bwd_ptr, f01 = d->raw01, f10 = d->raw10;
		unsigned int fp = d->fwd_pitch, bp = d->bwd_pitch;
		void *args[] = { &fwd, &fp, &bwd, &bp, &bw, &bh, &f01, &f10 };
		r = cu->cuLaunchKernel(d->k_unpack, gx, gy, 1, 16, 8, 1, 0, cu_stream, args, nullptr);
	}
	for (int i = 0; i < 2 && r == CHIAKI_CUDA_SUCCESS; i++) {
		CUdeviceptr s = i ? d->raw10 : d->raw01;
		CUdeviceptr t = i ? d->f10 : d->f01;
		void *args[] = { &s, &t, &bw, &bh };
		r = cu->cuLaunchKernel(d->k_median, gx, gy, 1, 16, 8, 1, 0, cu_stream, args, nullptr);
	}
	if (r == CHIAKI_CUDA_SUCCESS)
		r = cu->cuMemsetD32Async(d->counters, 0, 4, cu_stream);
	if (r == CHIAKI_CUDA_SUCCESS) {
		CUdeviceptr f01 = d->f01, f10 = d->f10, v0 = d->vis0, v1 = d->vis1, c = d->counters;
		void *args[] = { &f01, &f10, &bw, &bh, &grid, &v0, &v1, &c };
		r = cu->cuLaunchKernel(d->k_consistency, gx, gy, 1, 16, 8, 1, 0, cu_stream, args, nullptr);
	}
	uint32_t counters[4] = {};
	if (r == CHIAKI_CUDA_SUCCESS)
		r = cu->cuMemcpyDtoHAsync(counters, d->counters, sizeof(counters), cu_stream);
	if (r == CHIAKI_CUDA_SUCCESS)
		cu->cuEventRecord(d->ev_end, cu_stream);
	if (r == CHIAKI_CUDA_SUCCESS)
		r = cu->cuStreamSynchronize(cu_stream);
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("Optical flow post-processing", r);
		reset();
		return res;
	}
	cu->cuEventElapsedTime(&res.flow_ms, d->ev_begin, d->ev_end);
#ifdef CHIAKI_FRAMEGEN_DEBUG
	{
		std::vector<FlowF> a(bw * bh), b(bw * bh);
		std::vector<float> va(bw * bh);
		cu->cuMemcpyDtoHAsync(a.data(), d->f01, a.size() * sizeof(FlowF), cu_stream);
		cu->cuMemcpyDtoHAsync(b.data(), d->f10, b.size() * sizeof(FlowF), cu_stream);
		cu->cuMemcpyDtoHAsync(va.data(), d->vis0, va.size() * sizeof(float), cu_stream);
		cu->cuStreamSynchronize(cu_stream);
		for (int yy = bh / 8; yy < bh; yy += bh / 4)
			for (int xx = bw / 8; xx < bw; xx += bw / 4) {
				const int i = yy * bw + xx;
				fprintf(stderr, "blk(%d,%d) f01=(%.2f,%.2f) f10=(%.2f,%.2f) v0=%.2f\n", xx, yy, a[i].x, a[i].y, b[i].x, b[i].y, va[i]);
			}
	}
#endif
	const float blocks = static_cast<float>(bw * bh);
	res.unreliable_ratio = counters[0] / blocks;
	res.moving_ratio = counters[1] / blocks;
	res.scene_cut = res.unreliable_ratio > 0.45f;
	res.valid = !res.scene_cut;
	d->pair_valid = res.valid;
	// Hints from a scene cut would mislead the next pair
	d->hints_valid = !res.scene_cut;
	return res;
}

bool OfaInterpolator::synthesize(float t, const GpuFrame &dst, std::string *error)
{
	if (!initialized() || !d->pair_valid || t <= 0.0f || t >= 1.0f) {
		if (error)
			*error = "No valid frame pair to interpolate";
		return false;
	}
	const CudaApi *cu = d->cu;
	CudaContextScope scope(ctx);
	const int i1 = d->cur, i0 = d->cur ^ 1;
	int bw = d->bw, bh = d->bh;
	float grid = static_cast<float>(d->grid);
	CUdeviceptr f01 = d->f01, f10 = d->f10, v0 = d->vis0, v1 = d->vis1;
	const bool hi = fmt == PixelLayout::P010;

	CUdeviceptr y0 = d->hist_luma[i0], y1 = d->hist_luma[i1], yd = dst.luma;
	unsigned int y0p = static_cast<unsigned int>(d->hist_luma_pitch[i0]);
	unsigned int y1p = static_cast<unsigned int>(d->hist_luma_pitch[i1]);
	unsigned int ydp = static_cast<unsigned int>(dst.luma_pitch);
	int ww = w, hh = h;
	CUresult r = CHIAKI_CUDA_SUCCESS;
	if (d->nzones > 0 && !d->mask_valid) {
		// once per pair: protection weight = feathered zone x "barely changes between originals"
		CUdeviceptr zones = d->zones, mask = d->mask;
		int nzones = d->nzones, mask_pitch = w;
		float feather = std::max(8.0f, 24.0f * h / 1080.0f);
		float static_lo = 3.0f, static_hi = 10.0f;
		void *margs[] = { &y0, &y0p, &y1, &y1p, &ww, &hh, &zones, &nzones, &feather, &static_lo, &static_hi, &mask, &mask_pitch };
		r = cu->cuLaunchKernel(hi ? d->k_mask16 : d->k_mask8, divUp(w, 32), divUp(h, 8), 1, 32, 8, 1, 0, cu_stream, margs, nullptr);
		if (r != CHIAKI_CUDA_SUCCESS) {
			if (error)
				*error = cuError("Protected zone mask kernel", r);
			return false;
		}
		d->mask_valid = true;
	}
	CUdeviceptr mask = d->nzones > 0 ? d->mask : 0;
	int mask_pitch = w;
	void *largs[] = { &y0, &y0p, &y1, &y1p, &yd, &ydp, &ww, &hh, &f01, &f10, &v0, &v1, &bw, &bh, &grid, &t, &mask, &mask_pitch };
	r = cu->cuLaunchKernel(hi ? d->k_luma16 : d->k_luma8, divUp(w, 32), divUp(h, 8), 1, 32, 8, 1, 0, cu_stream, largs, nullptr);
	if (r == CHIAKI_CUDA_SUCCESS) {
		CUdeviceptr c0 = d->hist_chroma[i0], c1 = d->hist_chroma[i1], cd = dst.chroma;
		unsigned int c0p = static_cast<unsigned int>(d->hist_chroma_pitch[i0]);
		unsigned int c1p = static_cast<unsigned int>(d->hist_chroma_pitch[i1]);
		unsigned int cdp = static_cast<unsigned int>(dst.chroma_pitch);
		int cw = w / 2, ch = h / 2;
		void *cargs[] = { &c0, &c0p, &c1, &c1p, &cd, &cdp, &cw, &ch, &f01, &f10, &v0, &v1, &bw, &bh, &grid, &t, &mask, &mask_pitch };
		r = cu->cuLaunchKernel(hi ? d->k_chroma16 : d->k_chroma8, divUp(cw, 32), divUp(ch, 8), 1, 32, 8, 1, 0, cu_stream, cargs, nullptr);
	}
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("Frame synthesis kernel", r);
		return false;
	}
	return true;
}

bool OfaInterpolator::setProtectedZones(const std::vector<ProtectedZone> &zones, std::string *error)
{
	if (!initialized())
		return false;
	float data[kMaxProtectedZones * 4] = {};
	int n = 0;
	for (const ProtectedZone &z : zones) {
		if (n >= kMaxProtectedZones)
			break;
		const float x0 = std::clamp(std::min(z.x0, z.x1), 0.0f, 1.0f) * w;
		const float x1 = std::clamp(std::max(z.x0, z.x1), 0.0f, 1.0f) * w;
		const float y0 = std::clamp(std::min(z.y0, z.y1), 0.0f, 1.0f) * h;
		const float y1 = std::clamp(std::max(z.y0, z.y1), 0.0f, 1.0f) * h;
		if (x1 - x0 < 1.0f || y1 - y0 < 1.0f)
			continue;
		data[4 * n + 0] = x0;
		data[4 * n + 1] = y0;
		data[4 * n + 2] = x1;
		data[4 * n + 3] = y1;
		n++;
	}
	CudaContextScope scope(ctx);
	// cuMemcpyHtoD is synchronous with respect to the blocking stream used for synthesis
	CUresult r = n > 0 ? d->cu->cuMemcpyHtoD(d->zones, data, sizeof(float) * 4 * n) : CHIAKI_CUDA_SUCCESS;
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("Uploading protected zones", r);
		return false;
	}
	d->nzones = n;
	d->mask_valid = false;
	return true;
}

bool OfaInterpolator::synchronize(std::string *error)
{
	if (!initialized())
		return false;
	CudaContextScope scope(ctx);
	CUresult r = d->cu->cuStreamSynchronize(cu_stream);
	if (r != CHIAKI_CUDA_SUCCESS) {
		if (error)
			*error = cuError("cuStreamSynchronize", r);
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------------------
// Self test
// ---------------------------------------------------------------------------------------

namespace {

struct SceneParams
{
	float bg_vx = 5.0f, bg_vy = 2.0f;    // background motion per original frame (px)
	float fg_vx = -12.0f, fg_vy = 7.0f;  // foreground square motion per original frame (px)
	float fg_x = 0.0f, fg_y = 0.0f;      // foreground top-left at time 0
	float fg_size = 0.0f;
	// static overlay (game HUD) drawn on top of everything, same in every frame
	float hud_x = 0.0f, hud_y = 0.0f, hud_w = 0.0f, hud_h = 0.0f;
};

// Smooth value noise: non-periodic texture so the motion search has a unique answer, like
// natural video (a periodic test pattern would make block matching legitimately ambiguous).
float hashLattice(int x, int y, int seed)
{
	uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u + static_cast<uint32_t>(seed) * 2147483647u;
	h = (h ^ (h >> 13)) * 1274126177u;
	h ^= h >> 16;
	return static_cast<float>(h & 0xffffu) / 65535.0f;
}

float valueNoise(float x, float y, int seed)
{
	const float fx = floorf(x), fy = floorf(y);
	const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
	float tx = x - fx, ty = y - fy;
	tx = tx * tx * (3.0f - 2.0f * tx);
	ty = ty * ty * (3.0f - 2.0f * ty);
	const float a = hashLattice(ix, iy, seed), b = hashLattice(ix + 1, iy, seed);
	const float c = hashLattice(ix, iy + 1, seed), d = hashLattice(ix + 1, iy + 1, seed);
	return (a + (b - a) * tx) * (1.0f - ty) + (c + (d - c) * tx) * ty;
}

float texture(float x, float y, int seed)
{
	return 0.5f * valueNoise(x / 23.0f, y / 23.0f, seed) + 0.3f * valueNoise(x / 9.0f, y / 9.0f, seed + 1)
	     + 0.2f * valueNoise(x / 4.0f, y / 4.0f, seed + 2);
}

float bgLuma(float x, float y)
{
	return 30.0f + 200.0f * texture(x, y, 11);
}

float fgLuma(float x, float y)
{
	return 20.0f + 220.0f * texture(x, y, 47);
}

float bgChroma(float x, float y, int c)
{
	return 98.0f + 60.0f * texture(x * 0.5f, y * 0.5f, 101 + c * 7);
}

float fgChroma(float x, float y, int c)
{
	return 70.0f + 120.0f * texture(x * 0.5f, y * 0.5f, 131 + c * 7);
}

// Renders the scene at time `time` (in original-frame units) as 8-bit NV12.
void renderScene(const SceneParams &sp, float time, int w, int h, std::vector<uint8_t> &luma, std::vector<uint8_t> &chroma)
{
	luma.resize(static_cast<size_t>(w) * h);
	chroma.resize(static_cast<size_t>(w) * (h / 2));
	const float fx0 = sp.fg_x + sp.fg_vx * time, fy0 = sp.fg_y + sp.fg_vy * time;
	const float ox = sp.bg_vx * time, oy = sp.bg_vy * time;
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++) {
			const float cx = x + 0.5f, cy = y + 0.5f;
			float v;
			if (cx >= sp.hud_x && cx < sp.hud_x + sp.hud_w && cy >= sp.hud_y && cy < sp.hud_y + sp.hud_h)
				v = 40.0f + 200.0f * texture(cx, cy, 211);
			else if (cx >= fx0 && cx < fx0 + sp.fg_size && cy >= fy0 && cy < fy0 + sp.fg_size)
				v = fgLuma(cx - fx0, cy - fy0);
			else
				v = bgLuma(cx - ox, cy - oy);
			luma[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v + 0.5f)));
		}
	for (int y = 0; y < h / 2; y++)
		for (int x = 0; x < w / 2; x++)
			for (int c = 0; c < 2; c++) {
				const float cx = 2.0f * (x + 0.5f), cy = 2.0f * (y + 0.5f);
				float v;
				if (cx >= sp.hud_x && cx < sp.hud_x + sp.hud_w && cy >= sp.hud_y && cy < sp.hud_y + sp.hud_h)
					v = 60.0f + 130.0f * texture(cx * 0.5f, cy * 0.5f, 241 + c * 7);
				else if (cx >= fx0 && cx < fx0 + sp.fg_size && cy >= fy0 && cy < fy0 + sp.fg_size)
					v = fgChroma(cx - fx0, cy - fy0, c);
				else
					v = bgChroma(cx - ox, cy - oy, c);
				chroma[static_cast<size_t>(y) * w + 2 * x + c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v + 0.5f)));
			}
}

double psnr(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b, int w, int h, int border)
{
	double se = 0.0;
	size_t n = 0;
	for (int y = border; y < h - border; y++)
		for (int x = border; x < w - border; x++) {
			const double dd = static_cast<double>(a[static_cast<size_t>(y) * w + x]) - b[static_cast<size_t>(y) * w + x];
			se += dd * dd;
			n++;
		}
	const double mse = se / std::max<size_t>(n, 1);
	return mse <= 1e-9 ? 99.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

} // namespace

bool OfaInterpolator::selfTest(CUcontext ctx, std::string *report, int w, int h, PixelLayout layout)
{
	std::string err;
	const CudaApi *cu = cudaApi();
	if (!cu) {
		if (report)
			*report = "CUDA unavailable";
		return false;
	}
	OfaInterpolator ip;
	if (!ip.init(ctx, w, h, layout, &err)) {
		if (report)
			*report = "init: " + err;
		return false;
	}
	CudaContextScope scope(ctx);
	SceneParams sp;
	sp.fg_size = std::min(w, h) * 0.3f;
	sp.fg_x = w * 0.55f;
	sp.fg_y = h * 0.2f;

	// P010 samples carry the 8-bit test values in their high byte
	const size_t bpp = layout == PixelLayout::P010 ? 2 : 1;
	CUdeviceptr y_buf = 0, c_buf = 0, oy = 0, oc = 0;
	size_t yp = 0, cp = 0, oyp = 0, ocp = 0;
	bool ok = cu->cuMemAllocPitch(&y_buf, &yp, w * bpp, h, 16) == CHIAKI_CUDA_SUCCESS
	       && cu->cuMemAllocPitch(&c_buf, &cp, w * bpp, h / 2, 16) == CHIAKI_CUDA_SUCCESS
	       && cu->cuMemAllocPitch(&oy, &oyp, w * bpp, h, 16) == CHIAKI_CUDA_SUCCESS
	       && cu->cuMemAllocPitch(&oc, &ocp, w * bpp, h / 2, 16) == CHIAKI_CUDA_SUCCESS;
	std::string log;
	auto widen = [&](const std::vector<uint8_t> &v) {
		std::vector<uint8_t> out(v.size() * bpp);
		for (size_t i = 0; i < v.size(); i++) {
			if (bpp == 2) {
				out[2 * i] = 0;
				out[2 * i + 1] = v[i];
			} else {
				out[i] = v[i];
			}
		}
		return out;
	};
	auto copyPlane = [&](const void *host_src, CUdeviceptr dev, size_t pitch, int rows, bool to_device, void *host_dst) {
		CUDA_MEMCPY2D c = {};
		if (to_device) {
			c.srcMemoryType = CU_MEMORYTYPE_HOST;
			c.srcHost = host_src;
			c.srcPitch = w * bpp;
			c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			c.dstDevice = dev;
			c.dstPitch = pitch;
		} else {
			c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
			c.srcDevice = dev;
			c.srcPitch = pitch;
			c.dstMemoryType = CU_MEMORYTYPE_HOST;
			c.dstHost = host_dst;
			c.dstPitch = w * bpp;
		}
		c.WidthInBytes = w * bpp;
		c.Height = rows;
		return cu->cuMemcpy2DAsync(&c, ip.stream()) == CHIAKI_CUDA_SUCCESS && cu->cuStreamSynchronize(ip.stream()) == CHIAKI_CUDA_SUCCESS;
	};
	auto upload = [&](const std::vector<uint8_t> &luma, const std::vector<uint8_t> &chroma) {
		const std::vector<uint8_t> l = widen(luma), c = widen(chroma);
		return copyPlane(l.data(), y_buf, yp, h, true, nullptr) && copyPlane(c.data(), c_buf, cp, h / 2, true, nullptr);
	};
	auto download = [&](std::vector<uint8_t> &luma, std::vector<uint8_t> &chroma) {
		std::vector<uint8_t> l(static_cast<size_t>(w) * h * bpp), c(static_cast<size_t>(w) * (h / 2) * bpp);
		if (!copyPlane(nullptr, oy, oyp, h, false, l.data()) || !copyPlane(nullptr, oc, ocp, h / 2, false, c.data()))
			return false;
		luma.resize(l.size() / bpp);
		chroma.resize(c.size() / bpp);
		for (size_t i = 0; i < luma.size(); i++)
			luma[i] = l[i * bpp + bpp - 1];
		for (size_t i = 0; i < chroma.size(); i++)
			chroma[i] = c[i * bpp + bpp - 1];
		return true;
	};

	GpuFrame in{ y_buf, yp, c_buf, cp };
	GpuFrame out{ oy, oyp, oc, ocp };
	std::vector<uint8_t> l0, c0, l1, c1, gt, gc, res, res_c;
	PairAnalysis pa;
	// three frames so the last pair runs with hardware temporal hints, as in streaming
	for (int f = 0; ok && f < 3; f++) {
		std::vector<uint8_t> l, c;
		renderScene(sp, static_cast<float>(f), w, h, l, c);
		if (f == 1) { l0 = l; c0 = c; }
		if (f == 2) { l1 = l; c1 = c; }
		ok = upload(l, c);
		if (ok) {
			pa = ip.pushFrame(in, &err);
			ok = err.empty();
		}
	}
	if (ok && !pa.valid) {
		ok = false;
		err = "pair rejected as scene cut (unreliable=" + std::to_string(pa.unreliable_ratio) + ")";
	}
	const int border = 16;
	const float positions[] = { 0.25f, 1.0f / 3.0f, 0.5f, 2.0f / 3.0f, 0.75f };
	char line[256];
	snprintf(line, sizeof(line), "flow %.2f ms, unreliable %.1f%%, moving %.1f%%\n", pa.flow_ms, pa.unreliable_ratio * 100.0f, pa.moving_ratio * 100.0f);
	log += line;
	for (float t : positions) {
		if (!ok)
			break;
		ok = ip.synthesize(t, out, &err) && download(res, res_c);
		if (!ok)
			break;
		renderScene(sp, 1.0f + t, w, h, gt, gc);
		std::vector<uint8_t> blend(l0.size());
		for (size_t i = 0; i < blend.size(); i++)
			blend[i] = static_cast<uint8_t>((1.0f - t) * l0[i] + t * l1[i] + 0.5f);
		const double p_gen = psnr(res, gt, w, h, border);
		const double p_rep = psnr(t < 0.5f ? l0 : l1, gt, w, h, border);
		const double p_rep_other = psnr(t < 0.5f ? l1 : l0, gt, w, h, border);
		const double p_blend = psnr(blend, gt, w, h, border);
		const double p_chroma = psnr(res_c, gc, w, h / 2, border / 2);
		const bool pass = p_gen >= 26.0 && p_gen >= p_blend + 4.0 && p_gen >= std::max(p_rep, p_rep_other) + 4.0
		               && p_chroma >= 30.0;
		snprintf(line, sizeof(line), "t=%.3f generated %.2f dB (chroma %.2f dB) | repeat %.2f dB | blend %.2f dB -> %s\n",
		         t, p_gen, p_chroma, std::max(p_rep, p_rep_other), p_blend, pass ? "ok" : "FAIL");
		log += line;
		if (!pass) {
			ok = false;
			err = "generated frame does not match the expected motion";
		}
	}
	// Protected zone over a static HUD while the whole background pans behind it: the HUD
	// must stay exactly the nearest original, the background inside the zone must still be
	// interpolated (no frozen strip / seam at the zone border).
	if (ok) {
		SceneParams hp = sp;
		hp.fg_size = 0.0f; // background motion only, plus the HUD
		hp.bg_vx = 14.0f;  // fast pan
		hp.bg_vy = 3.0f;
		hp.hud_x = std::floor(w * 0.06f);
		hp.hud_y = std::floor(h * 0.62f);
		hp.hud_w = std::floor(w * 0.16f);
		hp.hud_h = std::floor(h * 0.20f);
		// zone generously larger than the HUD, as a user would draw it
		std::vector<ProtectedZone> zones(1);
		zones[0].x0 = (hp.hud_x - w * 0.04f) / w;
		zones[0].y0 = (hp.hud_y - h * 0.08f) / h;
		zones[0].x1 = (hp.hud_x + hp.hud_w + w * 0.06f) / w;
		zones[0].y1 = (hp.hud_y + hp.hud_h + h * 0.08f) / h;
		std::vector<uint8_t> hl0, hc0, hl1, hc1;
		ip.reset();
		for (int f = 0; ok && f < 3; f++) {
			std::vector<uint8_t> l, c;
			renderScene(hp, static_cast<float>(f), w, h, l, c);
			if (f == 1) { hl0 = l; hc0 = c; }
			if (f == 2) { hl1 = l; hc1 = c; }
			ok = upload(l, c);
			if (ok) {
				pa = ip.pushFrame(in, &err);
				ok = err.empty() && (f < 2 || pa.valid);
			}
		}
		ok = ok && ip.setProtectedZones(zones, &err);
		const int zx0 = static_cast<int>(zones[0].x0 * w), zx1 = static_cast<int>(zones[0].x1 * w);
		const int zy0 = static_cast<int>(zones[0].y0 * h), zy1 = static_cast<int>(zones[0].y1 * h);
		const int feather = static_cast<int>(std::max(8.0f, 24.0f * h / 1080.0f));
		const int hx0 = static_cast<int>(hp.hud_x), hx1 = static_cast<int>(hp.hud_x + hp.hud_w);
		const int hy0 = static_cast<int>(hp.hud_y), hy1 = static_cast<int>(hp.hud_y + hp.hud_h);
		for (float t : { 0.25f, 0.5f, 0.75f }) {
			if (!ok)
				break;
			ok = ip.synthesize(t, out, &err) && download(res, res_c);
			if (!ok)
				break;
			renderScene(hp, 1.0f + t, w, h, gt, gc);
			const std::vector<uint8_t> &nearest = t < 0.5f ? hl0 : hl1;
			// 1) HUD interior (away from its own edge, where the 5x5 window sees the background)
			size_t hud_mismatch = 0;
			for (int y = hy0 + 4; y < hy1 - 4; y++)
				for (int x = hx0 + 4; x < hx1 - 4; x++)
					hud_mismatch += res[static_cast<size_t>(y) * w + x] != nearest[static_cast<size_t>(y) * w + x];
			// 2) background inside the zone (outside the HUD) vs ground truth and vs frozen original
			// 3) band across the zone border vs ground truth (a seam would show up here)
			double se_bg = 0.0, se_bg_frozen = 0.0, se_edge = 0.0;
			size_t n_bg = 0, n_edge = 0;
			for (int y = std::max(zy0 - feather, border); y < std::min(zy1 + feather, h - border); y++)
				for (int x = std::max(zx0 - feather, border); x < std::min(zx1 + feather, w - border); x++) {
					const bool near_hud = x >= hx0 - 6 && x < hx1 + 6 && y >= hy0 - 6 && y < hy1 + 6;
					if (near_hud)
						continue;
					const size_t i = static_cast<size_t>(y) * w + x;
					const double d = static_cast<double>(res[i]) - gt[i];
					const bool inside = x >= zx0 + feather && x < zx1 - feather && y >= zy0 + feather && y < zy1 - feather;
					const bool edge = !inside;
					if (inside) {
						const double df = static_cast<double>(nearest[i]) - gt[i];
						se_bg += d * d;
						se_bg_frozen += df * df;
						n_bg++;
					} else if (edge) {
						se_edge += d * d;
						n_edge++;
					}
				}
			const auto to_psnr = [](double se, size_t n) { return se <= 1e-9 ? 99.0 : 10.0 * log10(255.0 * 255.0 * n / se); };
			const double p_bg = to_psnr(se_bg, n_bg), p_frozen = to_psnr(se_bg_frozen, n_bg), p_edge = to_psnr(se_edge, n_edge);
			const bool pass = hud_mismatch == 0 && p_bg >= 26.0 && p_bg >= p_frozen + 4.0 && p_edge >= 26.0;
			snprintf(line, sizeof(line),
			         "protected zone t=%.2f: HUD %zu px differ | background in zone %.2f dB (frozen would be %.2f dB) | zone border %.2f dB -> %s\n",
			         t, hud_mismatch, p_bg, p_frozen, p_edge, pass ? "ok" : "FAIL");
			log += line;
			if (!pass) {
				ok = false;
				err = "protected zone not applied correctly";
			}
		}
		ip.setProtectedZones({}, nullptr);
	}
	if (y_buf) cu->cuMemFree(y_buf);
	if (c_buf) cu->cuMemFree(c_buf);
	if (oy) cu->cuMemFree(oy);
	if (oc) cu->cuMemFree(oc);
	if (!ok && !err.empty())
		log += "error: " + err + "\n";
	if (report)
		*report = log;
	return ok;
}

} // namespace framegen
