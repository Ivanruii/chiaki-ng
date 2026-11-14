// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// Standalone check of the frame generation engine on the local NVIDIA GPU.
// Build (Linux/WSL, no CUDA toolkit needed):
//   g++ -std=c++17 -O2 -Igui/include -Igui/src/framegen gui/src/framegen/cuda_dyn.cpp
//       gui/src/framegen/ofa_interpolator.cpp gui/src/framegen/tests/ofa_selftest.cpp -ldl
// or configure CMake with -DCHIAKI_FRAMEGEN_SELFTEST=ON.

#include "framegen/ofa_interpolator.h"

#include <chrono>
#include <string.h>
#include <stdio.h>
#include <vector>

using namespace framegen;

static bool benchmark(int w, int h, PixelLayout layout, int multiplier, bool extrapolate = false)
{
	const CudaApi *cu = cudaApi();
	CUdevice dev = 0;
	CUcontext ctx = nullptr;
	cu->cuDeviceGet(&dev, 0);
	cu->cuDevicePrimaryCtxRetain(&ctx, dev);
	OfaInterpolator ip;
	std::string err;
	if (!ip.init(ctx, w, h, layout, &err)) {
		printf("benchmark init failed: %s\n", err.c_str());
		return false;
	}
	CudaContextScope scope(ctx);
	const size_t bpp = layout == PixelLayout::P010 ? 2 : 1;
	// Textured frames panning 4,2 px per frame (flat frames would be rejected as unreliable)
	const int frames = 8, margin = 4 * frames + 8;
	const int tw = w + margin, th = h + margin;
	std::vector<uint8_t> tex(static_cast<size_t>(tw) * th), blurred(tex.size());
	uint32_t seed = 12345u;
	for (auto &v : tex) {
		seed = seed * 1664525u + 1013904223u;
		v = static_cast<uint8_t>(seed >> 24);
	}
	for (int y = 0; y < th; y++)
		for (int x = 0; x < tw; x++) {
			int sum = 0, n = 0;
			for (int dy = -2; dy <= 2; dy++)
				for (int dx = -2; dx <= 2; dx++) {
					const int sx = x + dx, sy = y + dy;
					if (sx >= 0 && sy >= 0 && sx < tw && sy < th) {
						sum += tex[static_cast<size_t>(sy) * tw + sx];
						n++;
					}
				}
			blurred[static_cast<size_t>(y) * tw + x] = static_cast<uint8_t>(sum / n);
		}
	std::vector<CUdeviceptr> src_y(frames), src_c(frames);
	size_t syp = 0, scp = 0, dyp = 0, dcp = 0;
	CUdeviceptr dst_y = 0, dst_c = 0;
	for (int i = 0; i < frames; i++) {
		std::vector<uint8_t> luma(static_cast<size_t>(w) * h * bpp), chroma(static_cast<size_t>(w) * (h / 2) * bpp);
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				const uint8_t v = blurred[static_cast<size_t>(y + 2 * i) * tw + x + 4 * i];
				if (bpp == 2) {
					const uint16_t v16 = static_cast<uint16_t>(v << 8);
					memcpy(&luma[(static_cast<size_t>(y) * w + x) * 2], &v16, 2);
				} else {
					luma[static_cast<size_t>(y) * w + x] = v;
				}
			}
		for (size_t k = 0; k < chroma.size(); k++)
			chroma[k] = bpp == 2 && (k & 1) == 0 ? 0 : 128;
		cu->cuMemAllocPitch(&src_y[i], &syp, w * bpp, h, 16);
		cu->cuMemAllocPitch(&src_c[i], &scp, w * bpp, h / 2, 16);
		CUDA_MEMCPY2D c = {};
		c.srcMemoryType = CU_MEMORYTYPE_HOST;
		c.srcHost = luma.data();
		c.srcPitch = w * bpp;
		c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		c.dstDevice = src_y[i];
		c.dstPitch = syp;
		c.WidthInBytes = w * bpp;
		c.Height = h;
		cu->cuMemcpy2DAsync(&c, ip.stream());
		c.srcHost = chroma.data();
		c.dstDevice = src_c[i];
		c.dstPitch = scp;
		c.Height = h / 2;
		cu->cuMemcpy2DAsync(&c, ip.stream());
		cu->cuStreamSynchronize(ip.stream());
	}
	cu->cuMemAllocPitch(&dst_y, &dyp, w * bpp, h, 16);
	cu->cuMemAllocPitch(&dst_c, &dcp, w * bpp, h / 2, 16);
	GpuFrame out{ dst_y, dyp, dst_c, dcp };
	const int iterations = 240;
	double total_flow_ms = 0.0, total_wall_ms = 0.0, worst_wall_ms = 0.0;
	int measured = 0;
	for (int i = 0; i < iterations; i++) {
		GpuFrame in{ src_y[i % frames], syp, src_c[i % frames], scp };
		const auto t0 = std::chrono::steady_clock::now();
		PairAnalysis pa = ip.pushFrame(in, &err);
		if (!err.empty()) {
			printf("push failed: %s\n", err.c_str());
			return false;
		}
		bool synthesized = false;
		if (pa.valid) {
			for (int j = 1; j < multiplier; j++)
				if (!(extrapolate ? ip.extrapolate(static_cast<float>(j) / multiplier, out, &err)
				                  : ip.synthesize(static_cast<float>(j) / multiplier, out, &err))) {
					printf("synth failed: %s\n", err.c_str());
					return false;
				}
			ip.synchronize(&err);
			synthesized = true;
		}
		const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
		if (i >= 10 && synthesized) {
			total_flow_ms += pa.flow_ms;
			total_wall_ms += wall;
			worst_wall_ms = std::max(worst_wall_ms, wall);
			measured++;
		}
	}
	printf("%dx%d %s x%d%s: flow(GPU) %.2f ms avg, pair total(wall) %.2f ms avg / %.2f ms worst over %d pairs\n",
	       w, h, layout == PixelLayout::P010 ? "P010" : "NV12", multiplier, extrapolate ? " low-latency" : "",
	       total_flow_ms / std::max(measured, 1), total_wall_ms / std::max(measured, 1), worst_wall_ms, measured);
	for (int i = 0; i < frames; i++) {
		cu->cuMemFree(src_y[i]);
		cu->cuMemFree(src_c[i]);
	}
	cu->cuMemFree(dst_y);
	cu->cuMemFree(dst_c);
	return measured > iterations / 2;
}

int main()
{
	OfaRuntimeInfo info = OfaInterpolator::probe(nullptr);
	printf("probe: available=%d device=\"%s\" cc=%d.%d cuda_driver=%d grid_mask=0x%x reason=\"%s\"\n",
	       info.available, info.device_name.c_str(), info.cc_major, info.cc_minor, info.driver_version,
	       info.grid_sizes, info.reason.c_str());
	if (!info.available)
		return 2;

	const CudaApi *cu = cudaApi();
	CUdevice dev = 0;
	CUcontext ctx = nullptr;
	cu->cuDeviceGet(&dev, 0);
	cu->cuDevicePrimaryCtxRetain(&ctx, dev);

	int failures = 0;
	struct { int w, h; PixelLayout layout; const char *name; } cases[] = {
		{ 640, 368, PixelLayout::NV12, "NV12" },
		{ 1920, 1080, PixelLayout::NV12, "NV12" },
		{ 1920, 1080, PixelLayout::P010, "P010" },
	};
	for (auto &c : cases) {
		std::string report;
		bool ext_ok = false;
		const bool ok = OfaInterpolator::selfTest(ctx, &report, c.w, c.h, c.layout, &ext_ok);
		printf("self test %dx%d %s: %s, low-latency %s\n%s", c.w, c.h, c.name, ok ? "PASS" : "FAIL",
		       ext_ok ? "PASS" : "FAIL", report.c_str());
		failures += ok && ext_ok ? 0 : 1;
	}
	for (int m = 2; m <= 4; m++)
		failures += benchmark(1920, 1080, PixelLayout::NV12, m) ? 0 : 1;
	failures += benchmark(1920, 1080, PixelLayout::P010, 4) ? 0 : 1;
	failures += benchmark(1920, 1080, PixelLayout::NV12, 4, true) ? 0 : 1;
	failures += benchmark(3840, 2160, PixelLayout::NV12, 2) ? 0 : 1;
	printf("%s\n", failures ? "FAILED" : "ALL PASSED");
	return failures ? 1 : 0;
}
