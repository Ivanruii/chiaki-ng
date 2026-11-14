// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// Optical-flow frame interpolator built on the NVIDIA Optical Flow Accelerator (NVOFA) and
// CUDA. Flow is computed once per pair of original frames (forward and backward in a single
// hardware call); any number of intermediate positions can then be synthesized from it.
// This is what makes x3 and x4 possible: NvOFFRUC only produces one frame per input and its
// documentation discourages feeding the same input twice.
//
// The class has no Qt or FFmpeg dependency so it can be exercised by a standalone test.

#ifndef CHIAKI_FRAMEGEN_OFA_INTERPOLATOR_H
#define CHIAKI_FRAMEGEN_OFA_INTERPOLATOR_H

#include "framegen/cuda_dyn.h"

#include <string>
#include <vector>

namespace framegen {

enum class PixelLayout
{
	NV12, // 8-bit luma plane + interleaved 8-bit UV plane (half resolution)
	P010, // same layout with 16-bit samples (10 significant bits in the high bits)
};

struct GpuFrame
{
	CUdeviceptr luma = 0;
	size_t luma_pitch = 0;
	CUdeviceptr chroma = 0;
	size_t chroma_pitch = 0;
};

// Rectangle in normalized video coordinates (0..1, origin top-left).
struct ProtectedZone
{
	float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
};

constexpr int kMaxProtectedZones = 16;

struct PairAnalysis
{
	bool valid = false;            // flow for the current pair is available
	bool scene_cut = false;        // most blocks failed the consistency check
	float unreliable_ratio = 0.0f; // share of blocks with failed forward/backward check
	float moving_ratio = 0.0f;     // share of blocks with motion > 0.75 px
	float flow_ms = 0.0f;          // GPU time of hardware flow + post-processing
};

struct OfaRuntimeInfo
{
	bool available = false;
	std::string reason;          // why it is unavailable
	std::string device_name;
	int cc_major = 0;
	int cc_minor = 0;
	int driver_version = 0;      // CUDA driver API version, e.g. 13030
	unsigned int grid_sizes = 0; // NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES bitmask
};

class OfaInterpolator
{
public:
	OfaInterpolator();
	~OfaInterpolator();
	OfaInterpolator(const OfaInterpolator &) = delete;
	OfaInterpolator &operator=(const OfaInterpolator &) = delete;

	// Checks that CUDA, the optical flow driver library and the embedded kernels work on
	// the CUDA device of `ctx` (or device 0 when ctx is null).
	static OfaRuntimeInfo probe(CUcontext ctx);

	// Allocates all resources for width x height frames. `ctx` must stay alive.
	bool init(CUcontext ctx, int width, int height, PixelLayout layout, std::string *error);
	void destroy();
	bool initialized() const { return of_handle != nullptr; }
	int width() const { return w; }
	int height() const { return h; }
	PixelLayout layout() const { return fmt; }
	CUstream stream() const { return cu_stream; }

	// Forgets the previous frame and the hardware temporal hints (scene change, seek,
	// stream discontinuity, mode change).
	void reset();

	// Uploads a new original frame (device-to-device copy into the history ring). When a
	// previous frame exists, computes forward/backward flow for (previous, new) once.
	// Must be called with `ctx` current or from any thread (it pushes the context itself).
	PairAnalysis pushFrame(const GpuFrame &src, std::string *error);

	// Renders the frame at time t in (0,1) between the last two pushed frames into dst.
	// Work is enqueued on stream(); call synchronize() before the CPU or another API reads dst.
	bool synthesize(float t, const GpuFrame &dst, std::string *error);

	// Low-latency mode: predicts the frame at time 1+s, s in (0,1), after the newest pushed
	// frame from that frame and the motion of the last pair (no wait for the next original).
	// Same stream rules as synthesize().
	bool extrapolate(float s, const GpuFrame &dst, std::string *error);

	bool synchronize(std::string *error);

	// Areas (e.g. a game HUD) where generated frames show the temporally nearest original
	// instead of motion-compensated pixels. Up to kMaxProtectedZones; empty disables.
	bool setProtectedZones(const std::vector<ProtectedZone> &zones, std::string *error);

	// Synthetic-motion self test: renders frames at 1/4..3/4 of a known motion and checks
	// that they match the ground truth far better than repeating or blending the originals.
	// Returns false and fills *report with details when the generated frames are wrong.
	// The low-latency (extrapolation) mode is checked too; its result goes to
	// *extrapolation_ok and does not affect the return value.
	static bool selfTest(CUcontext ctx, std::string *report, int width = 640, int height = 368,
	                     PixelLayout layout = PixelLayout::NV12, bool *extrapolation_ok = nullptr);

private:
	bool ensureMask(std::string *error);

	struct Impl;
	Impl *d;
	CUcontext ctx = nullptr;
	CUstream cu_stream = nullptr;
	void *of_handle = nullptr;
	int w = 0;
	int h = 0;
	PixelLayout fmt = PixelLayout::NV12;
};

} // namespace framegen

#endif // CHIAKI_FRAMEGEN_OFA_INTERPOLATOR_H
