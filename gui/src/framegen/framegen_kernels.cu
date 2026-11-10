// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// CUDA kernels for optical-flow based frame interpolation.
//
// This file is NOT compiled by the regular build. scripts/framegen/gen_ptx.py compiles it
// to PTX with NVRTC and writes gui/src/framegen/framegen_kernels_ptx.h, which is embedded
// in chiaki-ng and JIT-compiled by the NVIDIA driver for whatever GPU is present
// (Turing .. Blackwell and newer). Re-run the script after editing this file.
//
// Pipeline for one pair of original frames (I0 = previous, I1 = current):
//   1. NVOFA computes block flow F01 (I0 -> I1) and F10 (I1 -> I0) once per pair.
//   2. fg_unpack_flow converts the S10.5 hardware vectors to float pixels.
//   3. fg_median_flow removes outlier vectors (3x3 vector median).
//   4. fg_consistency derives per-block visibility from forward/backward agreement and
//      counts unreliable blocks (scene-cut detection).
//   5. fg_synth_* renders the frame at time t in (0,1) by solving, per output pixel x,
//      p + t*F01(p) = x and q + (1-t)*F10(q) = x with a fixed-point iteration, sampling
//      I0(p) and I1(q) and blending them with visibility-weighted temporal weights.
// Steps 1-4 run once per pair; step 5 runs once per requested position (1/2 for x2,
// 1/3 and 2/3 for x3, 1/4, 1/2 and 3/4 for x4), reusing the same flow.

struct FgFlow { float x, y; };

__device__ __forceinline__ float fg_clampf(float v, float lo, float hi)
{
	return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ FgFlow fg_load_flow(const FgFlow *f, int bw, int bh, float u, float v)
{
	u = fg_clampf(u, 0.0f, (float)(bw - 1));
	v = fg_clampf(v, 0.0f, (float)(bh - 1));
	const int x0 = (int)u;
	const int y0 = (int)v;
	const int x1 = min(x0 + 1, bw - 1);
	const int y1 = min(y0 + 1, bh - 1);
	const float fx = u - (float)x0;
	const float fy = v - (float)y0;
	const FgFlow a = f[y0 * bw + x0];
	const FgFlow b = f[y0 * bw + x1];
	const FgFlow c = f[y1 * bw + x0];
	const FgFlow d = f[y1 * bw + x1];
	FgFlow r;
	r.x = (a.x * (1.0f - fx) + b.x * fx) * (1.0f - fy) + (c.x * (1.0f - fx) + d.x * fx) * fy;
	r.y = (a.y * (1.0f - fx) + b.y * fx) * (1.0f - fy) + (c.y * (1.0f - fx) + d.y * fx) * fy;
	return r;
}

__device__ __forceinline__ float fg_load_scalar(const float *f, int bw, int bh, float u, float v)
{
	u = fg_clampf(u, 0.0f, (float)(bw - 1));
	v = fg_clampf(v, 0.0f, (float)(bh - 1));
	const int x0 = (int)u;
	const int y0 = (int)v;
	const int x1 = min(x0 + 1, bw - 1);
	const int y1 = min(y0 + 1, bh - 1);
	const float fx = u - (float)x0;
	const float fy = v - (float)y0;
	return (f[y0 * bw + x0] * (1.0f - fx) + f[y0 * bw + x1] * fx) * (1.0f - fy)
	     + (f[y1 * bw + x0] * (1.0f - fx) + f[y1 * bw + x1] * fx) * fy;
}

// Continuous pixel coordinate (pixel centres at i + 0.5) -> block-grid sample coordinate.
__device__ __forceinline__ float fg_to_grid(float c, float grid)
{
	return c / grid - 0.5f;
}

template<typename T>
__device__ __forceinline__ float fg_sample1(const unsigned char *plane, size_t pitch, int w, int h, float x, float y)
{
	// x, y are continuous coordinates; convert to texel-centre space
	x = fg_clampf(x - 0.5f, 0.0f, (float)(w - 1));
	y = fg_clampf(y - 0.5f, 0.0f, (float)(h - 1));
	const int x0 = (int)x;
	const int y0 = (int)y;
	const int x1 = min(x0 + 1, w - 1);
	const int y1 = min(y0 + 1, h - 1);
	const float fx = x - (float)x0;
	const float fy = y - (float)y0;
	const T *r0 = reinterpret_cast<const T *>(plane + (size_t)y0 * pitch);
	const T *r1 = reinterpret_cast<const T *>(plane + (size_t)y1 * pitch);
	return ((float)r0[x0] * (1.0f - fx) + (float)r0[x1] * fx) * (1.0f - fy)
	     + ((float)r1[x0] * (1.0f - fx) + (float)r1[x1] * fx) * fy;
}

template<typename T>
__device__ __forceinline__ void fg_sample2(const unsigned char *plane, size_t pitch, int w, int h, float x, float y, float &o0, float &o1)
{
	x = fg_clampf(x - 0.5f, 0.0f, (float)(w - 1));
	y = fg_clampf(y - 0.5f, 0.0f, (float)(h - 1));
	const int x0 = (int)x;
	const int y0 = (int)y;
	const int x1 = min(x0 + 1, w - 1);
	const int y1 = min(y0 + 1, h - 1);
	const float fx = x - (float)x0;
	const float fy = y - (float)y0;
	const T *r0 = reinterpret_cast<const T *>(plane + (size_t)y0 * pitch);
	const T *r1 = reinterpret_cast<const T *>(plane + (size_t)y1 * pitch);
	const float w00 = (1.0f - fx) * (1.0f - fy), w01 = fx * (1.0f - fy);
	const float w10 = (1.0f - fx) * fy, w11 = fx * fy;
	o0 = (float)r0[2 * x0] * w00 + (float)r0[2 * x1] * w01 + (float)r1[2 * x0] * w10 + (float)r1[2 * x1] * w11;
	o1 = (float)r0[2 * x0 + 1] * w00 + (float)r0[2 * x1 + 1] * w01 + (float)r1[2 * x0 + 1] * w10 + (float)r1[2 * x1 + 1] * w11;
}

// Solve p + t*F(p) = x by fixed-point iteration (x in continuous luma coordinates).
__device__ __forceinline__ void fg_trace(const FgFlow *flow, int bw, int bh, float grid, float t,
                                         float x, float y, float &px, float &py)
{
	px = x;
	py = y;
#pragma unroll
	for (int i = 0; i < 4; i++) {
		const FgFlow f = fg_load_flow(flow, bw, bh, fg_to_grid(px, grid), fg_to_grid(py, grid));
		px = x - t * f.x;
		py = y - t * f.y;
	}
}

// Temporal weights for the two warped samples. Visibility close to zero means the flow at
// that source position failed the forward/backward check (occlusion or bad match).
__device__ __forceinline__ void fg_weights(float t, float v0, float v1, float &w0, float &w1)
{
	w0 = (1.0f - t) * (v0 + 0.02f);
	w1 = t * (v1 + 0.02f);
	const float s = w0 + w1;
	w0 /= s;
	w1 /= s;
}

template<typename T>
__device__ __forceinline__ T fg_store(float v, float maxv)
{
	v = fg_clampf(v + 0.5f, 0.0f, maxv);
	return (T)v;
}

extern "C" {

// S10.5 hardware vectors -> float pixels. One thread per block.
__global__ void fg_unpack_flow(const short2 *fwd, unsigned int fwd_pitch, const short2 *bwd, unsigned int bwd_pitch,
                               int bw, int bh, FgFlow *f01, FgFlow *f10)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= bw || y >= bh)
		return;
	const short2 a = reinterpret_cast<const short2 *>(reinterpret_cast<const unsigned char *>(fwd) + (size_t)y * fwd_pitch)[x];
	const short2 b = reinterpret_cast<const short2 *>(reinterpret_cast<const unsigned char *>(bwd) + (size_t)y * bwd_pitch)[x];
	FgFlow fa; fa.x = (float)a.x * (1.0f / 32.0f); fa.y = (float)a.y * (1.0f / 32.0f);
	FgFlow fb; fb.x = (float)b.x * (1.0f / 32.0f); fb.y = (float)b.y * (1.0f / 32.0f);
	f01[y * bw + x] = fa;
	f10[y * bw + x] = fb;
}

// 3x3 vector median: picks the neighbour vector with the smallest summed L1 distance.
__global__ void fg_median_flow(const FgFlow *src, FgFlow *dst, int bw, int bh)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= bw || y >= bh)
		return;
	FgFlow v[9];
	int n = 0;
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++) {
			const int sx = min(max(x + dx, 0), bw - 1);
			const int sy = min(max(y + dy, 0), bh - 1);
			v[n++] = src[sy * bw + sx];
		}
	int best = 4;
	float best_cost = 3.4e38f;
	for (int i = 0; i < 9; i++) {
		float cost = 0.0f;
		for (int j = 0; j < 9; j++)
			cost += fabsf(v[i].x - v[j].x) + fabsf(v[i].y - v[j].y);
		if (cost < best_cost) {
			best_cost = cost;
			best = i;
		}
	}
	dst[y * bw + x] = v[best];
}

// Forward/backward consistency -> visibility in [0,1] for each block of both frames.
// counters[0] += blocks of I0 with low visibility, counters[1] += moving blocks.
__global__ void fg_consistency(const FgFlow *f01, const FgFlow *f10, int bw, int bh, float grid,
                               float *vis0, float *vis1, unsigned int *counters)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= bw || y >= bh)
		return;
	const float cx = ((float)x + 0.5f) * grid;
	const float cy = ((float)y + 0.5f) * grid;

	const FgFlow a = f01[y * bw + x];
	const FgFlow ab = fg_load_flow(f10, bw, bh, fg_to_grid(cx + a.x, grid), fg_to_grid(cy + a.y, grid));
	const float e0 = fabsf(a.x + ab.x) + fabsf(a.y + ab.y);
	const float tol0 = 1.0f + 0.08f * (fabsf(a.x) + fabsf(a.y));
	const float r0 = e0 / tol0;
	const float v0 = 1.0f / (1.0f + r0 * r0);

	const FgFlow b = f10[y * bw + x];
	const FgFlow ba = fg_load_flow(f01, bw, bh, fg_to_grid(cx + b.x, grid), fg_to_grid(cy + b.y, grid));
	const float e1 = fabsf(b.x + ba.x) + fabsf(b.y + ba.y);
	const float tol1 = 1.0f + 0.08f * (fabsf(b.x) + fabsf(b.y));
	const float r1 = e1 / tol1;
	const float v1 = 1.0f / (1.0f + r1 * r1);

	vis0[y * bw + x] = v0;
	vis1[y * bw + x] = v1;
	if (v0 < 0.2f)
		atomicAdd(&counters[0], 1u);
	if (fabsf(a.x) + fabsf(a.y) > 0.75f)
		atomicAdd(&counters[1], 1u);
}

#define FG_SYNTH_LUMA(NAME, T, MAXV) \
__global__ void NAME(const unsigned char *y0p, unsigned int y0_pitch, const unsigned char *y1p, unsigned int y1_pitch, \
                     unsigned char *dst, unsigned int dst_pitch, int w, int h, \
                     const FgFlow *f01, const FgFlow *f10, const float *vis0, const float *vis1, \
                     int bw, int bh, float grid, float t) \
{ \
	const int x = blockIdx.x * blockDim.x + threadIdx.x; \
	const int y = blockIdx.y * blockDim.y + threadIdx.y; \
	if (x >= w || y >= h) \
		return; \
	const float cx = (float)x + 0.5f; \
	const float cy = (float)y + 0.5f; \
	float px, py, qx, qy; \
	fg_trace(f01, bw, bh, grid, t, cx, cy, px, py); \
	fg_trace(f10, bw, bh, grid, 1.0f - t, cx, cy, qx, qy); \
	const float v0 = fg_load_scalar(vis0, bw, bh, fg_to_grid(px, grid), fg_to_grid(py, grid)); \
	const float v1 = fg_load_scalar(vis1, bw, bh, fg_to_grid(qx, grid), fg_to_grid(qy, grid)); \
	float w0, w1; \
	fg_weights(t, v0, v1, w0, w1); \
	const float s0 = fg_sample1<T>(y0p, y0_pitch, w, h, px, py); \
	const float s1 = fg_sample1<T>(y1p, y1_pitch, w, h, qx, qy); \
	reinterpret_cast<T *>(dst + (size_t)y * dst_pitch)[x] = fg_store<T>(w0 * s0 + w1 * s1, MAXV); \
}

#define FG_SYNTH_CHROMA(NAME, T, MAXV) \
__global__ void NAME(const unsigned char *uv0p, unsigned int uv0_pitch, const unsigned char *uv1p, unsigned int uv1_pitch, \
                     unsigned char *dst, unsigned int dst_pitch, int cw, int ch, \
                     const FgFlow *f01, const FgFlow *f10, const float *vis0, const float *vis1, \
                     int bw, int bh, float grid, float t) \
{ \
	const int x = blockIdx.x * blockDim.x + threadIdx.x; \
	const int y = blockIdx.y * blockDim.y + threadIdx.y; \
	if (x >= cw || y >= ch) \
		return; \
	/* chroma sample centre in luma coordinates */ \
	const float lx = ((float)x + 0.5f) * 2.0f; \
	const float ly = ((float)y + 0.5f) * 2.0f; \
	float px, py, qx, qy; \
	fg_trace(f01, bw, bh, grid, t, lx, ly, px, py); \
	fg_trace(f10, bw, bh, grid, 1.0f - t, lx, ly, qx, qy); \
	const float v0 = fg_load_scalar(vis0, bw, bh, fg_to_grid(px, grid), fg_to_grid(py, grid)); \
	const float v1 = fg_load_scalar(vis1, bw, bh, fg_to_grid(qx, grid), fg_to_grid(qy, grid)); \
	float w0, w1; \
	fg_weights(t, v0, v1, w0, w1); \
	float u0, vv0, u1, vv1; \
	fg_sample2<T>(uv0p, uv0_pitch, cw, ch, px * 0.5f, py * 0.5f, u0, vv0); \
	fg_sample2<T>(uv1p, uv1_pitch, cw, ch, qx * 0.5f, qy * 0.5f, u1, vv1); \
	T *row = reinterpret_cast<T *>(dst + (size_t)y * dst_pitch); \
	row[2 * x] = fg_store<T>(w0 * u0 + w1 * u1, MAXV); \
	row[2 * x + 1] = fg_store<T>(w0 * vv0 + w1 * vv1, MAXV); \
}

FG_SYNTH_LUMA(fg_synth_luma8, unsigned char, 255.0f)
FG_SYNTH_CHROMA(fg_synth_chroma8, unsigned char, 255.0f)
FG_SYNTH_LUMA(fg_synth_luma16, unsigned short, 65535.0f)
FG_SYNTH_CHROMA(fg_synth_chroma16, unsigned short, 65535.0f)

// P010 luma (10 bits in the high bits of 16) -> 8-bit grayscale for the optical flow engine.
__global__ void fg_luma16_to_gray8(const unsigned char *src, unsigned int src_pitch, unsigned char *dst, unsigned int dst_pitch, int w, int h)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h)
		return;
	const unsigned short v = reinterpret_cast<const unsigned short *>(src + (size_t)y * src_pitch)[x];
	dst[(size_t)y * dst_pitch + x] = (unsigned char)(v >> 8);
}

} // extern "C"
