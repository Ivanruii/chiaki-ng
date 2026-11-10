// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// Frame generation stage of the video path:
//
//   FFmpeg (NVDEC, CUDA frames) -> FrameGenerator -> presentFrame() -> pl_queue -> libplacebo
//
// The generator sits after decoding and before presentation (and therefore before the
// Chiaki-ng overlays, which are composited by the renderer). Originals and generated frames
// stay on the GPU: CUDA frames are copied into Vulkan images through FFmpeg's CUDA/Vulkan
// external-memory interop, so libplacebo maps them without a CPU round trip.
//
// Timing: when original N arrives, the frames between N-1 and N are generated at
// k/multiplier of the real interval and released at evenly spaced wall-clock times, followed
// by original N. Originals are therefore delayed by (multiplier-1)/multiplier of a source
// frame interval. Generated frames that would be late, or that the display/renderer cannot
// take, are dropped; originals are never dropped by this stage.

#ifndef CHIAKI_FRAMEGEN_FRAMEGENERATOR_H
#define CHIAKI_FRAMEGEN_FRAMEGENERATOR_H

#include <chiaki/ffmpegdecoder.h>

#include <QString>
#include <QVariantList>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

namespace framegen {

class OfaInterpolator;

constexpr int kMaxMultiplier = 4;

struct ModeAvailability
{
	bool enabled = false;
	QString reason;  // why the mode is disabled (empty when enabled)
	QString backend; // implementation used when enabled
};

struct Capabilities
{
	bool probed = false;
	QString gpu;              // GPU used by the session
	QString summary;          // one line for the settings page
	QString nvfruc;           // NvFRUC library status
	QString self_test_report; // per-position quality numbers of the GPU self test
	ModeAvailability modes[kMaxMultiplier + 1]; // index = multiplier (2..4)

	QVariantList toVariantList() const; // [{multiplier, enabled, reason, backend}, ...] for QML
};

struct ProbeInput
{
	bool vulkan_renderer = false;
	AVBufferRef *vulkan_device = nullptr; // FFmpeg Vulkan device sharing libplacebo's VkDevice
	QStringList vulkan_device_extensions;
	bool cuda_decoder_available = false;  // FFmpeg built with NVDEC and h264/hevc CUDA hwconfig
};

// Runs every check needed to enable the multipliers, including a GPU self test that renders
// generated frames for a known motion and compares them with the ground truth. Slow-ish
// (first PTX JIT); call from a worker thread. On success *cuda_device receives a CUDA device
// on the same GPU as the Vulkan renderer, to be used by the decoder and the generator.
Capabilities probeCapabilities(const ProbeInput &input, AVBufferRef **cuda_device);

struct FrameGeneratorStats
{
	uint64_t originals = 0;
	uint64_t generated = 0;       // generated frames delivered to the renderer
	uint64_t dropped_late = 0;    // generated frames dropped because they were late / superseded
	uint64_t dropped_backlog = 0; // generated frames dropped because the renderer was behind
	uint64_t skipped_pairs = 0;   // pairs without generation (scene cut, discontinuity)
	uint64_t resets = 0;
	double last_pair_ms = 0.0;    // CPU wall time spent on the last pair (flow + synthesis + interop)
};

class FrameGenerator
{
public:
	using DeliverFn = std::function<void(ChiakiFfmpegFrame frame, int32_t frames_lost)>;
	using BacklogFn = std::function<bool()>;
	using StatusFn = std::function<void(const QString &status)>;

	struct Config
	{
		int multiplier = 2;
		double source_fps = 60.0;
		double display_hz = 0.0;              // 0 = unknown
		AVBufferRef *cuda_device = nullptr;   // from probeCapabilities()
		AVBufferRef *vulkan_device = nullptr; // FFmpeg Vulkan device of the renderer
		DeliverFn deliver;                    // called from the generator thread
		BacklogFn renderer_backlogged;        // true while the renderer has frames waiting
		StatusFn status_changed;              // called from the generator thread
	};

	FrameGenerator();
	~FrameGenerator();
	FrameGenerator(const FrameGenerator &) = delete;
	FrameGenerator &operator=(const FrameGenerator &) = delete;

	bool start(const Config &config, QString *error);
	void stop();
	bool running() const { return worker.joinable(); }

	// Takes ownership of frame.frame. Thread-safe; called from the decoder frame thread.
	void submit(ChiakiFfmpegFrame frame, int32_t frames_lost);

	// 1 = pass-through. Takes effect on the next original and resets the interpolator.
	void setMultiplier(int multiplier);
	int multiplier() const { return requested_multiplier.load(); }

	FrameGeneratorStats stats() const;
	QString status() const;

private:
	struct Input
	{
		ChiakiFfmpegFrame frame;
		int32_t frames_lost = 0;
		int64_t arrival_us = 0;
	};
	struct Output
	{
		ChiakiFfmpegFrame frame;
		int32_t frames_lost = 0;
		int64_t due_us = 0;
		bool synthetic = false;
	};

	void run();
	void process(Input &in);
	void emitDue(int64_t now_us, bool flush);
	bool ensureEngine(const AVFrame *src, QString *error);
	AVFrame *toPresentable(AVFrame *cuda_frame, QString *error);
	void deliver(Output &out);
	void setStatus(const QString &status);
	void disableGeneration(const QString &reason);

	Config cfg;
	std::thread worker;
	mutable std::mutex mutex;
	std::condition_variable cond;
	bool stopping = false;
	std::deque<Input> inputs;
	std::deque<Output> outputs;
	std::atomic<int> requested_multiplier{1};
	int active_multiplier = 1;

	// worker-thread state
	std::unique_ptr<OfaInterpolator> engine;
	AVBufferRef *cuda_frames = nullptr;   // pool for generated CUDA frames
	AVBufferRef *vulkan_frames = nullptr; // pool for presentable Vulkan frames
	int frames_w = 0, frames_h = 0, frames_sw_format = -1;
	bool interop_gpu = true;              // false -> CPU copy fallback
	bool generation_failed = false;
	bool have_prev = false;
	double prev_pts = 0.0;
	int64_t last_emit_us = 0;

	mutable std::mutex stats_mutex;
	FrameGeneratorStats st;
	QString status_text;
};

// Free-standing helper: textual "60 → 120 FPS" style description.
QString targetFpsText(double source_fps, int multiplier);

} // namespace framegen

#endif // CHIAKI_FRAMEGEN_FRAMEGENERATOR_H
