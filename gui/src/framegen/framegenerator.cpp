// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

// cuda_dyn.h first: it provides the CUDA types FFmpeg's hwcontext_cuda.h expects
#include "framegen/cuda_dyn.h"
#include "framegen/framegenerator.h"
#include "framegen/ofa_interpolator.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QLoggingCategory>
#include <QSet>
#include <QVariantMap>

#include <algorithm>
#include <chrono>
#include <vector>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixdesc.h>
#include <libavutil/version.h>
}

Q_LOGGING_CATEGORY(chiakiFrameGen, "chiaki.framegen", QtInfoMsg);

namespace framegen {

namespace {

int64_t nowUs()
{
	return std::chrono::duration_cast<std::chrono::microseconds>(
		std::chrono::steady_clock::now().time_since_epoch()).count();
}

QString tr(const char *text)
{
	return QCoreApplication::translate("FrameGeneration", text);
}

const char *kBackendName = "NVIDIA Optical Flow (NVOFA) + CUDA warp";

// Inspects the CUDA fat binaries embedded in NvOFFRUC and reports which GPU architectures
// they can run on. NvOFFRUC ships pre-compiled kernels only, so a GPU whose compute
// capability is not covered (and no PTX for JIT) cannot execute it.
QString nvfrucStatus(int cc_major, int cc_minor)
{
#ifdef _WIN32
	const QString name = QStringLiteral("NvOFFRUC.dll");
#else
	const QString name = QStringLiteral("libNvOFFRUC.so");
#endif
	const QString path = QDir(QCoreApplication::applicationDirPath()).filePath(name);
	QFile file(path);
	if (!file.exists())
		return tr("NvOFFRUC not installed (%1 not found next to chiaki-ng)").arg(name);
	if (!file.open(QIODevice::ReadOnly))
		return tr("NvOFFRUC found but unreadable");
	const QByteArray data = file.readAll();
	QSet<int> sass, ptx;
	const char magic[4] = { '\x50', '\xed', '\x55', '\xba' };
	for (qsizetype i = data.indexOf(QByteArray(magic, 4)); i >= 0 && i + 16 <= data.size(); i = data.indexOf(QByteArray(magic, 4), i + 4)) {
		const auto rd16 = [&](qsizetype o) { return static_cast<quint16>(static_cast<quint8>(data[o]) | static_cast<quint8>(data[o + 1]) << 8); };
		const auto rd32 = [&](qsizetype o) { return static_cast<quint32>(rd16(o)) | static_cast<quint32>(rd16(o + 2)) << 16; };
		const auto rd64 = [&](qsizetype o) { return static_cast<quint64>(rd32(o)) | static_cast<quint64>(rd32(o + 4)) << 32; };
		const qsizetype header = rd16(i + 6);
		const qsizetype end = i + header + static_cast<qsizetype>(rd64(i + 8));
		for (qsizetype p = i + header; p + 32 <= end && p + 32 <= data.size();) {
			const int kind = rd16(p);
			const qsizetype entry_header = rd32(p + 4);
			const qsizetype size = static_cast<qsizetype>(rd64(p + 8));
			const int arch = static_cast<int>(rd32(p + 28));
			(kind == 1 ? ptx : sass).insert(arch);
			if (entry_header <= 0)
				break;
			p += entry_header + size;
		}
	}
	const int cc = cc_major * 10 + cc_minor;
	bool compatible = false;
	for (int a : sass)
		compatible |= a / 10 == cc_major && a <= cc;
	for (int a : ptx)
		compatible |= a <= cc;
	QStringList archs;
	QList<int> all = (sass + ptx).values();
	std::sort(all.begin(), all.end());
	for (int a : all)
		archs << QStringLiteral("sm_%1%2").arg(a).arg(ptx.contains(a) ? QStringLiteral(" (PTX)") : QString());
	if (compatible)
		return tr("NvOFFRUC found and has code for this GPU (%1); x2 uses the NVOFA engine, which also provides x3/x4").arg(archs.join(QStringLiteral(", ")));
	return tr("NvOFFRUC found but unusable on this GPU: it only contains code for %1 and this GPU is sm_%2")
		.arg(archs.isEmpty() ? tr("unknown architectures") : archs.join(QStringLiteral(", ")))
		.arg(cc);
}

bool isSupportedSwFormat(int fmt)
{
	return fmt == AV_PIX_FMT_NV12 || fmt == AV_PIX_FMT_P010LE;
}

} // namespace

QString targetFpsText(double source_fps, int multiplier)
{
	const int src = qRound(source_fps);
	if (multiplier <= 1)
		return QStringLiteral("%1 FPS").arg(src);
	return QStringLiteral("%1 → %2 FPS").arg(src).arg(src * multiplier);
}

QVariantList Capabilities::toVariantList() const
{
	QVariantList list;
	for (int m = 1; m <= kMaxMultiplier; m++) {
		QVariantMap entry;
		entry[QStringLiteral("multiplier")] = m;
		entry[QStringLiteral("enabled")] = m == 1 ? true : modes[m].enabled;
		entry[QStringLiteral("reason")] = m == 1 ? QString() : modes[m].reason;
		entry[QStringLiteral("backend")] = m == 1 ? QString() : modes[m].backend;
		list << entry;
	}
	return list;
}

Capabilities probeCapabilities(const ProbeInput &input, AVBufferRef **cuda_device)
{
	Capabilities caps;
	caps.probed = true;
	auto disableAll = [&](const QString &reason) {
		for (int m = 2; m <= kMaxMultiplier; m++)
			caps.modes[m] = { false, reason, QString() };
		caps.summary = reason;
		caps.low_latency_available = false;
		caps.low_latency_reason = reason;
		return caps;
	};
	if (cuda_device)
		*cuda_device = nullptr;

	if (!input.vulkan_renderer)
		return disableAll(tr("Requires the Vulkan renderer backend"));
	if (!input.vulkan_device)
		return disableAll(tr("The Vulkan device could not be shared with FFmpeg"));
#ifdef _WIN32
	const QStringList needed = { QStringLiteral("VK_KHR_external_memory_win32"), QStringLiteral("VK_KHR_external_semaphore_win32") };
#else
	const QStringList needed = { QStringLiteral("VK_KHR_external_memory_fd"), QStringLiteral("VK_KHR_external_semaphore_fd") };
#endif
	for (const QString &ext : needed)
		if (!input.vulkan_device_extensions.contains(ext))
			return disableAll(tr("Vulkan driver lacks %1 (needed to hand CUDA frames to the renderer)").arg(ext));
	if (!input.cuda_decoder_available)
		return disableAll(tr("FFmpeg build has no NVDEC (CUDA) decoder"));

	AVBufferRef *cuda = nullptr;
	int err = av_hwdevice_ctx_create_derived(&cuda, AV_HWDEVICE_TYPE_CUDA, input.vulkan_device, 0);
	if (err < 0 || !cuda)
		return disableAll(tr("The GPU used by the session is not an NVIDIA CUDA device"));
	auto *cuda_hw = reinterpret_cast<AVCUDADeviceContext *>(reinterpret_cast<AVHWDeviceContext *>(cuda->data)->hwctx);

	OfaRuntimeInfo info = OfaInterpolator::probe(cuda_hw->cuda_ctx);
	caps.gpu = info.device_name.empty() ? QString()
		: QStringLiteral("%1 (sm_%2%3, CUDA driver %4.%5)")
			.arg(QString::fromStdString(info.device_name)).arg(info.cc_major).arg(info.cc_minor)
			.arg(info.driver_version / 1000).arg((info.driver_version % 1000) / 10);
	caps.nvfruc = info.cc_major > 0 ? nvfrucStatus(info.cc_major, info.cc_minor) : QString();
	if (!info.available) {
		av_buffer_unref(&cuda);
		return disableAll(QString::fromStdString(info.reason));
	}

	std::string report;
	bool extrapolation_ok = false;
	const bool test_ok = OfaInterpolator::selfTest(cuda_hw->cuda_ctx, &report, 640, 368, PixelLayout::NV12, &extrapolation_ok);
	caps.self_test_report = QString::fromStdString(report).trimmed();
	qCInfo(chiakiFrameGen).noquote() << "Frame generation GPU self test" << (test_ok ? "passed" : "FAILED")
	                                 << "on" << caps.gpu << "\n" << caps.self_test_report;
	if (!test_ok) {
		av_buffer_unref(&cuda);
		return disableAll(tr("GPU self test failed: generated frames did not match the expected motion"));
	}
	for (int m = 2; m <= kMaxMultiplier; m++)
		caps.modes[m] = { true, QString(), QString::fromUtf8(kBackendName) };
	caps.summary = tr("Available on %1").arg(caps.gpu);
	caps.low_latency_available = extrapolation_ok;
	caps.low_latency_reason = extrapolation_ok ? QString() : tr("GPU self test of the low-latency mode failed");
	if (cuda_device)
		*cuda_device = cuda;
	else
		av_buffer_unref(&cuda);
	return caps;
}

// ---------------------------------------------------------------------------------------

FrameGenerator::FrameGenerator() = default;

FrameGenerator::~FrameGenerator()
{
	stop();
}

bool FrameGenerator::start(const Config &config, QString *error)
{
	stop();
	if (!config.cuda_device || !config.vulkan_device || !config.deliver) {
		if (error)
			*error = tr("Frame generation is not initialized");
		return false;
	}
	cfg = config;
	cfg.cuda_device = av_buffer_ref(config.cuda_device);
	cfg.vulkan_device = av_buffer_ref(config.vulkan_device);
	cfg.multiplier = std::clamp(config.multiplier, 1, kMaxMultiplier);
	cfg.source_fps = config.source_fps > 1.0 ? config.source_fps : 60.0;
	requested_multiplier.store(cfg.multiplier);
	{
		std::lock_guard<std::mutex> lock(mutex);
		zones_xywh = cfg.protected_zones;
		zones_dirty = true;
	}
	active_multiplier = cfg.multiplier;
	effective_multiplier.store(cfg.multiplier);
	stopping = false;
	interop_gpu = true;
	generation_failed = false;
	have_prev = false;
	last_emit_us = 0;
	sched_delay_us = 0;
	{
		std::lock_guard<std::mutex> lock(stats_mutex);
		st = FrameGeneratorStats();
	}
	requested_low_latency.store(cfg.low_latency);
	active_low_latency = cfg.low_latency;
	setStatus(activeStatusText());
	worker = std::thread([this]() { run(); });
	return true;
}

void FrameGenerator::stop()
{
	if (worker.joinable()) {
		{
			std::lock_guard<std::mutex> lock(mutex);
			stopping = true;
		}
		cond.notify_all();
		worker.join();
	}
	std::lock_guard<std::mutex> lock(mutex);
	for (auto &in : inputs)
		av_frame_free(&in.frame.frame);
	inputs.clear();
	for (auto &out : outputs)
		av_frame_free(&out.frame.frame);
	outputs.clear();
	engine.reset(); // needs the CUDA device, released below
	av_buffer_unref(&cuda_frames);
	av_buffer_unref(&vulkan_frames);
	frames_w = frames_h = 0;
	frames_sw_format = -1;
	av_buffer_unref(&cfg.cuda_device);
	av_buffer_unref(&cfg.vulkan_device);
}

void FrameGenerator::submit(ChiakiFfmpegFrame frame, int32_t frames_lost)
{
	if (!frame.frame)
		return;
	{
		std::lock_guard<std::mutex> lock(mutex);
		if (!worker.joinable() || stopping) {
			av_frame_free(&frame.frame);
			return;
		}
		// Bounded: the worker normally keeps up (a pair costs a few ms). If it falls behind,
		// keep the newest original and mark the gap as a discontinuity.
		while (inputs.size() >= 2) {
			Input &old = inputs.front();
			frames_lost += 1 + old.frames_lost;
			av_frame_free(&old.frame.frame);
			inputs.pop_front();
		}
		inputs.push_back({ frame, frames_lost, nowUs() });
	}
	cond.notify_one();
}

void FrameGenerator::setProtectedZones(const std::vector<float> &xywh)
{
	std::lock_guard<std::mutex> lock(mutex);
	zones_xywh = xywh;
	zones_dirty = true;
}

void FrameGenerator::setLowLatency(bool enabled)
{
	requested_low_latency.store(enabled);
	cond.notify_one();
}

QString FrameGenerator::activeStatusText() const
{
	if (active_multiplier <= 1)
		return tr("Off");
	return (active_low_latency ? tr("x%1 low latency active: %2") : tr("x%1 active: %2"))
		.arg(active_multiplier).arg(targetFpsText(cfg.source_fps, active_multiplier));
}

void FrameGenerator::setMultiplier(int multiplier)
{
	requested_multiplier.store(std::clamp(multiplier, 1, kMaxMultiplier));
	cond.notify_one();
}

FrameGeneratorStats FrameGenerator::stats() const
{
	std::lock_guard<std::mutex> lock(stats_mutex);
	return st;
}

QString FrameGenerator::status() const
{
	std::lock_guard<std::mutex> lock(stats_mutex);
	return status_text;
}

void FrameGenerator::setStatus(const QString &status)
{
	{
		std::lock_guard<std::mutex> lock(stats_mutex);
		if (status_text == status)
			return;
		status_text = status;
	}
	qCInfo(chiakiFrameGen).noquote() << "Frame generation:" << status;
	if (cfg.status_changed)
		cfg.status_changed(status);
}

void FrameGenerator::disableGeneration(const QString &reason)
{
	generation_failed = true;
	active_multiplier = 1;
	effective_multiplier.store(1);
	if (engine)
		engine->reset();
	setStatus(tr("Disabled during the stream, showing original frames: %1").arg(reason));
}

void FrameGenerator::run()
{
	using clock = std::chrono::steady_clock;
	while (true) {
		std::unique_lock<std::mutex> lock(mutex);
		const auto have_work = [this]() { return stopping || !inputs.empty(); };
		if (!have_work()) {
			if (outputs.empty()) {
				cond.wait(lock, have_work);
			} else {
				// sleep until shortly before the next release time, then spin the remainder
				const int64_t due = outputs.front().due_us;
				const int64_t wake = due - 600;
				if (wake > nowUs())
					cond.wait_until(lock, clock::time_point(std::chrono::microseconds(wake)), have_work);
				if (!have_work() && !outputs.empty()) {
					lock.unlock();
					while (nowUs() < due) {
						{
							std::lock_guard<std::mutex> peek(mutex);
							if (have_work())
								break;
						}
						std::this_thread::yield();
					}
					lock.lock();
				}
			}
		}
		if (stopping)
			return;
		if (!inputs.empty()) {
			Input in = inputs.front();
			inputs.pop_front();
			lock.unlock();
			process(in);
			continue;
		}
		lock.unlock();
		emitDue(nowUs(), false);
	}
}

bool FrameGenerator::ensureEngine(const AVFrame *src, QString *error)
{
	auto *src_fc = reinterpret_cast<AVHWFramesContext *>(src->hw_frames_ctx->data);
	const int sw_format = src_fc->sw_format;
	if (engine && cuda_frames && frames_w == src->width && frames_h == src->height && frames_sw_format == sw_format)
		return true;

	engine.reset();
	av_buffer_unref(&cuda_frames);
	av_buffer_unref(&vulkan_frames);
	frames_w = frames_h = 0;
	frames_sw_format = -1;
	{
		std::lock_guard<std::mutex> lock(stats_mutex);
		st.resets++;
	}
	have_prev = false;

	if (!isSupportedSwFormat(sw_format)) {
		const char *name = av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw_format));
		*error = tr("unsupported decoded pixel format %1").arg(QString::fromUtf8(name ? name : "?"));
		return false;
	}

	auto make_pool = [&](AVBufferRef *device, AVPixelFormat hw_format, bool vulkan) -> AVBufferRef * {
		AVBufferRef *ref = av_hwframe_ctx_alloc(device);
		if (!ref)
			return nullptr;
		auto *fc = reinterpret_cast<AVHWFramesContext *>(ref->data);
		fc->format = hw_format;
		fc->sw_format = static_cast<AVPixelFormat>(sw_format);
		fc->width = src->width;
		fc->height = src->height;
		fc->initial_pool_size = 0;
#if LIBAVUTIL_VERSION_MAJOR >= 59
		if (vulkan) {
			// CUDA can only import single-plane images
			auto *vkfc = reinterpret_cast<AVVulkanFramesContext *>(fc->hwctx);
			vkfc->flags = static_cast<AVVkFrameFlags>(vkfc->flags | AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE);
		}
#else
		Q_UNUSED(vulkan);
#endif
		if (av_hwframe_ctx_init(ref) < 0)
			av_buffer_unref(&ref);
		return ref;
	};
	cuda_frames = make_pool(cfg.cuda_device, AV_PIX_FMT_CUDA, false);
	vulkan_frames = make_pool(cfg.vulkan_device, AV_PIX_FMT_VULKAN, true);
	if (!cuda_frames) {
		*error = tr("could not allocate CUDA frames");
		return false;
	}
	if (!vulkan_frames && interop_gpu) {
		interop_gpu = false;
		qCWarning(chiakiFrameGen) << "Could not create Vulkan frames for CUDA interop, using CPU copies";
	}

	auto *cuda_hw = reinterpret_cast<AVCUDADeviceContext *>(reinterpret_cast<AVHWDeviceContext *>(cfg.cuda_device->data)->hwctx);
	engine = std::make_unique<OfaInterpolator>();
	std::string engine_error;
	if (!engine->init(cuda_hw->cuda_ctx, src->width, src->height,
	                  sw_format == AV_PIX_FMT_P010LE ? PixelLayout::P010 : PixelLayout::NV12, &engine_error)) {
		engine.reset();
		*error = QString::fromStdString(engine_error);
		return false;
	}
	frames_w = src->width;
	frames_h = src->height;
	frames_sw_format = sw_format;
	{
		std::lock_guard<std::mutex> lock(mutex);
		zones_dirty = true;
	}
	qCInfo(chiakiFrameGen) << "Interpolator ready for" << frames_w << "x" << frames_h
	                       << av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw_format))
	                       << (interop_gpu ? "(GPU interop)" : "(CPU copy)");
	return true;
}

AVFrame *FrameGenerator::toPresentable(AVFrame *cuda_frame, QString *error)
{
	if (interop_gpu && vulkan_frames) {
		AVFrame *vk = av_frame_alloc();
		int err = vk ? av_hwframe_get_buffer(vulkan_frames, vk, 0) : AVERROR(ENOMEM);
		if (err >= 0)
			err = av_hwframe_transfer_data(vk, cuda_frame, 0);
		if (err >= 0) {
			av_frame_copy_props(vk, cuda_frame);
			return vk;
		}
		av_frame_free(&vk);
		char buf[AV_ERROR_MAX_STRING_SIZE] = {};
		av_strerror(err, buf, sizeof(buf));
		qCWarning(chiakiFrameGen) << "CUDA -> Vulkan transfer failed (" << buf << "), falling back to CPU copies";
		interop_gpu = false;
		setStatus(tr("x%1 active with CPU copies (GPU interop unavailable: %2)").arg(active_multiplier).arg(QString::fromUtf8(buf)));
	}
	AVFrame *sw = av_frame_alloc();
	if (!sw || av_hwframe_transfer_data(sw, cuda_frame, 0) < 0) {
		av_frame_free(&sw);
		if (error)
			*error = tr("could not read back the frame");
		return nullptr;
	}
	av_frame_copy_props(sw, cuda_frame);
	return sw;
}

void FrameGenerator::process(Input &in)
{
	const int64_t t0 = nowUs();
	AVFrame *src = in.frame.frame;
	bool discontinuity = in.frames_lost > 0 || in.frame.recovered;

	const int requested = requested_multiplier.load();
	if (requested != active_multiplier && !(generation_failed && requested > 1)) {
		active_multiplier = requested;
		effective_multiplier.store(requested);
		discontinuity = true;
		if (engine)
			engine->reset();
		{
			std::lock_guard<std::mutex> lock(stats_mutex);
			st.resets++;
		}
		setStatus(activeStatusText());
	}
	const bool requested_ll = requested_low_latency.load();
	if (requested_ll != active_low_latency) {
		// The flow of the current pair stays valid in both modes: no reset needed, only a
		// fresh schedule (pending frames of the old mode are flushed just below).
		active_low_latency = requested_ll;
		sched_delay_us = 0;
		if (!generation_failed)
			setStatus(activeStatusText());
	}

	// A new original supersedes everything still waiting: pending originals are released
	// right away and pending generated frames are dropped as late.
	emitDue(t0, true);

	if (src->format != AV_PIX_FMT_CUDA || !src->hw_frames_ctx) {
		// Not a CUDA frame (should not happen when the session decodes with NVDEC): pass through.
		Output out{ in.frame, in.frames_lost, t0, false };
		deliver(out);
		return;
	}

	QString error;
	if (!generation_failed && !ensureEngine(src, &error))
		disableGeneration(error);

	const int n = generation_failed ? 1 : active_multiplier;
	const double nominal = 1.0 / cfg.source_fps;
	double interval = in.frame.pts - prev_pts;
	if (!have_prev || interval < nominal * 0.4 || interval > nominal * 2.5) {
		discontinuity = true;
		interval = nominal;
	}

	if (engine) {
		std::vector<float> zones;
		bool apply = false;
		{
			std::lock_guard<std::mutex> lock(mutex);
			if (zones_dirty) {
				zones = zones_xywh;
				zones_dirty = false;
				apply = true;
			}
		}
		if (apply) {
			std::vector<ProtectedZone> list;
			for (size_t i = 0; i + 3 < zones.size(); i += 4) {
				ProtectedZone z;
				z.x0 = zones[i];
				z.y0 = zones[i + 1];
				z.x1 = zones[i] + zones[i + 2];
				z.y1 = zones[i + 1] + zones[i + 3];
				list.push_back(z);
			}
			std::string zone_error;
			if (!engine->setProtectedZones(list, &zone_error))
				qCWarning(chiakiFrameGen) << "Could not apply protected zones:" << QString::fromStdString(zone_error);
			else
				qCInfo(chiakiFrameGen) << "Protected zones:" << list.size();
		}
	}

	// Low-latency mode: the original is shown right away and the generated frames are
	// predicted forward from it (extrapolation) instead of interpolated towards it.
	const bool low_latency = active_low_latency && n > 1 && !generation_failed;
	const double step = interval / std::max(n, 1);
	const int64_t step_us = static_cast<int64_t>(step * 1e6);
	int64_t original_shown_us = 0;
	if (low_latency) {
		AVFrame *original = toPresentable(src, &error);
		if (original) {
			ChiakiFfmpegFrame f = in.frame;
			f.frame = original;
			f.duration = step;
			{
				std::lock_guard<std::mutex> lock(mutex);
				outputs.push_back({ f, in.frames_lost, nowUs(), false });
			}
			emitDue(nowUs(), false);
		}
		original_shown_us = nowUs();
	}

	PairAnalysis pair;
	if (n > 1 && engine) {
		if (discontinuity)
			engine->reset();
		const GpuFrame gf{ reinterpret_cast<CUdeviceptr>(src->data[0]), static_cast<size_t>(src->linesize[0]),
		                   reinterpret_cast<CUdeviceptr>(src->data[1]), static_cast<size_t>(src->linesize[1]) };
		std::string engine_error;
		pair = engine->pushFrame(gf, &engine_error);
		if (!engine_error.empty())
			disableGeneration(QString::fromStdString(engine_error));
	}
	have_prev = true;
	prev_pts = in.frame.pts;

	std::vector<Output> produced;
	int generated_count = 0;

	if (n > 1 && !generation_failed && pair.valid && !discontinuity) {
		std::vector<AVFrame *> cuda_out;
		for (int j = 1; j < n; j++) {
			AVFrame *cf = av_frame_alloc();
			std::string engine_error;
			if (!cf || av_hwframe_get_buffer(cuda_frames, cf, 0) < 0) {
				av_frame_free(&cf);
				break;
			}
			const GpuFrame dst{ reinterpret_cast<CUdeviceptr>(cf->data[0]), static_cast<size_t>(cf->linesize[0]),
			                    reinterpret_cast<CUdeviceptr>(cf->data[1]), static_cast<size_t>(cf->linesize[1]) };
			const float pos = static_cast<float>(j) / n;
			if (!(low_latency ? engine->extrapolate(pos, dst, &engine_error) : engine->synthesize(pos, dst, &engine_error))) {
				av_frame_free(&cf);
				disableGeneration(QString::fromStdString(engine_error));
				break;
			}
			av_frame_copy_props(cf, src);
			cuda_out.push_back(cf);
		}
		std::string sync_error;
		if (!cuda_out.empty() && !engine->synchronize(&sync_error))
			disableGeneration(QString::fromStdString(sync_error));
		for (size_t k = 0; k < cuda_out.size(); k++) {
			if (!generation_failed) {
				AVFrame *presentable = toPresentable(cuda_out[k], &error);
				if (presentable) {
					const int j = static_cast<int>(k) + 1;
					ChiakiFfmpegFrame f{};
					f.frame = presentable;
					// interpolated: between the previous original and this one (slots 0..n-2,
					// this original in slot n-1); extrapolated: after this original (slots 1..n-1)
					f.pts = low_latency ? in.frame.pts + j * step : in.frame.pts - interval + j * step;
					f.duration = step;
					f.recovered = false;
					produced.push_back({ f, 0, low_latency ? j : j - 1, true });
					generated_count++;
				}
			}
			av_frame_free(&cuda_out[k]);
		}
	} else if (n > 1) {
		std::lock_guard<std::mutex> lock(stats_mutex);
		st.skipped_pairs++;
	}

	AVFrame *original = low_latency ? nullptr : toPresentable(src, &error);
	av_frame_free(&in.frame.frame);
	if (original) {
		ChiakiFfmpegFrame f = in.frame;
		f.frame = original;
		if (n > 1)
			f.duration = step;
		produced.push_back({ f, in.frames_lost, n - 1, false });
	}

	// Release schedule. The frames are only ready now, after flow, synthesis and the copies
	// to Vulkan, so anchoring the first generated frame at the arrival time would make it
	// late by the whole processing time (at x4 that exceeds the late tolerance and the
	// frame was always dropped). Anchor the schedule at arrival + a smoothed processing
	// delay instead: the cadence stays even from pair to pair, and never earlier than now.
	// In low-latency mode the schedule simply starts at the original, which is already shown.
	const int64_t ready_us = nowUs();
	int64_t base_us = ready_us;
	if (low_latency) {
		base_us = original_shown_us;
	} else if (n > 1) {
		if (generated_count > 0) {
			const int64_t delay_us = ready_us - in.arrival_us;
			sched_delay_us = sched_delay_us <= 0 ? delay_us : sched_delay_us + (delay_us - sched_delay_us) / 8;
		}
		base_us = std::max(in.arrival_us + sched_delay_us + kScheduleMarginUs, ready_us);
	}
	for (auto &o : produced)
		o.due_us = base_us + o.due_us * step_us;

	FrameGeneratorStats snapshot;
	bool log_stats = false;
	{
		std::lock_guard<std::mutex> lock(stats_mutex);
		st.last_pair_ms = (nowUs() - t0) / 1000.0;
		if (pair.scene_cut)
			st.skipped_pairs++;
		// about every 10 s of stream
		log_stats = st.originals > 0 && st.originals % static_cast<uint64_t>(std::max(1.0, cfg.source_fps * 10.0)) == 0;
		snapshot = st;
	}
	if (log_stats)
		qCInfo(chiakiFrameGen).nospace() << "x" << n << (low_latency ? " low-latency" : "") << " originals=" << snapshot.originals << " generated=" << snapshot.generated
		                                 << " dropped_late=" << snapshot.dropped_late << " dropped_backlog=" << snapshot.dropped_backlog
		                                 << " skipped_pairs=" << snapshot.skipped_pairs << " resets=" << snapshot.resets
		                                 << " last_pair_ms=" << snapshot.last_pair_ms << " flow_gpu_ms=" << pair.flow_ms
		                                 << " sched_delay_ms=" << sched_delay_us / 1000.0
		                                 << (interop_gpu ? " interop=gpu" : " interop=cpu");
	{
		std::lock_guard<std::mutex> lock(mutex);
		for (auto &o : produced)
			outputs.push_back(o);
	}
	emitDue(nowUs(), false);
}

void FrameGenerator::emitDue(int64_t now_us, bool flush)
{
	std::vector<Output> ready;
	{
		std::lock_guard<std::mutex> lock(mutex);
		const int64_t step_us = static_cast<int64_t>(1e6 / (cfg.source_fps * std::max(active_multiplier, 1)));
		const int64_t late_tolerance_us = step_us / 2;
		const int64_t min_spacing_us = cfg.display_hz > 1.0 ? static_cast<int64_t>(0.85e6 / cfg.display_hz) : 0;
		while (!outputs.empty()) {
			Output &o = outputs.front();
			if (!flush && o.due_us > now_us)
				break;
			if (o.synthetic) {
				const bool late = flush || now_us - o.due_us > late_tolerance_us;
				const bool too_close = min_spacing_us > 0 && last_emit_us > 0 && now_us - last_emit_us < min_spacing_us;
				const bool backlog = !late && !too_close && cfg.renderer_backlogged && cfg.renderer_backlogged();
				if (late || too_close || backlog) {
					av_frame_free(&o.frame.frame);
					outputs.pop_front();
					std::lock_guard<std::mutex> stats_lock(stats_mutex);
					if (backlog)
						st.dropped_backlog++;
					else
						st.dropped_late++;
					continue;
				}
			}
			ready.push_back(o);
			outputs.pop_front();
			last_emit_us = now_us;
		}
	}
	for (auto &o : ready)
		deliver(o);
}

void FrameGenerator::deliver(Output &out)
{
	{
		std::lock_guard<std::mutex> lock(stats_mutex);
		if (out.synthetic)
			st.generated++;
		else
			st.originals++;
	}
	cfg.deliver(out.frame, out.frames_lost);
	out.frame.frame = nullptr;
}

} // namespace framegen
