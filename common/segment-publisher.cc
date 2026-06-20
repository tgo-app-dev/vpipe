#include "common/segment-publisher.h"
#include "common/ffmpeg-libraries.h"
#include "common/segment-writer.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"

#include <sys/stat.h>
#include <utility>

namespace vpipe {

namespace {

void
populate_common(EncodedSegment&     es,
                const SegmentInfo&  seg,
                const std::string&  camera_name,
                std::string_view    db_key,
                const AVCodecParameters* par)
{
  es.db_key      = std::string(db_key);
  es.camera_name = camera_name;
  es.path        = seg.path;
  es.start_utc   = seg.start_utc;
  es.end_utc     = seg.end_utc;
  es.duration_us = seg.duration_us;
  es.codec_id    = static_cast<unsigned>(par->codec_id);
  if (par->extradata && par->extradata_size > 0) {
    es.extradata.assign(par->extradata,
                        par->extradata + par->extradata_size);
  }
}

int64_t
file_size_bytes(const std::string& path)
{
  struct stat st{};
  if (::stat(path.c_str(), &st) != 0) {
    return 0;
  }
  return static_cast<int64_t>(st.st_size);
}

}

ExtractResult
extract_encoded_segments(const FFmpegLibraries*    libs,
                         const SessionContextIntf* session,
                         const SegmentInfo&        seg,
                         const std::string&        camera_name,
                         std::string_view          db_key)
{
  ExtractResult result;
  if (!libs || !libs->valid() || seg.path.empty()) {
    return result;
  }

  const auto& fmt_api = libs->avformat().api;
  const auto& cdc_api = libs->avcodec().api;

  AVFormatContext* ictx = nullptr;
  int rc = fmt_api.open_input(&ictx, seg.path.c_str(), nullptr, nullptr);
  if (rc < 0 || !ictx) {
    if (session) {
      session->warn(fmt(
          "segment-publisher: avformat_open_input('{}') failed ({})",
          seg.path, rc));
    }
    return result;
  }
  rc = fmt_api.find_stream_info(ictx, nullptr);
  if (rc < 0) {
    if (session) {
      session->warn(fmt(
          "segment-publisher: find_stream_info('{}') failed ({})",
          seg.path, rc));
    }
    fmt_api.close_input(&ictx);
    return result;
  }

  int v_idx = -1;
  int a_idx = -1;
  for (unsigned i = 0; i < ictx->nb_streams; ++i) {
    auto* par = ictx->streams[i]->codecpar;
    if (v_idx < 0 && par->codec_type == AVMEDIA_TYPE_VIDEO
                  && par->codec_id   == AV_CODEC_ID_H264) {
      v_idx = static_cast<int>(i);
    } else if (a_idx < 0 && par->codec_type == AVMEDIA_TYPE_AUDIO
                         && par->codec_id   == AV_CODEC_ID_AAC) {
      a_idx = static_cast<int>(i);
    }
  }

  if (v_idx >= 0) {
    auto* par = ictx->streams[v_idx]->codecpar;
    EncodedSegment es;
    es.kind   = EncodedSegment::Kind::Video;
    es.width  = static_cast<unsigned>(par->width);
    es.height = static_cast<unsigned>(par->height);
    populate_common(es, seg, camera_name, db_key, par);
    // Reserve roughly the whole file size against the video plane --
    // we'd rather over-reserve once than grow the vector during the
    // packet loop.
    int64_t sz = file_size_bytes(seg.path);
    if (sz > 0) {
      es.data.reserve(static_cast<size_t>(sz));
    }
    result.video = std::move(es);
  }
  if (a_idx >= 0) {
    auto* par = ictx->streams[a_idx]->codecpar;
    EncodedSegment es;
    es.kind        = EncodedSegment::Kind::Audio;
    es.sample_rate = static_cast<unsigned>(par->sample_rate);
    es.channels    = static_cast<unsigned>(par->ch_layout.nb_channels);
    populate_common(es, seg, camera_name, db_key, par);
    result.audio = std::move(es);
  }

  if (v_idx < 0 && a_idx < 0) {
    fmt_api.close_input(&ictx);
    return result;
  }

  AVPacket* pkt = cdc_api.packet_alloc();
  if (!pkt) {
    fmt_api.close_input(&ictx);
    return result;
  }

  while (fmt_api.read_frame(ictx, pkt) >= 0) {
    int si = pkt->stream_index;
    if (si == v_idx && result.video) {
      auto& v = *result.video;
      v.data.insert(v.data.end(),
                    pkt->data, pkt->data + pkt->size);
    } else if (si == a_idx && result.audio) {
      auto& a = *result.audio;
      a.data.insert(a.data.end(),
                    pkt->data, pkt->data + pkt->size);
    }
    cdc_api.packet_unref(pkt);
  }

  cdc_api.packet_free(&pkt);
  fmt_api.close_input(&ictx);
  return result;
}

}
