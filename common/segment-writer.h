#ifndef SEGMENT_WRITER_H
#define SEGMENT_WRITER_H

#include "common/ffmpeg-libraries.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vpipe {

class SessionContextIntf;

// Pure-logic helper: should we close the current file and open a new
// one, given the packet currently in hand? Rolls only at a video
// keyframe, only after `target_ms` of wall-clock has elapsed in the
// current file. Exposed at namespace scope so it can be tested
// without owning any ffmpeg state.
bool
decide_rollover(bool    is_video_key,
                int64_t elapsed_ms,
                int64_t target_ms);

// One output stream as seen by the segment muxer. `codecpar` is
// copied into each new file via avcodec_parameters_copy -- it must
// outlive the SegmentWriter (typically it points at the source
// AVStream's codecpar for stream-copy paths, or the encoder
// context's codecpar for transcode paths). `input_time_base` is the
// time-base of the packets the caller will hand to write_packet:
// SegmentWriter rescales those PTS / DTS to the output stream's
// time-base on the fly. Exactly one of `is_video` must be true
// (the writer needs to know which stream supplies the keyframes
// that gate rollovers).
struct OutputStreamSpec {
  const AVCodecParameters* codecpar;
  AVRational               input_time_base;
  bool                     is_video;
};

struct SegmentSpec {
  std::string              output_dir;   // existing directory
  std::string              filename_stem; // typically the camera name
  std::chrono::seconds     target_duration;
};

// Metadata about a finalized file: returned by write_packet whenever
// rollover happens, and by close() for the last file.
struct SegmentInfo {
  std::string path;
  std::chrono::system_clock::time_point start_utc;
  std::chrono::system_clock::time_point end_utc;
  int64_t                  duration_us = 0;
  bool                     has_audio   = false;
  bool                     stream_copy_video = true;
  bool                     stream_copy_audio = true;
};

// MP4 segment muxer with IDR-aligned rollover. The writer owns one
// AVFormatContext for the current file; rollover closes (write
// trailer + avio_closep) and opens a fresh context with the same
// stream layout. PTS / DTS are zero-rebased per file so each segment
// is independently decodable.
//
// Lifetime: construct -> open(...) -> write_packet/write_packet/... ->
// close(). Once close() returns the writer is one-shot done; a fresh
// SegmentWriter must be constructed to record another session.
class SegmentWriter {
public:
  SegmentWriter(const FFmpegLibraries*    libs,
                const SessionContextIntf* session,
                SegmentSpec               spec,
                bool                      stream_copy_video,
                bool                      stream_copy_audio);
  ~SegmentWriter();

  SegmentWriter(const SegmentWriter&)            = delete;
  SegmentWriter& operator=(const SegmentWriter&) = delete;

  // Open the first output file. `streams[0]` MUST be the video
  // output spec; `streams[1]` (optional) is the audio spec. Returns
  // false on any FFmpeg failure; the session is informed via
  // warn/error already.
  bool open(const std::vector<OutputStreamSpec>& streams);

  // True iff the writer is ready to take packets (open() returned
  // true and close() hasn't been called yet).
  bool ready() const noexcept;

  // True iff we'd like the next video keyframe to land. Useful for
  // the transcode path: the stage queries this before sending a
  // frame to the encoder and, if true, sets pict_type =
  // AV_PICTURE_TYPE_I to force the encoder to emit an IDR.
  bool wants_keyframe() const noexcept;

  // Mux one packet. `out_idx` is 0 for video, 1 for audio (the
  // value from the OutputStreamSpec position in open()). `pkt`'s
  // PTS / DTS are in the time-base supplied at open() time;
  // SegmentWriter rescales internally. Returns the SegmentInfo for
  // the just-closed file when this call triggered a rollover, else
  // nullopt.
  std::optional<SegmentInfo>
  write_packet(AVPacket* pkt, int out_idx, bool is_video_key);

  // Finalize the in-flight file (av_write_trailer + closep). Safe
  // to call twice. The returned SegmentInfo describes the last
  // file; .path is empty if no file was ever opened.
  SegmentInfo close();

  // UTC wall-clock the in-flight file was opened at (set by
  // open() and refreshed on every rollover). Callers that need a
  // stable per-segment identifier (e.g. the rtsp-capture per-GOP
  // publisher tagging each Beat with the parent segment's LMDB
  // key) read this immediately after open() / write_packet to
  // pick up the new value.
  std::chrono::system_clock::time_point
  file_open_utc() const noexcept { return _file_open_utc; }
  const std::string&
  current_path() const noexcept { return _current_path; }

private:
  bool open_new_file_();
  void close_current_file_();
  std::string make_filename_() const;

  const FFmpegLibraries*    _libs    = nullptr;
  const SessionContextIntf* _session = nullptr;
  SegmentSpec               _spec;
  bool                      _stream_copy_v = true;
  bool                      _stream_copy_a = true;

  bool mux_after_open_(AVPacket* pkt, int out_idx);

  std::vector<OutputStreamSpec> _streams;
  AVFormatContext*              _ofctx        = nullptr;
  // Per output index: time-base of the new file's AVStream
  // (initialized by avformat_write_header). Used to rescale
  // incoming PTS/DTS before muxing.
  std::vector<AVRational>       _out_tbs;
  // Microsecond anchor (across all streams) for the first packet
  // muxed into the current file. Subtracted from every subsequent
  // packet's PTS so the file starts at PTS == 0 while preserving
  // cross-stream A/V offsets.
  int64_t                       _anchor_us  = 0;
  bool                          _anchor_set = false;

  std::chrono::system_clock::time_point _file_open_utc{};
  std::chrono::steady_clock::time_point _file_open_steady{};
  std::chrono::system_clock::time_point _last_packet_utc{};
  std::string                   _current_path;
  bool                          _closed = true;  // before first open()
};

}

#endif
