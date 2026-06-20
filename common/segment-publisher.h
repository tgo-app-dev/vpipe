#ifndef SEGMENT_PUBLISHER_H
#define SEGMENT_PUBLISHER_H

#include "common/encoded-segment.h"

#include <optional>
#include <string>
#include <string_view>

namespace vpipe {

class FFmpegLibraries;
class SessionContextIntf;
struct SegmentInfo;

struct ExtractResult {
  std::optional<EncodedSegment> video;
  std::optional<EncodedSegment> audio;
};

// Open the closed mp4 at `seg.path`, demux it, and return one
// EncodedSegment per stream (h.264 video on `.video`, AAC audio on
// `.audio`). Either may be left empty if the file lacks that stream
// or fails to open. Bytes are emitted in the mp4-native framing
// (AVCC for video, raw frames for audio); see encoded-segment.h.
//
// `db_key` is forwarded verbatim into both EncodedSegments; the
// capture stage sets it to the 8-byte BE microseconds-since-epoch
// key that names the segment in the <camera>-videos LMDB sub-db.
ExtractResult
extract_encoded_segments(const FFmpegLibraries*    libs,
                         const SessionContextIntf* session,
                         const SegmentInfo&        seg,
                         const std::string&        camera_name,
                         std::string_view          db_key);

}

#endif
