#include "common/media-line.h"

#include <array>
#include <cstdlib>
#include <utility>

using namespace std;

namespace vpipe::media_line {

namespace {

constexpr char kB64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

struct MarkerSpec {
  string_view start;
  string_view end;
  Modality    modality;
  bool        base64;
};

constexpr MarkerSpec kMarkers[] = {
  {kFsImageStart,  kFsImageEnd,  Modality::Image, false},
  {kFsAudioStart,  kFsAudioEnd,  Modality::Audio, false},
  {kB64ImageStart, kB64ImageEnd, Modality::Image, true},
  {kB64AudioStart, kB64AudioEnd, Modality::Audio, true},
};

const char*
glyph_(Modality m)
{
  return m == Modality::Image ? "\xF0\x9F\x96\xBC"    // 🖼
                              : "\xF0\x9F\x94\x8A";   // 🔊
}

}  // namespace

string
base64_encode(span<const uint8_t> bytes)
{
  string out;
  out.reserve(((bytes.size() + 2) / 3) * 4);
  size_t i = 0;
  for (; i + 3 <= bytes.size(); i += 3) {
    const uint32_t v = (static_cast<uint32_t>(bytes[i]) << 16)
        | (static_cast<uint32_t>(bytes[i + 1]) << 8)
        | static_cast<uint32_t>(bytes[i + 2]);
    out.push_back(kB64Alphabet[(v >> 18) & 63]);
    out.push_back(kB64Alphabet[(v >> 12) & 63]);
    out.push_back(kB64Alphabet[(v >> 6) & 63]);
    out.push_back(kB64Alphabet[v & 63]);
  }
  const size_t rem = bytes.size() - i;
  if (rem == 1) {
    const uint32_t v = static_cast<uint32_t>(bytes[i]) << 16;
    out.push_back(kB64Alphabet[(v >> 18) & 63]);
    out.push_back(kB64Alphabet[(v >> 12) & 63]);
    out.push_back('=');
    out.push_back('=');
  } else if (rem == 2) {
    const uint32_t v = (static_cast<uint32_t>(bytes[i]) << 16)
        | (static_cast<uint32_t>(bytes[i + 1]) << 8);
    out.push_back(kB64Alphabet[(v >> 18) & 63]);
    out.push_back(kB64Alphabet[(v >> 12) & 63]);
    out.push_back(kB64Alphabet[(v >> 6) & 63]);
    out.push_back('=');
  }
  return out;
}

optional<vector<uint8_t>>
base64_decode(string_view text)
{
  // Inverse alphabet, built once. -1 = invalid, -2 = whitespace,
  // -3 = padding.
  static const array<int8_t, 256> kInv = [] {
    array<int8_t, 256> t;
    t.fill(-1);
    for (int i = 0; i < 64; ++i) {
      t[static_cast<unsigned char>(kB64Alphabet[i])] =
          static_cast<int8_t>(i);
    }
    t[static_cast<unsigned char>(' ')]  = -2;
    t[static_cast<unsigned char>('\t')] = -2;
    t[static_cast<unsigned char>('\r')] = -2;
    t[static_cast<unsigned char>('\n')] = -2;
    t[static_cast<unsigned char>('=')]  = -3;
    return t;
  }();

  vector<uint8_t> out;
  out.reserve((text.size() / 4) * 3);
  uint32_t acc    = 0;
  int      nbits  = 0;
  bool     padded = false;
  for (char c : text) {
    const int8_t v = kInv[static_cast<unsigned char>(c)];
    if (v == -2) {
      continue;
    }
    if (v == -3) {
      padded = true;
      continue;
    }
    if (v < 0 || padded) {
      // Invalid byte, or data after '=' padding.
      return nullopt;
    }
    acc = (acc << 6) | static_cast<uint32_t>(v);
    nbits += 6;
    if (nbits >= 8) {
      nbits -= 8;
      out.push_back(static_cast<uint8_t>((acc >> nbits) & 0xff));
    }
  }
  // A lone trailing 6-bit group can't encode a byte -> malformed.
  if (nbits >= 6) {
    return nullopt;
  }
  return out;
}

string
render_think_markers_plain(string_view text)
{
  string out;
  out.reserve(text.size());
  size_t pos = 0;
  while (pos < text.size()) {
    const size_t s = text.find(kThinkStart, pos);
    const size_t e = text.find(kThinkEnd, pos);
    const size_t m = std::min(s, e);
    if (m == string_view::npos) {
      out.append(text.substr(pos));
      break;
    }
    out.append(text.substr(pos, m - pos));
    if (m == s) {
      out += "\xE2\x9F\xA6think\xE2\x9F\xA7";    // ⟦think⟧
      pos = m + kThinkStart.size();
    } else {
      out += "\xE2\x9F\xA6/think\xE2\x9F\xA7";   // ⟦/think⟧
      pos = m + kThinkEnd.size();
    }
  }
  return out;
}

bool
has_media_marker(string_view line)
{
  size_t pos = 0;
  while ((pos = line.find(kMarkerPrefix, pos)) != string_view::npos) {
    for (const auto& m : kMarkers) {
      if (line.substr(pos).substr(0, m.start.size()) == m.start) {
        return true;
      }
    }
    pos += kMarkerPrefix.size();
  }
  return false;
}

vector<Segment>
parse(string_view line, vector<string>* errors)
{
  vector<Segment> out;
  string          text_run;
  auto flush_text = [&] {
    if (!text_run.empty()) {
      Segment s;
      s.kind = Segment::Kind::Text;
      s.text = std::move(text_run);
      text_run.clear();
      out.push_back(std::move(s));
    }
  };
  auto note = [&](string msg) {
    if (errors) {
      errors->push_back(std::move(msg));
    }
  };

  size_t pos = 0;
  while (pos < line.size()) {
    const size_t m = line.find(kMarkerPrefix, pos);
    if (m == string_view::npos) {
      text_run.append(line.substr(pos));
      break;
    }
    text_run.append(line.substr(pos, m - pos));

    const MarkerSpec* spec = nullptr;
    for (const auto& cand : kMarkers) {
      if (line.substr(m).substr(0, cand.start.size()) == cand.start) {
        spec = &cand;
        break;
      }
    }
    if (!spec) {
      // A "<|__vpipe_..." sequence that isn't a known start marker
      // (e.g. a stray end marker). Keep the prefix literally and scan
      // on after it.
      note("unrecognized media marker sequence (kept as text)");
      text_run.append(kMarkerPrefix);
      pos = m + kMarkerPrefix.size();
      continue;
    }

    const size_t body_beg = m + spec->start.size();
    const size_t e        = line.find(spec->end, body_beg);
    if (e == string_view::npos) {
      // Unterminated marker: drop everything from the start marker on
      // (the tail is attachment payload, not prose).
      note("unterminated media marker (attachment dropped)");
      break;
    }
    const string_view body = line.substr(body_beg, e - body_beg);

    Segment s;
    s.modality = spec->modality;
    bool ok    = true;
    if (!spec->base64) {
      s.kind = Segment::Kind::FsPath;
      s.text = string(body);
      if (body.empty()) {
        ok = false;
        note("media marker with empty file path (dropped)");
      }
    } else {
      s.kind             = Segment::Kind::Bytes;
      const size_t comma = body.find(',');
      if (comma == string_view::npos) {
        ok = false;
        note("base64 media marker missing 'LENGTH,' prefix (dropped)");
      } else {
        const string len_str(body.substr(0, comma));
        char*                   endp = nullptr;
        const unsigned long long want =
            strtoull(len_str.c_str(), &endp, 10);
        if (endp == len_str.c_str() || *endp != '\0') {
          ok = false;
          note("base64 media marker with non-numeric LENGTH (dropped)");
        } else if (auto dec = base64_decode(body.substr(comma + 1))) {
          if (dec->size() != want) {
            ok = false;
            note("base64 media marker LENGTH mismatch (dropped)");
          } else {
            s.bytes = std::move(*dec);
          }
        } else {
          ok = false;
          note("base64 media marker with invalid base64 (dropped)");
        }
      }
    }
    if (ok) {
      flush_text();
      out.push_back(std::move(s));
    }
    pos = e + spec->end.size();
  }
  flush_text();
  return out;
}

string
make_fs_marker(Modality m, string_view path)
{
  string out;
  const string_view s =
      m == Modality::Image ? kFsImageStart : kFsAudioStart;
  const string_view e = m == Modality::Image ? kFsImageEnd : kFsAudioEnd;
  out.reserve(s.size() + path.size() + e.size());
  out.append(s);
  out.append(path);
  out.append(e);
  return out;
}

string
make_base64_marker(Modality m, span<const uint8_t> bytes)
{
  const string_view s =
      m == Modality::Image ? kB64ImageStart : kB64AudioStart;
  const string_view e =
      m == Modality::Image ? kB64ImageEnd : kB64AudioEnd;
  string out(s);
  out += to_string(bytes.size());
  out += ',';
  out += base64_encode(bytes);
  out.append(e);
  return out;
}

string
to_display(string_view line)
{
  if (!has_media_marker(line)) {
    return string(line);
  }
  // Re-walk with the same scanner as parse(), but render markers as
  // glyphs (including malformed ones, so a broken attachment never
  // dumps its payload into a transcript).
  string out;
  size_t pos = 0;
  while (pos < line.size()) {
    const size_t m = line.find(kMarkerPrefix, pos);
    if (m == string_view::npos) {
      out.append(line.substr(pos));
      break;
    }
    out.append(line.substr(pos, m - pos));
    const MarkerSpec* spec = nullptr;
    for (const auto& cand : kMarkers) {
      if (line.substr(m).substr(0, cand.start.size()) == cand.start) {
        spec = &cand;
        break;
      }
    }
    if (!spec) {
      out.append(kMarkerPrefix);
      pos = m + kMarkerPrefix.size();
      continue;
    }
    const size_t body_beg = m + spec->start.size();
    const size_t e        = line.find(spec->end, body_beg);
    if (e == string_view::npos) {
      out += "\xE2\x9F\xA6";           // ⟦
      out += glyph_(spec->modality);
      out += "?\xE2\x9F\xA7";          // ?⟧
      break;
    }
    const string_view body = line.substr(body_beg, e - body_beg);
    out += "\xE2\x9F\xA6";             // ⟦
    out += glyph_(spec->modality);
    out += ' ';
    if (!spec->base64) {
      out.append(body);
    } else {
      const size_t comma = body.find(',');
      out.append(comma == string_view::npos ? string_view("?")
                                            : body.substr(0, comma));
      out += " bytes";
    }
    out += "\xE2\x9F\xA7";             // ⟧
    pos = e + spec->end.size();
  }
  return out;
}

}
