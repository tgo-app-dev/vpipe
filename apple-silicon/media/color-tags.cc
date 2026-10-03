#include "apple-silicon/media/color-tags.h"

#include "common/beat-keys.h"

namespace vpipe::apple_media {

namespace {

FlexData
numbers(const std::vector<double>& v)
{
  FlexData a = FlexData::make_array();
  auto arr = a.as_array();
  for (double x : v) { arr.push_back(x); }
  return a;
}

std::vector<double>
read_numbers(const FlexData& v, std::size_t n)
{
  std::vector<double> out;
  if (!v.is_array()) { return out; }
  auto arr = v.as_array();
  if (arr.size() != n) { return out; }
  for (std::size_t i = 0; i < n; ++i) { out.push_back(arr[i].as_real()); }
  return out;
}

}  // namespace

void
ColorTags::write(FlexData& sideband) const
{
  if (!sideband.is_object()) { sideband = FlexData::make_object(); }
  auto o = sideband.as_object();
  o.insert_or_assign(sideband::kColorPrimaries,
                     FlexData::make_uint((std::uint64_t)primaries));
  o.insert_or_assign(sideband::kColorTransfer,
                     FlexData::make_uint((std::uint64_t)transfer));
  o.insert_or_assign(sideband::kColorMatrix,
                     FlexData::make_uint((std::uint64_t)matrix));
  o.insert_or_assign(sideband::kColorFullRange,
                     FlexData::make_bool(full_range));
  o.insert_or_assign(sideband::kAlphaMode,
                     FlexData::make_string(premultiplied ? "premultiplied"
                                                         : "straight"));
  if (source_bits > 0) {
    o.insert_or_assign(sideband::kSourceBits,
                       FlexData::make_uint((std::uint64_t)source_bits));
  }
  if (mastering.size() == 10) {
    o.insert_or_assign(sideband::kMasteringDisplay, numbers(mastering));
  }
  if (content_light.size() == 2) {
    o.insert_or_assign(sideband::kContentLight, numbers(content_light));
  }
}

ColorTags
ColorTags::read(const FlexData& sideband)
{
  ColorTags t;
  if (!sideband.is_object()) { return srgb(); }
  FlexData sb = sideband;  // as_object() on a const value is a view
  auto o = sb.as_object();
  const bool has_p = o.contains(sideband::kColorPrimaries);
  const bool has_t = o.contains(sideband::kColorTransfer);
  if (!has_p && !has_t) { t = srgb(); }
  if (has_p) {
    t.primaries = (int)o.at(sideband::kColorPrimaries).as_int(2);
  }
  if (has_t) {
    t.transfer = (int)o.at(sideband::kColorTransfer).as_int(2);
  }
  if (o.contains(sideband::kColorMatrix)) {
    t.matrix = (int)o.at(sideband::kColorMatrix).as_int(2);
  }
  if (o.contains(sideband::kColorFullRange)) {
    t.full_range = o.at(sideband::kColorFullRange).as_bool(true);
  }
  if (o.contains(sideband::kAlphaMode)) {
    t.premultiplied =
        o.at(sideband::kAlphaMode).as_string("") == "premultiplied";
  }
  if (o.contains(sideband::kSourceBits)) {
    t.source_bits = (int)o.at(sideband::kSourceBits).as_int(0);
  }
  if (o.contains(sideband::kMasteringDisplay)) {
    t.mastering = read_numbers(o.at(sideband::kMasteringDisplay), 10);
  }
  if (o.contains(sideband::kContentLight)) {
    t.content_light = read_numbers(o.at(sideband::kContentLight), 2);
  }
  return t;
}

}  // namespace vpipe::apple_media
