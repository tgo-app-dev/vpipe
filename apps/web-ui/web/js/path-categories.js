// The file categories a stage's `path_filter` may name, shared by both
// shells' file pickers.
//
// One table rather than two, because there were two and they had
// drifted: the desktop dialog read `path_filter` as category KEYWORDS
// ("image", "image,video,audio") while the phone sheet parsed it as
// extension patterns ("*.png;*.jpg") -- so every keyword filter reached
// the phone as "no filter" and it offered every file. extsForFilter()
// reads both forms.
//
// Kept in sync with the keywords stage-config.h documents. Lower-case,
// dot-led.
export const CATEGORY_EXTS = {
  image: ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.ppm',
          '.pgm', '.tiff', '.tif', '.heic'],
  audio: ['.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.opus',
          '.aiff', '.aif'],
  video: ['.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.ts',
          '.flv', '.mpg', '.mpeg'],
  text:  ['.txt', '.md', '.json', '.csv', '.log', '.yaml', '.yml',
          '.xml', '.srt', '.vtt'],
  // Model weights: what a LoRA field takes. Only .safetensors -- the
  // adapter reader opens nothing else, so offering .pt / .ckpt would be
  // offering a file the stage then refuses.
  weights: ['.safetensors'],
};

// The extensions a `path_filter` allows, or null for "no restriction".
// Category keywords (comma-separated) and extension patterns ("*.png",
// ".png") may be mixed. An unparseable filter yields null, which shows
// too much rather than too little.
export function extsForFilter(filter) {
  if (!filter) { return null; }
  const out = [];
  for (const raw of String(filter).split(/[;,\s]+/)) {
    const part = raw.trim().toLowerCase();
    if (!part) { continue; }
    if (CATEGORY_EXTS[part]) {
      out.push(...CATEGORY_EXTS[part]);
      continue;
    }
    const m = /\.([a-z0-9_]+)$/.exec(part);
    if (m) { out.push('.' + m[1]); }
  }
  return out.length ? [...new Set(out)] : null;
}
