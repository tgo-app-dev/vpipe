// What kind of file a name looks like, by extension -- the one table
// both file browsers (the desktop three-column view and the phone's
// single-column one) decide previews from.
//
// It lives on its own because a media-type table is exactly the sort of
// thing that drifts when it is copied: a format added to one browser and
// not the other is a file that previews on a laptop and offers a
// download on a phone, with nothing to say why.

// Lower-case, no leading dot. Anything not listed has no inline preview.
export const IMAGE = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif',
  'svg', 'tiff', 'tif', 'heic', 'ico']);
export const AUDIO = new Set(['wav', 'mp3', 'flac', 'aac', 'm4a', 'ogg',
  'opus', 'aiff', 'aif']);
export const VIDEO = new Set(['mp4', 'm4v', 'mov', 'webm', 'mkv', 'avi',
  'ts', 'flv', 'mpg', 'mpeg']);
export const TEXT = new Set(['txt', 'md', 'log', 'csv', 'srt', 'vtt',
  'json', 'xml', 'yaml', 'yml', 'html', 'htm', 'js', 'css', 'c', 'cc',
  'h', 'hpp', 'py', 'sh', 'ini', 'conf', 'toml', 'ts', 'tsx', 'jsx']);

export function extOf(name) {
  const i = String(name || '').lastIndexOf('.');
  return i >= 0 ? String(name).slice(i + 1).toLowerCase() : '';
}

// 'image' | 'audio' | 'video' | 'text' | null.
export function categorize(name) {
  const e = extOf(name);
  if (IMAGE.has(e)) { return 'image'; }
  if (AUDIO.has(e)) { return 'audio'; }
  if (VIDEO.has(e)) { return 'video'; }
  if (TEXT.has(e))  { return 'text'; }
  return null;
}

// Icon name for a listing entry ({name, dir}).
export function iconFor(entry) {
  if (entry && entry.dir) { return 'folder'; }
  const c = categorize(entry && entry.name);
  return c === 'image' ? 'image'
    : c === 'audio' ? 'audio'
    : c === 'video' ? 'video' : 'file';
}
