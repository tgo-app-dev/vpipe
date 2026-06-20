// Color-theme helper. The theme is stored in localStorage and applied
// as `data-theme` on <html> (an inline script in index.html sets it on
// first paint to avoid a flash). CSS maps the value to a palette:
//   dark  -> default :root palette
//   light -> :root[data-theme="light"]
//   auto  -> light when the OS prefers light, else dark.

const KEY = 'vpipe_theme';
export const THEMES = ['dark', 'light', 'auto'];

export function getTheme() {
  try {
    const v = localStorage.getItem(KEY);
    return THEMES.includes(v) ? v : 'auto';
  } catch (e) {
    return 'auto';
  }
}

export function applyTheme(t) {
  const v = THEMES.includes(t) ? t : 'auto';
  document.documentElement.dataset.theme = v;
  try { localStorage.setItem(KEY, v); } catch (e) {}
  return v;
}
