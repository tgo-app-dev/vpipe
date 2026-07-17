// REST client for the web-ui backend (/api/*).

// First-cut auth: a remote server gates /api/* on an 8-char key printed
// in its console. We persist it and attach it as X-Auth-Key; on a 401
// we ask the UI (via the registered prompt) for the key and retry once.
const KEY_STORE = 'vpipe_key';
let authKey = '';
try { authKey = localStorage.getItem(KEY_STORE) || ''; } catch (e) {}
let authPrompt = null;
// A single shared key-prompt promise. The User I/O view polls a few
// times a second, so without this every 401 would open its own modal
// and the flood would keep stealing focus / clearing the field. Any
// number of concurrent 401s now await the SAME open prompt.
let authInflight = null;

export function setAuthKey(k) {
  authKey = k || '';
  try { localStorage.setItem(KEY_STORE, authKey); } catch (e) {}
}
export function setAuthPrompt(fn) { authPrompt = fn; }

function ensureKey() {
  if (!authPrompt) { return Promise.resolve(null); }
  if (!authInflight) {
    authInflight = Promise.resolve()
      .then(() => authPrompt())
      .finally(() => { authInflight = null; });
  }
  return authInflight;
}

// After the user supplies a key and a request succeeds with it, the page
// is likely showing data fetched before this server (re)authenticated --
// e.g. a previously connected client whose server just restarted. Force a
// one-time hard refresh so every view reloads against the new connection.
let reloadedAfterAuth = false;
function reloadAfterAuth() {
  if (reloadedAfterAuth) { return; }
  reloadedAfterAuth = true;
  setTimeout(() => location.reload(), 0);
}

async function req(method, url, body, retried) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  if (authKey) { opt.headers['X-Auth-Key'] = authKey; }
  const r = await fetch(url, opt);
  const text = await r.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch (e) { data = { error: text }; }
  }
  if (r.status === 401 && !retried) {
    const k = await ensureKey();
    if (k) {
      setAuthKey(k);
      const out = await req(method, url, body, true);
      reloadAfterAuth();   // succeeded with the new key -> refresh the page
      return out;
    }
  }
  if (!r.ok) {
    const msg = (data && data.error) ? data.error : ('HTTP ' + r.status);
    throw new Error(msg);
  }
  return data;
}

const pid = (id) => encodeURIComponent(id);

// Read an NDJSON stream from a POST endpoint, invoking callbacks as each
// record arrives (rather than buffering the whole response). Records are
// one JSON object per line, tagged with a `t` field. cbs = { onMeta,
// onRow, onDone, onError }. Pass an AbortSignal to cancel mid-stream. The
// returned promise resolves when the stream ends (or is aborted); a
// transport/HTTP error rejects. 401 triggers the same key prompt + retry
// as req().
async function ndjsonStream(url, body, cbs, signal, retried) {
  const opt = { method: 'POST', headers: {}, signal };
  opt.headers['Content-Type'] = 'application/json';
  opt.body = JSON.stringify(body);
  if (authKey) { opt.headers['X-Auth-Key'] = authKey; }
  let r;
  try {
    r = await fetch(url, opt);
  } catch (e) {
    if (e && e.name === 'AbortError') { return; }
    throw e;
  }
  if (r.status === 401 && !retried) {
    const k = await ensureKey();
    if (k) {
      setAuthKey(k);
      const out = await ndjsonStream(url, body, cbs, signal, true);
      reloadAfterAuth();
      return out;
    }
  }
  if (!r.ok || !r.body) {
    let msg = 'HTTP ' + r.status;
    try { const j = JSON.parse(await r.text()); if (j && j.error) { msg = j.error; } }
    catch (e) {}
    throw new Error(msg);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const dispatch = (line) => {
    const s = line.trim();
    if (!s) { return; }
    let o;
    try { o = JSON.parse(s); } catch (e) { return; }
    if (o.t === 'meta' && cbs.onMeta) { cbs.onMeta(o); }
    else if (o.t === 'row' && cbs.onRow) { cbs.onRow(o); }
    else if (o.t === 'done' && cbs.onDone) { cbs.onDone(o); }
    else if (o.t === 'error' && cbs.onError) { cbs.onError(o.error || 'error'); }
  };
  for (;;) {
    let chunk;
    try { chunk = await reader.read(); }
    catch (e) { if (e && e.name === 'AbortError') { return; } throw e; }
    if (chunk.done) { break; }
    buf += dec.decode(chunk.value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      dispatch(buf.slice(0, nl));
      buf = buf.slice(nl + 1);
    }
  }
  if (buf) { dispatch(buf); }
}

export const api = {
  health:        ()        => req('GET',  '/api/health'),
  stageTypes:    ()        => req('GET',  '/api/stage-types'),
  listPipelines: ()        => req('GET',  '/api/pipelines'),
  createPipeline:(id)      => req('POST', '/api/pipelines', { id }),
  renamePipeline:(id, to)  => req('POST',
                       `/api/pipelines/${pid(id)}/rename`, { to }),
  loadPipeline:  (path)    => req('POST', '/api/pipelines/load', { path }),
  // {cwd, files:[...]} -- .vpipeline files at the server's cwd for
  // the Load-Pipeline dialog's autocomplete.
  cwdPipelines:  ()        => req('GET',  '/api/cwd-pipelines'),
  // One directory's entries in the session's namespace (virtual "/"-rooted
  // when sandboxed). Returns {sandboxed, path, parent, mounts, total,
  // offset, entries:[{name, dir, size?}]}. `opts` (all optional):
  //   offset,limit -- return only entries [offset, offset+limit) of the
  //                   sorted listing (large-directory windowing); omit for
  //                   the whole listing (the open/save dialog's default).
  //   dirsOnly     -- return only sub-directories (the tree pane).
  //   exts         -- array of dot-led extensions ('.png', ...); keep only
  //                   files matching one (dirs always shown). Empty = all.
  fsList:        (path, opts = {}) => {
    let q = '/api/fs/list?path=' + encodeURIComponent(path || '');
    if (opts.dirsOnly) { q += '&dirs_only=1'; }
    if (Number.isInteger(opts.offset)) { q += '&offset=' + opts.offset; }
    if (Number.isInteger(opts.limit))  { q += '&limit=' + opts.limit; }
    if (Array.isArray(opts.exts) && opts.exts.length) {
      q += '&exts=' + encodeURIComponent(opts.exts.join(','));
    }
    return req('GET', q);
  },
  // Direct URL to a file's raw bytes (Range-aware) for the file-browser
  // preview -- used as an <img>/<audio>/<video> `src`. A media element
  // can't set headers, so the access key rides as ?key= (accepted by the
  // server the same way the Preview WebSocket does).
  fsFileUrl:     (path)    => {
    let u = '/api/fs/file?path=' + encodeURIComponent(path || '');
    if (authKey) { u += '&key=' + encodeURIComponent(authKey); }
    return u;
  },
  // Fetch up to `maxBytes` of a file decoded as text (one preview
  // "screen"). A Range request bounds the transfer so a huge file is
  // never pulled whole. -> { text, truncated, total }.
  fsText:        async (path, maxBytes = 65536) => {
    const headers = { Range: 'bytes=0-' + Math.max(0, maxBytes - 1) };
    if (authKey) { headers['X-Auth-Key'] = authKey; }
    const r = await fetch('/api/fs/file?path='
                          + encodeURIComponent(path || ''), { headers });
    if (!r.ok && r.status !== 206) {
      let msg = 'HTTP ' + r.status;
      try { const j = JSON.parse(await r.text()); if (j && j.error) { msg = j.error; } }
      catch (e) {}
      throw new Error(msg);
    }
    const text = await r.text();
    let total = text.length;
    let truncated = false;
    const cr = r.headers.get('Content-Range');   // "bytes 0-65535/123456"
    if (cr) {
      const m = /\/(\d+)\s*$/.exec(cr);
      if (m) {
        total = parseInt(m[1], 10);
        truncated = total > maxBytes;
      }
    }
    return { text, truncated, total };
  },
  fsMkdir:       (path, name) => req('POST', '/api/fs/mkdir', { path, name }),
  fsRename:      (path, to)   => req('POST', '/api/fs/rename', { path, to }),
  getPipeline:   (id)      => req('GET',  `/api/pipelines/${pid(id)}`),
  savePipeline:  (id, path)=> req('POST', `/api/pipelines/${pid(id)}/save`,
                                   path ? { path } : {}),
  unloadPipeline:(id)      => req('POST', `/api/pipelines/${pid(id)}/unload`),
  launch:        (id)      => req('POST', `/api/pipelines/${pid(id)}/launch`),
  pause:         (id)      => req('POST', `/api/pipelines/${pid(id)}/pause`),
  stop:          (id)      => req('POST', `/api/pipelines/${pid(id)}/stop`),
  // Per-edge buffer utilization of a running pipeline (graph overlay).
  bufferStatus:  (id)      => req('GET',
                       `/api/pipelines/${pid(id)}/buffer-status`),
  insertStage:   (id, s)   => req('POST', `/api/pipelines/${pid(id)}/stages`, s),
  removeStage:   (id, sid) => req('DELETE',
                       `/api/pipelines/${pid(id)}/stages/${pid(sid)}`),
  renameStage:   (id, sid, to) => req('POST',
                       `/api/pipelines/${pid(id)}/stages/${pid(sid)}/rename`,
                       { to }),
  // Duplicate a stage's settings under a fresh, non-colliding id (the
  // server generates "<sid>-N" unless `to` is given). No connections.
  duplicateStage:(id, sid, to) => req('POST',
                       `/api/pipelines/${pid(id)}/stages/${pid(sid)}/duplicate`,
                       to ? { to } : {}),
  // Edge editing for the composer. connect re-points an existing input
  // (omit-or-equal to_port appends a new one); disconnect drops one.
  // edge = {from, from_port, to, to_port?} / {to, to_port}.
  stageConnect:  (id, edge) => req('POST',
                       `/api/pipelines/${pid(id)}/connect`, edge),
  stageDisconnect:(id, edge)=> req('POST',
                       `/api/pipelines/${pid(id)}/disconnect`, edge),
  getStageConfig:(id, sid) => req('GET',
                       `/api/pipelines/${pid(id)}/stages/${pid(sid)}/config`),
  setStageConfig:(id, sid, cfg) => req('PUT',
                       `/api/pipelines/${pid(id)}/stages/${pid(sid)}/config`,
                       cfg),

  // User I/O console + interactive getline.
  ioConsole:     (since)   => req('GET',
                       `/api/io/console?since=${since || 0}`),
  ioPending:     ()        => req('GET',  '/api/io/pending'),
  ioInput:       (id, text)=> req('POST', '/api/io/input', { id, text }),
  ioClear:       ()        => req('POST', '/api/io/clear'),
  ioGetLimit:    ()        => req('GET',  '/api/io/limit'),
  ioSetLimit:    (n)       => req('PUT',  '/api/io/limit',
                                  { max_console: n }),

  // Session log console (diagnostic log_* stream) + debug-level
  // control for the Session Log view.
  logConsole:    (since)   => req('GET',
                       `/api/log/console?since=${since || 0}`),
  logClear:      ()        => req('POST', '/api/log/clear'),
  logGetLevel:   ()        => req('GET',  '/api/log/level'),
  logSetLevel:   (level)   => req('PUT',  '/api/log/level', { level }),
  logGetLimit:   ()        => req('GET',  '/api/log/limit'),
  logSetLimit:   (n)       => req('PUT',  '/api/log/limit', { max_log: n }),

  // System status (GPU util / memory) for the bottom status bar.
  systemStatus:  ()        => req('GET',  '/api/system/status'),

  // Startup permission self-test report, shown in a dialog on connect.
  // {ready, has_warnings, checks:[{name,status,detail,hints:[...]}]}.
  startupChecks: ()        => req('GET',  '/api/startup-checks'),

  // UI/message language: GET {language, supported:[...]}; PUT sets it so
  // server-produced messages match the browser's chosen locale.
  getI18n:       ()        => req('GET',  '/api/i18n'),
  setLanguage:   (language)=> req('PUT',  '/api/i18n', { language }),

  // Active HLS streams (live "hls-broadcast" stages) for the User I/O
  // workspace's HLS video view.
  hlsStreams:    ()        => req('GET',  '/api/hls/streams'),

  // Active preview streams (live "preview" stages) for the low-latency
  // Preview view.
  previewStreams:()        => req('GET',  '/api/preview/streams'),
  // WebSocket URL of a preview stage's live stream (fMP4 video + PCM). Same
  // origin as the page (ws:// or wss:// per the page scheme). The access
  // key rides as ?key= because browsers can't set headers on a WebSocket.
  previewWsUrl:  (pipeline, stage) => {
    const proto = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
    let u = proto + '//' + window.location.host + '/api/preview/'
          + pid(pipeline) + '/' + pid(stage) + '/ws';
    if (authKey) { u += '?key=' + encodeURIComponent(authKey); }
    return u;
  },

  // Performance profiler: capture control + timeline retrieval.
  profilerStatus:()        => req('GET',  '/api/profiler/status'),
  profilerStart: (maxEv)   => req('POST', '/api/profiler/start',
                                  maxEv ? { max_events: maxEv } : {}),
  profilerStop:  ()        => req('POST', '/api/profiler/stop'),
  profilerReset: ()        => req('POST', '/api/profiler/reset'),
  profilerData:  ()        => req('GET',  '/api/profiler/data'),

  // Database browser. list/keys/value read; delete-key/drop mutate and
  // are refused by the backend unless every pipeline is stopped (list
  // reports a `deletable` flag).
  // Installed (registered) models enriched with catalogue metadata
  // (category / input-output modalities / parent linkage) for the
  // compatibility-aware model browser. -> { models: [...] }.
  modelsInstalled: ()      => req('GET',  '/api/models/installed'),
  dbList:        ()        => req('GET',  '/api/db/list'),
  dbKeys:        (body)    => req('POST', '/api/db/keys', body),
  // Streaming value-filtered scan. Same key query as dbKeys plus
  // {combine, filters:[{cond, kw}]}; results (up to 64k) arrive as they
  // are found via cbs.onMeta/onRow/onDone/onError. Pass signal to cancel.
  dbScan:        (body, cbs, signal) =>
                     ndjsonStream('/api/db/scan', body, cbs, signal),
  dbValue:       (db, key) => req('POST', '/api/db/value', { db, key }),
  dbDeleteKey:   (db, key) => req('POST', '/api/db/delete-key', { db, key }),
  dbDrop:        (db)      => req('POST', '/api/db/drop', { db }),
};
