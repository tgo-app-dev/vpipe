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

export const api = {
  health:        ()        => req('GET',  '/api/health'),
  stageTypes:    ()        => req('GET',  '/api/stage-types'),
  listPipelines: ()        => req('GET',  '/api/pipelines'),
  createPipeline:(id)      => req('POST', '/api/pipelines', { id }),
  loadPipeline:  (path)    => req('POST', '/api/pipelines/load', { path }),
  // {cwd, files:[...]} -- .vpipeline files at the server's cwd for
  // the Load-Pipeline dialog's autocomplete.
  cwdPipelines:  ()        => req('GET',  '/api/cwd-pipelines'),
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
  dbList:        ()        => req('GET',  '/api/db/list'),
  dbKeys:        (body)    => req('POST', '/api/db/keys', body),
  dbValue:       (db, key) => req('POST', '/api/db/value', { db, key }),
  dbDeleteKey:   (db, key) => req('POST', '/api/db/delete-key', { db, key }),
  dbDrop:        (db)      => req('POST', '/api/db/drop', { db }),
};
