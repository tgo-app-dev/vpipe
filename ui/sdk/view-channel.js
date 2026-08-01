// The contract between a stage-provided GUI view and its C++ backend.
//
// A view is an ES module that libvpipe embeds and the front-end app
// serves (see ui/ui-view-registry.h). It never imports from the app:
// everything it needs arrives through this module and through the
// `ctx.host` widget helpers the app passes to mount(). That is what
// lets a stage's panel ship with the stage instead of with the app.
//
// A view module exports:
//
//   export const strings = { 'key': ['English', '简体', '繁體'], ... };
//   export function mount(body, actions, ctx) { ...; return cleanup; }
//
// where ctx is:
//   host       { el, clear, makeIcon, t }  -- the app's widget helpers
//   onTitle(s)                             -- rename the panel title bar
//   config                                 -- persisted per-panel blob
//   onConfigChange()                       -- ask the host to save it
//   openChannel(handlers)                  -- see below; pre-bound to
//                                             this view's id
//
// TRANSPORT. One WebSocket per mounted panel, carrying the FlexData
// protocol the view and its backend define between themselves. The app
// only forwards:
//
//   view -> backend    a JSON object, delivered to
//                      UiViewBackendIntf::on_message
//   backend -> view    a JSON object, delivered to onMessage(msg)
//   backend -> view    a bulk BINARY payload with a JSON header,
//                      delivered to onBinary(header, arrayBuffer) --
//                      the path for media that must not pay JSON
//                      encoding. Wire layout of the binary frame:
//                        [u32 header_len LE][header JSON][payload]
//
// The socket reconnects automatically with a fixed backoff; onOpen
// fires on every (re)connection, so a view re-sends whatever request
// establishes its state rather than assuming the backend remembers.

const RECONNECT_MS = 1500;

// Same origin as the page. The access key rides as ?key= because a
// browser cannot set headers on a WebSocket.
function channelUrl(viewId, authKey) {
  const proto = (window.location.protocol === 'https:') ? 'wss:' : 'ws:';
  let u = proto + '//' + window.location.host + '/api/ui/view/'
        + encodeURIComponent(viewId) + '/ws';
  if (authKey) { u += '?key=' + encodeURIComponent(authKey); }
  return u;
}

// Split one binary frame into its JSON header and the payload bytes.
// Returns null on a malformed frame (dropped rather than thrown -- a
// view must not die on a bad frame).
function splitBinary(data) {
  if (!(data instanceof ArrayBuffer) || data.byteLength < 4) { return null; }
  const hlen = new DataView(data, 0, 4).getUint32(0, true);
  if (hlen > data.byteLength - 4) { return null; }
  let header;
  try {
    header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(data, 4, hlen)));
  } catch (e) { return null; }
  return { header, payload: data.slice(4 + hlen) };
}

// Open a channel to `viewId`'s backend.
//
//   handlers.onOpen()                  connected (fires on reconnect too)
//   handlers.onMessage(msg)            a structured message
//   handlers.onBinary(header, buffer)  a bulk payload
//   handlers.onClose()                 disconnected (a reconnect follows)
//
// Returns { send(msg), close() }. send() before the socket is open is
// dropped, not queued -- state is re-established from onOpen.
export function openViewChannel(viewId, handlers, authKey) {
  const h = handlers || {};
  let ws = null;
  let gone = false;
  let timer = null;

  function connect() {
    if (gone) { return; }
    try {
      ws = new WebSocket(channelUrl(viewId, authKey));
    } catch (e) { scheduleReconnect(); return; }
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { if (!gone && h.onOpen) { h.onOpen(); } };
    ws.onerror = () => {};
    ws.onmessage = (ev) => {
      if (gone) { return; }
      if (typeof ev.data === 'string') {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        if (h.onMessage) { h.onMessage(msg); }
        return;
      }
      const b = splitBinary(ev.data);
      if (b && h.onBinary) { h.onBinary(b.header, b.payload); }
    };
    ws.onclose = () => {
      if (gone) { return; }
      ws = null;
      if (h.onClose) { h.onClose(); }
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    if (gone || timer) { return; }
    timer = setTimeout(() => { timer = null; connect(); }, RECONNECT_MS);
  }

  connect();

  return {
    send(msg) {
      if (gone || !ws || ws.readyState !== WebSocket.OPEN) { return false; }
      try { ws.send(JSON.stringify(msg)); } catch (e) { return false; }
      return true;
    },
    close() {
      gone = true;
      if (timer) { clearTimeout(timer); timer = null; }
      if (ws) {
        try { ws.onclose = null; ws.close(); } catch (e) {}
        ws = null;
      }
    },
  };
}
