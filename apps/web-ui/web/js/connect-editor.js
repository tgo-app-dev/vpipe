// The lane graph's connection editor: tap a node, pick one of its
// INPUTS, then pick the output that should feed it.
//
// It exists because the lane graph is the only view of a pipeline that
// fits a narrow column, and until now it was read-only about topology --
// the canvas in graph.js edits connections by aiming at a port, which
// needs width this view does not have. Two lists need none: the ports
// are named, so choosing from a list says more than a 6px target does.
//
// Shared by the phone shell and the desktop editor's narrow fallback for
// the same reason lane-graph.js is: two implementations would drift
// about which ports they offer, and that is the part a user cannot check.
//
// The BACKEND is authoritative. It re-checks payload type and tags on
// every connect and refuses a running pipeline outright (409); the
// compatibility shown here only decides what the list offers and how it
// is labelled, so a rule that falls behind under-offers rather than
// corrupting a graph.

import { el, clear, openModal, toast } from './dom.js';
import { t } from './i18n.js';
import { api } from './api.js';
import { shortType } from './graph.js';
import { portSlots, slotMismatch } from './port-compat.js';

// One row of either list. `sub` is the second line (a current source, a
// port type); `why` marks it unusable and says what is wrong.
function pickRow(label, sub, opts = {}) {
  const row = el('button', {
    class: 'conn-item' + (opts.why ? ' bad' : '')
         + (opts.danger ? ' danger' : ''),
    type: 'button',
    disabled: opts.disabled || !!opts.why,
  },
    el('span', { class: 'conn-item-main' }, label),
    sub ? el('span', { class: 'conn-item-sub' }, sub) : null);
  if (!opts.why && !opts.disabled && opts.onPick) {
    row.addEventListener('click', opts.onPick);
  }
  return row;
}

// `specFor(type)` -> the registered stage spec, or null. Both shells
// already hold the spec list for their own reasons; passing a lookup in
// keeps this module from fetching (and caching) a third copy.
//
// `onChanged(detail)` receives the pipeline detail the backend returns,
// which carries the new graph -- so the host re-renders from the server's
// answer rather than from an optimistic local edit.
export function openConnectEditor({ pipelineId, node, graph, specFor,
                                    editable, onChanged }) {
  const nodes = (graph && graph.nodes) || [];
  const edges = (graph && graph.edges) || [];
  const specOf = typeof specFor === 'function' ? specFor : () => null;
  const ins = portSlots(node, specOf(node.type), 'in');

  const body = el('div', { class: 'conn-body' });
  let close = () => {};

  // The source currently wired into iport `i`, or null.
  const sourceOf = (i) =>
    edges.find((e) => e.to === node.id && e.to_port === i) || null;

  const label = (slot) =>
    '[' + slot.index + '] ' + (slot.name || shortType(slot.type));

  // ---- step 1: which input ------------------------------------------
  function stepInputs() {
    clear(body);
    body.append(el('div', { class: 'conn-step' }, t('conn.pick_iport')));
    if (!editable) {
      body.append(el('div', { class: 'conn-note' }, t('conn.stop_to_edit')));
    }
    if (!ins.length) {
      body.append(el('div', { class: 'conn-empty' }, t('conn.no_iports')));
      return;
    }
    const list = el('div', { class: 'conn-list' });
    for (const slot of ins) {
      const src = sourceOf(slot.index);
      list.append(pickRow(label(slot),
        src ? src.from + ' [' + src.from_port + ']'
            : t('conn.unconnected'),
        { disabled: !editable, onPick: () => stepSources(slot) }));
    }
    body.append(list);
  }

  // ---- step 2: which stage's which output ---------------------------
  function stepSources(slot) {
    clear(body);
    const back = el('button', { class: 'conn-back', type: 'button' },
      '‹ ' + t('conn.back'));
    back.addEventListener('click', stepInputs);
    body.append(el('div', { class: 'conn-step' }, back,
      el('span', {}, t('conn.pick_source', { port: label(slot) }))));

    const list = el('div', { class: 'conn-list' });
    const src = sourceOf(slot.index);
    if (src) {
      list.append(pickRow(t('conn.disconnect'),
        src.from + ' [' + src.from_port + ']',
        { danger: true, onPick: () => doDisconnect(slot) }));
    }
    // Every other stage's outputs. The node itself is skipped: a
    // self-edge is a cycle, which the runtime refuses at launch, so
    // offering it would only produce a pipeline that will not start.
    let offered = 0;
    for (const other of nodes) {
      if (other.id === node.id) { continue; }
      const outs = portSlots(other, specOf(other.type), 'out');
      for (const o of outs) {
        const why = slotMismatch(o, slot);
        const cur = src && src.from === other.id && src.from_port === o.index;
        if (!why) { ++offered; }
        list.append(pickRow(
          other.id + ' [' + o.index + '] '
            + (o.name || shortType(o.type)),
          why ? t('conn.incompatible_' + why,
                  { want: shortType(slot.type) || '—',
                    got: shortType(o.type) || '—' })
              : (cur ? t('conn.current') : shortType(o.type)),
          { why, onPick: () => doConnect(slot, other.id, o.index) }));
      }
    }
    if (!offered) {
      list.append(el('div', { class: 'conn-empty' }, t('conn.no_sources')));
    }
    body.append(list);
  }

  // ---- apply ---------------------------------------------------------
  async function apply(fn, okKey, failKey) {
    try {
      const detail = await fn();
      close();
      toast(t(okKey), 'ok');
      if (typeof onChanged === 'function') { onChanged(detail); }
    } catch (e) {
      // Left OPEN on failure: the backend's reason (a 409 for a running
      // pipeline, a type it would not take) is about the choice just
      // made, so the list it was made from is what the user needs to see.
      toast(t(failKey, { msg: e.message }), 'error');
    }
  }
  const doConnect = (slot, from, from_port) =>
    apply(() => api.stageConnect(pipelineId,
            { from, from_port, to: node.id, to_port: slot.index }),
          'pl.connected', 'pl.connect_failed');
  const doDisconnect = (slot) =>
    apply(() => api.stageDisconnect(pipelineId,
            { to: node.id, to_port: slot.index }),
          'pl.disconnected', 'pl.disconnect_failed');

  stepInputs();
  close = openModal({
    title: t('conn.title', { id: node.id }),
    className: 'conn-modal',
    body,
    actions: [{ label: t('common.close'), cancel: true,
                onClick: (c) => c() }],
  });
}
