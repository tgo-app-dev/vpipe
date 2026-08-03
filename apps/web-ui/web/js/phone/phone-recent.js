// The pipeline the operator last looked at, remembered across view
// switches and reloads.
//
// It is shared state rather than the pipeline view's private business,
// because on a phone one view is on screen at a time: a panel that opens
// with an empty picker has no way to know what the session is ABOUT,
// even though the user just spent a minute in the pipeline view deciding
// exactly that. Writing it down once here lets the stage panels open on
// the right thing instead of asking again.
//
// Advisory only. Nothing depends on it being present, valid, or still
// loaded -- a pipeline can be unloaded or renamed from anywhere -- so
// every reader treats a miss as "no opinion" and falls back to whatever
// it would have done before.

const KEY = 'vpipe_phone_pipeline';

export function lastPipeline() {
  try { return localStorage.getItem(KEY) || null; }
  catch (e) { return null; }
}

export function setLastPipeline(id) {
  try {
    if (id) { localStorage.setItem(KEY, id); }
    else { localStorage.removeItem(KEY); }
  } catch (e) { /* storage blocked -- the choice just won't persist */ }
}
