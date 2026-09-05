// Checks for model-filter.js's COMPATIBILITY rules.
//
// No runner and no dependencies, because this tree has neither node nor
// a JS test harness. It runs on any stock macOS under JavaScriptCore:
//
//   /System/Library/Frameworks/JavaScriptCore.framework/Versions/A/\
//     Helpers/jsc --module-file=apps/web-ui/web/js/model-filter.test.mjs
//
// Lives beside the module rather than under tests/ on purpose. The bug
// it pins came from this rule being COPIED into the desktop browser and
// the phone sheet and drifting; the thing most likely to stop that
// happening again is a maintainer editing the rule seeing the checks in
// the same directory listing.
//
// Exit status is not set (jsc always returns 0) -- read the output.

import { compatibleModels, parentAccepts, fieldAccepts }
  from './model-filter.js';

// Installed models as /api/models/installed describes them.
const H3 = { key: 'MiniMaxAI/MiniMax-H3-FL2VA', hf_path: 'MiniMaxAI/MiniMax-H3',
             model_type: 'minimax-h3-fl2va', category: 'model',
             param_class: '33B' };
const VDN = { key: 'OpenVDN/vdn-minimax-h3-stage-dmd',
              hf_path: 'OpenVDN/vdn-minimax-h3',
              model_type: 'minimax-h3-vdn', category: 'supplement',
              parent_model_type: 'minimax-h3-fl2va' };
const LARRY = { key: 'larryvrh/MiniMax-H3-Turbo-Lora',
                hf_path: 'larryvrh/MiniMax-H3-Turbo-Lora',
                model_type: 'minimax-h3-lora', category: 'supplement',
                parent_model_type: 'minimax-h3-fl2va' };
const LIGHT = { key: 'lightx2v/Minimax-h3-Turbo',
                hf_path: 'lightx2v/Minimax-h3-Turbo',
                model_type: 'minimax-h3-lora', category: 'supplement',
                parent_model_type: 'minimax-h3-fl2va' };
const KREA_LORA = { key: 'someone/Krea-2-Lora', hf_path: 'someone/Krea-2-Lora',
                    model_type: 'krea2-lora', category: 'supplement',
                    parent_model_type: 'krea2' };
const ALL = [H3, VDN, LARRY, LIGHT, KREA_LORA];

const LORA_FIELD = { suggest_db_type: 'minimax-h3-lora' };
const VDN_FIELD = { suggest_db_type: 'minimax-h3-vdn' };

let failed = 0;
const keys = (ms) => ms.map((m) => m.key).sort().join(', ');
function check(what, got, want) {
  const ok = got === want;
  if (!ok) { failed++; }
  print(`${ok ? 'ok  ' : 'FAIL'}  ${what}`);
  if (!ok) { print(`        got  ${got}\n        want ${want}`); }
}

// THE REPORTED BUG. Choosing a VDN branch on minimax-h3-model-config hid
// every H3 Turbo LoRA from the sibling `lora` field: the branch is a
// SUPPLEMENT, was treated as a candidate PARENT, and its model_type
// (minimax-h3-vdn) matches nothing -- so it vetoed its own siblings.
const BOTH_TURBOS = 'larryvrh/MiniMax-H3-Turbo-Lora, lightx2v/Minimax-h3-Turbo';
check('lora field, nothing else chosen: both turbos',
      keys(compatibleModels(ALL, LORA_FIELD, [])), BOTH_TURBOS);
check('lora field, VDN branch chosen: STILL both turbos',
      keys(compatibleModels(ALL, LORA_FIELD, [VDN.key])), BOTH_TURBOS);
check('linear_branch field, a turbo chosen: still the branch',
      keys(compatibleModels(ALL, VDN_FIELD, [LARRY.key])), VDN.key);
// A value may name a model by hf_path as well as by registry key.
check('a peer named by hf_path resolves too',
      keys(compatibleModels(ALL, LORA_FIELD, [VDN.hf_path])), BOTH_TURBOS);

// AND THE FILTER MUST STILL FILTER. Widening it until nothing is hidden
// would "fix" the above and be worthless.
check('a Krea-2 adapter stays hidden beside an H3 branch',
      parentAccepts(KREA_LORA, [VDN]), false);
check('a real parent admits its own supplement',
      parentAccepts(LARRY, [H3]), true);
check('a real parent excludes a foreign supplement',
      parentAccepts(KREA_LORA, [H3]), false);
check('a non-supplement is never filtered', parentAccepts(H3, [VDN]), true);

// parent_param_class, which the vision towers carry and the H3
// supplements do not.
const TOWER = { key: 't', model_type: 'vision-tower', category: 'supplement',
                parent_model_type: 'qwen3.5', parent_param_class: 'E4B' };
check('tower excluded by param_class',
      parentAccepts(TOWER, [{ key: 'l', model_type: 'qwen3.5',
                              category: 'model', param_class: '4B' }]), false);
// A registry record written before the catalog switched "e4b" -> "E4B".
check('tower admitted case-insensitively',
      parentAccepts(TOWER, [{ key: 'l', model_type: 'qwen3.5',
                              category: 'model', param_class: 'e4b' }]), true);

// The field's own half: the allow-list and the I/O requirement.
check('an untyped field shows plain models only',
      keys(ALL.filter((m) => fieldAccepts(m, {}))), H3.key);
check('need_outputs is required of the model',
      fieldAccepts({ model_type: 'x', category: 'model', outputs: ['text'] },
                   { need_outputs: 'video' }), false);

print(failed === 0 ? '\nALL PASS' : `\n${failed} FAILED`);
