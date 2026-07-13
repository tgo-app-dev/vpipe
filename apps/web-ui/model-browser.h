#ifndef WEBUI_MODEL_BROWSER_H
#define WEBUI_MODEL_BROWSER_H

#include "common/flex-data.h"

#include <string>

namespace vpipe {
class SessionContextIntf;
}

namespace vpipe::webui {

// Lists the installed (registered) models from the `db_name` LMDB sub-db
// (the registry model-fetch writes, normally "models"), each enriched with
// curated catalogue metadata -- derived `category` ("model" | "supplement"
// | "dataset"), input/output modalities, and parent linkage -- so the
// web-ui model browser can filter to models compatible with a stage field
// (by model_type) and with a chosen parent model. Records not in the
// catalogue (user-typed paths) fall back to category "model" with empty
// I/O. On a read error sets `err`; the models array is always present.
//
//   { models: [ { key, hf_path, local_path, model_type, family, version,
//                 param_class, variant, category, inputs, outputs,
//                 parent_model_type?, parent_param_class? } ] }
FlexData
list_installed_models(SessionContextIntf* sctx, const std::string& db_name,
                      std::string& err);

}

#endif
