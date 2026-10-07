#ifndef COMMON_HOST_REPORT_H
#define COMMON_HOST_REPORT_H

// host-report.h -- a STRUCTURED report for the host, beside a log line.
//
// A stage that refuses work says why in the log, in English, for a
// person. A host that wants to ACT on the refusal -- show what a clip
// asked for, point at the setting that would make it fit -- had only
// that sentence to go on, and parsing it would tie the host to wording
// that is free to change. So the numbers go to the host as a document
// too: the session keeps the latest few, and SessionIntf::reports()
// hands them over (include/vpipe/session-intf.h has the shape).
//
// IN-TREE ONLY. This is not part of the plugin SDK: growing a host
// interface a plugin calls is a feature-flagged change (docs/PLUGINS.md,
// "Versioning"), and the stages that refuse for memory are libvpipe's.
// A session that is not vpipe's own Session drops the report.
//
// Kinds, and what their documents hold (byte figures are unsigned):
//
//   "memory"       a stage refused work it could not fit:
//                  {"stage", "step" ("denoise" | "decode"), "need",
//                   "parts": [{"name", "bytes"}...] -- summing to need,
//                   "gates": [{"name", "need", "have", "ok"}...] -- what
//                   each budget wanted, margins included, and had,
//                   "parked", ...the step's own geometry}
//   "memory-plan"  the resource plan's peak, before anything loaded:
//                  {"peak", "phase", "phases": [{"name", "bytes"}...],
//                   "ram", "pool"}

#include "common/flex-data.h"

#include <string>

namespace vpipe {

class SessionContextIntf;

void host_report(const SessionContextIntf* session, std::string kind,
                 FlexData data);

}

#endif
