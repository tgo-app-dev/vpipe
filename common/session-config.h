#ifndef SESSION_CONFIG_H
#define SESSION_CONFIG_H

#include "common/flex-data.h"
#include <string_view>

namespace vpipe {

// Parses a Session config string into a FlexData tree. Rules:
//
//   * empty / all-whitespace                  -> empty object
//   * starts with '{' or '['                  -> inline JSON
//                                                (FlexData::from_json)
//   * otherwise                               -> filesystem path; read
//                                                the file, sniff first
//                                                4 bytes for 'V','P',
//                                                'F','D' magic; if it
//                                                matches dispatch to
//                                                FlexData::from_binary,
//                                                else treat the whole
//                                                file as JSON.
//
// Throws on parse / IO error. Config is required to be valid -- the
// fail-soft policy lives one level up in Session.
FlexData parse_session_config(std::string_view src);

}

#endif
