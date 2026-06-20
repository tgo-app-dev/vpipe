#ifndef ONVIF_DISCOVERY_STAGE_H
#define ONVIF_DISCOVERY_STAGE_H

#include "pipeline/typed-stage.h"

namespace vpipe {

// Interactive one-shot stage:
//   1) WS-Discovery multicast probe to 239.255.255.250:3702
//   2) print discovered cameras to stdout, prompt the user for a
//      selection + credentials + a short human-readable camera name
//      on stdin
//   3) authenticate to the camera via ONVIF SOAP, fetch the RTSP URI
//   4) encrypt the password with a key bound to host machine UUID +
//      OS user id, persist the result to LMDB sub-db "cameras" keyed
//      by the user-supplied camera name (the UUID is stored as a
//      record field so a downstream recorder can re-discover the
//      camera if DHCP moved it)
//   5) signal_done and terminate.
//
// 0 iports, 0 oports. APPLE-only in the first cut (the cipher relies
// on CommonCrypto).
class OnvifDiscoveryStage final
    : public TypedStage<OnvifDiscoveryStage>
{
public:
  static constexpr const char* kTypeName = "onvif-discovery";

  OnvifDiscoveryStage(const SessionContextIntf* s,
                      std::string               id,
                      std::vector<InEdge>       iports,
                      FlexData                  config);

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // Config attributes; defaults live in kSpec.attrs and are read in the
  // constructor via attr_*. Declarations carry no non-zero default.
  unsigned    _probe_timeout_ms{};
  std::string _db_name;
  bool        _overwrite_existing{};
  bool        _mask_password{};
};

}

#endif
