#include "vpipe/vpipe.h"
#include "vpipe-version.h"  /* generated */

namespace vpipe {

const char* vpipe_version()
{
  return VPIPE_VERSION_MAJOR "." VPIPE_VERSION_MINOR " (" GIT_HASH ")";
}

}

