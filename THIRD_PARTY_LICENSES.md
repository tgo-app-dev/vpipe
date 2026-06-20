# Third-Party Licenses

This project bundles, vendors, and/or statically links the following
third-party software. Each entry below names the component, identifies
the upstream source, and reproduces the copyright statement and the
verbatim license text as required by that component's license.

vpipe itself is licensed under the Apache License, Version 2.0; see the
root `LICENSE` file.

---

## LMDB (liblmdb)

The `vpipe` shared library statically links LMDB
(<https://git.openldap.org/openldap/openldap>), vendored as a git
submodule at `extern/lmdb/`. Only the C sources `mdb.c` and `midl.c`
are compiled into `vpipe`; the headers, sources, license, and copyright
notice are preserved unmodified at
`extern/lmdb/libraries/liblmdb/`.

LMDB is distributed under the OpenLDAP Public License, Version 2.8.

### Copyright notice

Reproduced verbatim from `extern/lmdb/libraries/liblmdb/COPYRIGHT`:

> Copyright 2011-2021 Howard Chu, Symas Corp.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted only as authorized by the OpenLDAP
> Public License.
>
> A copy of this license is available in the file LICENSE in the
> top-level directory of the distribution or, alternatively, at
> <http://www.OpenLDAP.org/license.html>.
>
> OpenLDAP is a registered trademark of the OpenLDAP Foundation.
>
> Individual files and/or contributed packages may be copyright by
> other parties and/or subject to additional restrictions.
>
> This work also contains materials derived from public sources.
>
> Additional information about OpenLDAP can be obtained at
> <http://www.openldap.org/>.

### License

Reproduced verbatim from `extern/lmdb/libraries/liblmdb/LICENSE`:

> The OpenLDAP Public License
>   Version 2.8, 17 August 2003
>
> Redistribution and use of this software and associated documentation
> ("Software"), with or without modification, are permitted provided
> that the following conditions are met:
>
> 1. Redistributions in source form must retain copyright statements
>    and notices,
>
> 2. Redistributions in binary form must reproduce applicable copyright
>    statements and notices, this list of conditions, and the following
>    disclaimer in the documentation and/or other materials provided
>    with the distribution, and
>
> 3. Redistributions must contain a verbatim copy of this document.
>
> The OpenLDAP Foundation may revise this license from time to time.
> Each revision is distinguished by a version number.  You may use
> this Software under terms of this license revision or under the
> terms of any subsequent revision of the license.
>
> THIS SOFTWARE IS PROVIDED BY THE OPENLDAP FOUNDATION AND ITS
> CONTRIBUTORS ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
> AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT
> SHALL THE OPENLDAP FOUNDATION, ITS CONTRIBUTORS, OR THE AUTHOR(S)
> OR OWNER(S) OF THE SOFTWARE BE LIABLE FOR ANY DIRECT, INDIRECT,
> INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
> BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
> LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
> ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
>
> The names of the authors and copyright holders must not be used in
> advertising or otherwise to promote the sale, use or other dealing
> in this Software without specific, written prior permission.  Title
> to copyright in this Software shall at all times remain with copyright
> holders.
>
> OpenLDAP is a registered trademark of the OpenLDAP Foundation.
>
> Copyright 1999-2003 The OpenLDAP Foundation, Redwood City,
> California, USA.  All Rights Reserved.  Permission to copy and
> distribute verbatim copies of this document is granted.

---

## pugixml

The `vpipe` shared library statically links pugixml
(<https://github.com/zeux/pugixml>), vendored as a git submodule at
`extern/pugixml/`. Only `extern/pugixml/src/pugixml.cpp` is compiled
into the `pugixml` static target; the headers, source, and license
are preserved unmodified under `extern/pugixml/`.

pugixml is distributed under the MIT License. The verbatim license
text is preserved at `extern/pugixml/LICENSE.md`.

### Copyright notice

> Copyright (c) 2006-2023 Arseny Kapoulkine

### License

Reproduced verbatim from `extern/pugixml/LICENSE.md`:

> MIT License
>
> Copyright (c) 2006-2023 Arseny Kapoulkine
>
> Permission is hereby granted, free of charge, to any person
> obtaining a copy of this software and associated documentation
> files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use,
> copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
> OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
> HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
> WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
> OTHER DEALINGS IN THE SOFTWARE.

---

## pocketfft

The `vpipe` shared library compiles the header-only pocketfft FFT
library (<https://gitlab.mpcdf.mpg.de/mtr/pocketfft>), vendored at
`3rd-party/pocketfft.h`. It is `#include`d by the audio feature
extractors (`generative-models/shared/whisper-feature-extractor.cc`
and `generative-models/gemma4/gemma4-audio-feature-extractor.cc`) and
inlined into `libvpipe`.

pocketfft is distributed under the BSD 3-Clause License. The verbatim
license text is preserved in the header at `3rd-party/pocketfft.h` and
reproduced below.

### Copyright notice

> Copyright (C) 2010-2022 Max-Planck-Society
> Copyright (C) 2019-2020 Peter Bell
>
> For the odd-sized DCT-IV transforms:
>   Copyright (C) 2003, 2007-14 Matteo Frigo
>   Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology
>
> Authors: Martin Reinecke, Peter Bell

### License

Reproduced verbatim from `3rd-party/pocketfft.h`:

> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification,
> are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice, this
>   list of conditions and the following disclaimer in the documentation and/or
>   other materials provided with the distribution.
> * Neither the name of the copyright holder nor the names of its contributors may
>   be used to endorse or promote products derived from this software without
>   specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
> ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## nanobind

The Python extension module `_vpipe` (built under `python/`) statically
links nanobind (<https://github.com/wjakob/nanobind>), vendored as a
git submodule at `extern/nanobind/` and compiled with `NB_STATIC`.
nanobind is linked only into the Python extension module, not into
`libvpipe` or the `vpipe` executables.

nanobind is distributed under the BSD 3-Clause License. The verbatim
license text is reproduced below from `extern/nanobind/LICENSE`.

### Copyright notice

> Copyright (c) 2022 Wenzel Jakob

### License

Reproduced verbatim from `extern/nanobind/LICENSE`:

> Copyright (c) 2022 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## metal-cpp (Foundation headers)

When `vpipe` is built on Apple platforms with `VPIPE_BUILD_APPLE_SILICON`
enabled (the default on Apple), it consumes the `Foundation/` headers
from metal-cpp (<https://github.com/bkaradzic/metal-cpp>), vendored as
a git submodule at `extern/metal-cpp/`. The headers are pulled in by
the CoreML C++ wrapper under `apple-silicon/coreml/coreml-cpp/` and
inlined into `libvpipe`.

metal-cpp is distributed under the Apache License, Version 2.0 — the
same license as vpipe itself. The full verbatim text is in the root
`LICENSE` file; an upstream copy is also preserved at
`extern/metal-cpp/LICENSE.txt` when the submodule is checked out.

### Copyright notice

> Copyright 2020-2024 Apple Inc.

---

## MLX (vendored Metal kernels)

`vpipe` does not link the MLX library; it vendors only a small set of
MLX's Metal kernel headers (<https://github.com/ml-explore/mlx>) — the
"steel" GEMM / attention templates and their supporting helpers — under
`gpu-kernels/metal/vendored/mlx/`. Those headers are `#include`d by
vpipe's own `.metal` kernel sources and compiled into the embedded
metallibs shipped in `libvpipe`. No other MLX code is present or
compiled.

MLX is distributed under the MIT License. The vendored headers carry
`Copyright © 2023-2025 Apple Inc.`; the verbatim license text is
reproduced below.

### Copyright notice

> Copyright © 2023-2025 Apple Inc.

### License

Reproduced verbatim from MLX's `LICENSE`:

> MIT License
>
> Copyright © 2023 Apple Inc.
>
> Permission is hereby granted, free of charge, to any person
> obtaining a copy of this software and associated documentation
> files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy,
> modify, merge, publish, distribute, sublicense, and/or sell copies
> of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
> BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
> ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
> CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

---

## ByteTrack

The `byte-track` stage at `stages/byte-track-stage.{h,cc}` is a
derivative work of ByteTrack
(<https://github.com/FoundationVision/ByteTrack>), specifically the
reference C++ deployment at `deploy/ncnn/cpp/`. The tracker
algorithm, `STrack` bookkeeping, and the dense Jonker-Volgenant
linear-assignment solver (`lapjv_internal_` and helpers) are ported
into the stage's translation unit. The reference's Eigen-backed
8-dimensional Kalman filter and OpenCV bbox primitives are
re-implemented inline with `std::array` arithmetic so vpipe does
not take a dependency on Eigen or OpenCV.

ByteTrack is distributed under the MIT License.

### Copyright notice

> Copyright (c) 2021 Yifu Zhang

### License

Reproduced verbatim from
<https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE>:

> MIT License
>
> Copyright (c) 2021 Yifu Zhang
>
> Permission is hereby granted, free of charge, to any person
> obtaining a copy of this software and associated documentation
> files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use,
> copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following
> conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
> OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
> HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
> WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
> OTHER DEALINGS IN THE SOFTWARE.
