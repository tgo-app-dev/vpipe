#ifndef VPIPE_GENERATIVE_MODELS_SHARED_TORCH_ZIP_H
#define VPIPE_GENERATIVE_MODELS_SHARED_TORCH_ZIP_H

// Reading a torch.save() checkpoint without torch.
//
// WHAT THE FORMAT IS, which is what keeps this small. Since torch 1.6 a
// `.pt` / `.pth` / `.ckpt` is a ZIP archive whose entries are STORED,
// never deflated: `<archive>/data.pkl`, a pickle describing the object
// graph, and one raw little-endian byte blob per tensor storage at
// `<archive>/data/<key>`, each padded to a 64-byte boundary. The tensor
// bytes are therefore already lying in the file, contiguous, at offsets
// the archive's own directory gives -- exactly as they lie in a
// safetensors file -- and the pickle is only the table of contents.
//
// So a torch file can be MAPPED as a shard. This reads the table and
// hands back (name, dtype, shape, file offset, byte count) per tensor,
// and MetalLlamaWeights serves every read path from that unchanged:
// copy, pread, zero-copy map.
//
// WHAT IT REFUSES TO DO IS EXECUTE ANYTHING. A pickle is a program, and
// torch.load() runs it with whatever imports it names. This interpreter
// builds DATA -- dicts, tuples, strings, numbers, storages, tensors --
// and recognises a short allowlist of globals BY NAME, calling none of
// them: `collections.OrderedDict` (and `dict`), torch's
// `_rebuild_tensor_v2` / `_rebuild_tensor` / `_rebuild_parameter`, its
// storage classes and dtypes, and `torch.Size`. Any other global fails
// the read, naming it. That is all a plain state_dict needs -- MEASURED
// against FlashVSR-v1.1's three torch files: an OrderedDict of bf16
// tensors, a 194-tensor fp32 Wan VAE whose tensors are views into a few
// shared storages at non-zero offsets, and a bare bf16 tensor.
//
// REFUSED, each with its reason rather than guessed at: the legacy
// pre-1.6 tar/pickle format, deflated entries, a big-endian archive,
// and a tensor whose strides are not C order (it would need a copy, and
// this layer maps; it does not copy).

#include <cstdint>
#include <string>
#include <vector>

namespace vpipe::genai::torch_zip {

struct Tensor {
  // A state_dict key, with nested dicts joined by '.' (a training
  // checkpoint's `state_dict` unwrapped; its other dicts, e.g.
  // `ema_state_dict`, kept under their own keys). A checkpoint whose
  // top-level object is a bare tensor names it after its archive, which
  // is what torch.save() named the file's root ("posi_prompt").
  std::string               name;
  // The safetensors spelling: F32, F64, F16, BF16, I64, I32, I16, I8,
  // U8, BOOL.
  std::string               dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t             offset = 0;   // from the START of the file
  std::uint64_t             nbytes = 0;
};

// True when the file starts with a ZIP local header. Cheap: one read of
// four bytes. Says nothing about whether the archive is a torch one.
bool looks_like_zip(int fd);

// Read the tensor table of an open torch zip `size` bytes long. False
// with `err` set on anything this reader does not accept; `out` is then
// left empty.
bool read_index(int fd, std::uint64_t size, std::vector<Tensor>* out,
                std::string* err);

}  // namespace vpipe::genai::torch_zip

#endif  // VPIPE_GENERATIVE_MODELS_SHARED_TORCH_ZIP_H
