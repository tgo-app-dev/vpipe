#ifndef VPIPE_APPLE_SILICON_METAL_COMPUTE_FENCE_H
#define VPIPE_APPLE_SILICON_METAL_COMPUTE_FENCE_H

namespace MTL { class Fence; }

namespace vpipe::metal_compute {

class MetalCompute;

// Lightweight, intra-CB synchronisation primitive (MTL::Fence). Sits
// alongside Event (MTL::SharedEvent): a Fence ordered encoders
// within ONE command buffer; an Event signals across buffers /
// streams.
//
// Typical use:
//   auto fence = mc.make_fence();
//   auto buf_a = mc.make_shared_buffer(N,
//                  /*alignment=*/64,
//                  HazardTracking::Untracked);
//   auto buf_b = mc.make_shared_buffer(N,
//                  /*alignment=*/64,
//                  HazardTracking::Untracked);
//   {
//     auto enc = stream.begin_compute();
//     enc.set_function(fill);
//     enc.set_buffer(0, buf_a);
//     enc.dispatch(...);
//     enc.update_fence(fence);          // signal once A's work
//                                       // finishes
//   }
//   {
//     auto enc = stream.begin_compute();
//     enc.wait_for_fence(fence);        // park until A's work
//                                       // retires
//     enc.set_function(copy);
//     enc.set_buffer(0, buf_a);
//     enc.set_buffer(1, buf_b);
//     enc.dispatch(...);
//   }
//   stream.commit().wait();
//
// On Untracked-hazard buffers the encoder pair would race without
// the fence; with the fence the dependency is explicit.
class Fence {
public:
  Fence() noexcept = default;
  Fence(Fence&&) noexcept;
  Fence& operator=(Fence&&) noexcept;
  Fence(const Fence&)            = delete;
  Fence& operator=(const Fence&) = delete;
  ~Fence();

  bool valid() const noexcept { return _fence != nullptr; }

  MTL::Fence* mtl_fence() const noexcept { return _fence; }

private:
  friend class MetalCompute;
  explicit Fence(MTL::Fence* f) noexcept;

  MTL::Fence* _fence = nullptr;
};

}  // namespace vpipe::metal_compute

#endif
