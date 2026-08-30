#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

namespace fssr::sota
{
enum class Variant
{
  control_delay32,
  candidate_delay24,
};

[[nodiscard]] Variant parse_variant(std::string_view identifier);
[[nodiscard]] std::string_view variant_identifier(Variant variant) noexcept;

// Executable native scaffold for AMP-SOTA-PROTOTYPE-v1.1. It deliberately
// implements delayed identity only; learned-layer export is not yet supported.
class Prototype final
{
public:
  explicit Prototype(Variant variant) noexcept;

  void reset() noexcept;
  void process(const float* input, float* output, std::size_t sample_count) noexcept;

  // Refuse learned weights until a frozen native export format exists. This
  // prevents a caller from silently benchmarking the delayed-identity scaffold
  // as though it were a trained candidate.
  [[noreturn]] void load_learned_weights(std::span<const float> weights);

  [[nodiscard]] Variant variant() const noexcept;
  [[nodiscard]] std::string_view identifier() const noexcept;
  [[nodiscard]] std::string_view implementation() const noexcept;
  [[nodiscard]] int latency_samples() const noexcept;
  [[nodiscard]] std::size_t parameters() const noexcept;
  [[nodiscard]] std::size_t weight_bytes() const noexcept;
  [[nodiscard]] std::size_t persistent_state_bytes() const noexcept;
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;
  [[nodiscard]] bool learned_weights_supported() const noexcept;

private:
  static constexpr std::size_t maximum_latency_ = 32;

  std::array<float, maximum_latency_> delay_{};
  std::size_t write_index_ = 0;
  Variant variant_;
};
} // namespace fssr::sota
