#include "sota_prototype.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace fssr::sota
{
namespace
{
static_assert(sizeof(float) == 4, "AMP-SOTA native ABI requires 32-bit float");
static_assert(
  std::numeric_limits<float>::is_iec559,
  "AMP-SOTA native ABI requires IEEE-754 float");
} // namespace

Variant parse_variant(std::string_view identifier)
{
  if (identifier == "control" || identifier == "control_identity_delay32")
    return Variant::control_delay32;
  if (identifier == "candidate" || identifier == "candidate_identity_delay24")
    return Variant::candidate_delay24;
  throw std::invalid_argument("unknown AMP-SOTA native scaffold variant");
}

std::string_view variant_identifier(Variant variant) noexcept
{
  switch (variant)
  {
  case Variant::control_delay32:
    return "control_identity_delay32";
  case Variant::candidate_delay24:
    return "candidate_identity_delay24";
  }
  return "invalid_variant";
}

Prototype::Prototype(Variant variant) noexcept : variant_(variant)
{
  reset();
}

void Prototype::reset() noexcept
{
  std::fill(delay_.begin(), delay_.end(), 0.0f);
  write_index_ = 0;
}

void Prototype::process(
  const float* input,
  float* output,
  std::size_t sample_count) noexcept
{
  const auto latency = static_cast<std::size_t>(latency_samples());
  for (std::size_t index = 0; index < sample_count; ++index)
  {
    const float current = input[index];
    const float delayed = delay_[write_index_];
    delay_[write_index_] = current;
    output[index] = delayed;
    ++write_index_;
    if (write_index_ == latency)
      write_index_ = 0;
  }
}

void Prototype::load_learned_weights(std::span<const float>)
{
  throw std::runtime_error(
    "learned AMP-SOTA weights are unsupported by delayed_identity_scaffold");
}

Variant Prototype::variant() const noexcept
{
  return variant_;
}

std::string_view Prototype::identifier() const noexcept
{
  return variant_identifier(variant_);
}

std::string_view Prototype::implementation() const noexcept
{
  return "delayed_identity_scaffold";
}

int Prototype::latency_samples() const noexcept
{
  return variant_ == Variant::control_delay32 ? 32 : 24;
}

std::size_t Prototype::parameters() const noexcept
{
  return 0;
}

std::size_t Prototype::weight_bytes() const noexcept
{
  return 0;
}

std::size_t Prototype::persistent_state_bytes() const noexcept
{
  return delay_.size() * sizeof(float) + sizeof(write_index_);
}

std::size_t Prototype::scratch_bytes() const noexcept
{
  return 0;
}

bool Prototype::learned_weights_supported() const noexcept
{
  return false;
}
} // namespace fssr::sota
