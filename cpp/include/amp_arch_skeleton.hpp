#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace fssr::arch
{
class Skeleton
{
public:
  Skeleton(const std::string& family, const std::string& profile);
  ~Skeleton();

  Skeleton(Skeleton&&) noexcept;
  Skeleton& operator=(Skeleton&&) noexcept;
  Skeleton(const Skeleton&) = delete;
  Skeleton& operator=(const Skeleton&) = delete;

  void reset() noexcept;
  void process(const float* input, float* output, std::size_t sample_count) noexcept;

  [[nodiscard]] const std::string& family() const noexcept;
  [[nodiscard]] const std::string& profile() const noexcept;
  [[nodiscard]] int latency_samples() const noexcept;
  [[nodiscard]] std::size_t parameters() const noexcept;
  [[nodiscard]] std::size_t weight_bytes() const noexcept;
  [[nodiscard]] std::size_t persistent_state_bytes() const noexcept;
  [[nodiscard]] std::size_t scratch_bytes() const noexcept;
  [[nodiscard]] std::size_t estimated_macs_per_sample() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace fssr::arch
