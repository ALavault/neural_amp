#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

namespace fssr::r1
{
class Model
{
public:
  explicit Model(const std::filesystem::path& model_path);
  ~Model();

  Model(Model&&) noexcept;
  Model& operator=(Model&&) noexcept;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void reset() noexcept;
  void process(const float* input, float* output, std::size_t sample_count) noexcept;

  [[nodiscard]] int sample_rate_hz() const noexcept;
  [[nodiscard]] int latency_samples() const noexcept;
  [[nodiscard]] std::size_t state_size_bytes() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace fssr::r1
