#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>

#include "sota_prototype.hpp"

namespace
{
std::atomic<std::size_t> allocation_count{0};

struct Result
{
  std::string_view identifier;
  int latency_samples;
  float identity_error;
  float block_error;
  float reset_error;
  float in_place_error;
  std::size_t audio_loop_allocations;
  bool finite;
  bool weights_refused;
};

template <std::size_t Size>
float maximum_error(
  const std::array<float, Size>& first,
  const std::array<float, Size>& second) noexcept
{
  float error = 0.0f;
  for (std::size_t index = 0; index < Size; ++index)
    error = std::max(error, std::abs(first[index] - second[index]));
  return error;
}

Result check(fssr::sota::Variant variant)
{
  constexpr std::size_t sample_count = 1031;
  constexpr std::array<std::size_t, 9> chunks = {1, 127, 3, 64, 5, 257, 2, 128, 31};
  std::array<float, sample_count> input{};
  std::array<float, sample_count> expected{};
  std::array<float, sample_count> complete{};
  std::array<float, sample_count> blocked{};
  std::array<float, sample_count> repeated{};
  std::array<float, sample_count> in_place{};

  fssr::sota::Prototype model(variant);
  const auto latency = static_cast<std::size_t>(model.latency_samples());
  for (std::size_t index = 0; index < sample_count; ++index)
  {
    input[index] = static_cast<float>(
      0.31 * std::sin(0.071 * static_cast<double>(index))
      + 0.08 * std::cos(0.013 * static_cast<double>(index)));
    expected[index] = index < latency ? 0.0f : input[index - latency];
  }
  in_place = input;

  const auto allocations_before = allocation_count.load(std::memory_order_relaxed);
  model.process(input.data(), complete.data(), sample_count);
  model.reset();
  std::size_t offset = 0;
  std::size_t chunk_index = 0;
  while (offset < sample_count)
  {
    const auto count = std::min(chunks[chunk_index % chunks.size()], sample_count - offset);
    model.process(input.data() + offset, blocked.data() + offset, count);
    offset += count;
    ++chunk_index;
  }
  model.reset();
  model.process(input.data(), repeated.data(), sample_count);
  model.reset();
  model.process(in_place.data(), in_place.data(), sample_count);
  model.process(nullptr, nullptr, 0);
  const auto allocations_after = allocation_count.load(std::memory_order_relaxed);

  bool finite = true;
  for (std::size_t index = 0; index < sample_count; ++index)
  {
    finite = finite && std::isfinite(complete[index]) && std::isfinite(blocked[index])
             && std::isfinite(repeated[index]) && std::isfinite(in_place[index]);
  }

  bool weights_refused = false;
  try
  {
    const std::array<float, 1> unsupported_weights = {1.0f};
    model.load_learned_weights(unsupported_weights);
  }
  catch (const std::runtime_error&)
  {
    weights_refused = true;
  }

  return {
    model.identifier(),
    model.latency_samples(),
    maximum_error(complete, expected),
    maximum_error(complete, blocked),
    maximum_error(complete, repeated),
    maximum_error(in_place, expected),
    allocations_after - allocations_before,
    finite,
    weights_refused};
}

bool passed(const Result& result) noexcept
{
  constexpr float tolerance = 2.0e-5f;
  return result.finite && result.weights_refused
         && result.identity_error <= tolerance && result.block_error <= tolerance
         && result.reset_error <= tolerance && result.in_place_error <= tolerance
         && result.audio_loop_allocations == 0;
}

void print(const Result& result)
{
  std::cout << "{\"identifier\":\"" << result.identifier << "\""
            << ",\"latency_samples\":" << result.latency_samples
            << ",\"identity_max_abs_error\":" << result.identity_error
            << ",\"block_max_abs_error\":" << result.block_error
            << ",\"reset_max_abs_error\":" << result.reset_error
            << ",\"in_place_max_abs_error\":" << result.in_place_error
            << ",\"audio_loop_allocations\":" << result.audio_loop_allocations
            << ",\"finite\":" << (result.finite ? "true" : "false")
            << ",\"learned_weights_refused\":"
            << (result.weights_refused ? "true" : "false") << "}";
}
} // namespace

void* operator new(std::size_t size)
{
  allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size))
    return pointer;
  throw std::bad_alloc();
}

void operator delete(void* pointer) noexcept
{
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept
{
  std::free(pointer);
}

int main()
{
  const auto control = check(fssr::sota::Variant::control_delay32);
  const auto candidate = check(fssr::sota::Variant::candidate_delay24);
  const bool success = passed(control) && passed(candidate);
  std::cout << std::setprecision(12)
            << "{\"format\":\"fssr-sota-native-block-check-v1\""
            << ",\"implementation\":\"delayed_identity_scaffold\""
            << ",\"learned_weights_supported\":false"
            << ",\"profiles\":[";
  print(control);
  std::cout << ',';
  print(candidate);
  std::cout << "],\"passed\":" << (success ? "true" : "false") << "}\n";
  return success ? 0 : 1;
}
