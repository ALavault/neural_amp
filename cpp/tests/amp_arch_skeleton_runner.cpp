#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "amp_arch_skeleton.hpp"

namespace
{
std::atomic<std::size_t> allocation_count{0};
}

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

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: amp_arch_skeleton_runner FAMILY PROFILE\n";
    return 2;
  }
  try
  {
    constexpr std::size_t samples = 257;
    std::vector<float> input(samples);
    std::vector<float> complete(samples, 0.0f);
    std::vector<float> blocked(samples, 0.0f);
    std::vector<float> repeated(samples, 0.0f);
    for (std::size_t index = 0; index < samples; ++index)
      input[index] = static_cast<float>(
        0.3 * std::sin(0.11 * static_cast<double>(index))
        + 0.07 * std::cos(0.037 * static_cast<double>(index)));
    fssr::arch::Skeleton model(argv[1], argv[2]);
    model.process(input.data(), complete.data(), samples);
    model.reset();
    constexpr std::size_t chunks[] = {1, 63, 5, 97, 17, 74};
    std::size_t offset = 0;
    for (const auto chunk : chunks)
    {
      model.process(input.data() + offset, blocked.data() + offset, chunk);
      offset += chunk;
    }
    if (offset != samples)
      throw std::runtime_error("irregular block fixture has wrong length");
    model.reset();
    const auto allocations_before = allocation_count.load(std::memory_order_relaxed);
    model.process(input.data(), repeated.data(), samples);
    const auto allocations_after = allocation_count.load(std::memory_order_relaxed);
    float block_error = 0.0f;
    float reset_error = 0.0f;
    bool finite = true;
    for (std::size_t index = 0; index < samples; ++index)
    {
      block_error = std::max(block_error, std::abs(complete[index] - blocked[index]));
      reset_error = std::max(reset_error, std::abs(complete[index] - repeated[index]));
      finite = finite && std::isfinite(complete[index]) && std::isfinite(blocked[index])
               && std::isfinite(repeated[index]);
    }
    std::cout << std::setprecision(12)
              << "{\"format\":\"fssr-amp-arch-skeleton-check-v1\""
              << ",\"family\":\"" << model.family() << "\""
              << ",\"profile\":\"" << model.profile() << "\""
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"block_max_abs_error\":" << block_error
              << ",\"reset_max_abs_error\":" << reset_error
              << ",\"audio_loop_allocations\":"
              << allocations_after - allocations_before
              << ",\"latency_samples\":" << model.latency_samples()
              << ",\"parameters\":" << model.parameters()
              << ",\"weight_bytes\":" << model.weight_bytes()
              << ",\"persistent_state_bytes\":" << model.persistent_state_bytes()
              << ",\"scratch_bytes\":" << model.scratch_bytes()
              << ",\"estimated_macs_per_sample\":"
              << model.estimated_macs_per_sample() << "}\n";
    return finite && block_error <= 2.0e-6f && reset_error <= 2.0e-6f
               && allocations_after == allocations_before
             ? 0
             : 1;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
