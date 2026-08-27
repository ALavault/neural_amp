#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "fssr_r1_native.hpp"

namespace
{
double percentile(const std::vector<double>& sorted, double fraction)
{
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cerr << "Usage: r1_benchmark MODEL SECONDS ITERATIONS\n";
    return 2;
  }
  try
  {
    constexpr std::size_t block_size = 64;
    constexpr double pi = 3.14159265358979323846;
    const int seconds = std::stoi(argv[2]);
    const int iterations = std::stoi(argv[3]);
    if (seconds < 1 || iterations < 1)
      throw std::runtime_error("seconds and iterations must be positive");
    fssr::r1::Model model(argv[1]);
    const auto sample_rate = model.sample_rate_hz();
    const auto blocks_per_iteration =
      (static_cast<std::size_t>(seconds) * sample_rate + block_size - 1) / block_size;
    std::vector<float> input(block_size, 0.0f);
    std::vector<float> output(block_size, 0.0f);
    for (std::size_t index = 0; index < block_size; ++index)
    {
      const double time = static_cast<double>(index) / sample_rate;
      input[index] = static_cast<float>(
        0.25 * std::sin(2.0 * pi * 220.0 * time) + 0.1 * std::sin(2.0 * pi * 1230.0 * time));
    }
    model.reset();
    for (int warmup = 0; warmup < 3; ++warmup)
      for (std::size_t block = 0; block < blocks_per_iteration; ++block)
        model.process(input.data(), output.data(), block_size);
    std::vector<double> nanoseconds;
    nanoseconds.reserve(static_cast<std::size_t>(iterations) * blocks_per_iteration);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
      for (std::size_t block = 0; block < blocks_per_iteration; ++block)
      {
        const auto start = std::chrono::steady_clock::now();
        model.process(input.data(), output.data(), block_size);
        const auto stop = std::chrono::steady_clock::now();
        nanoseconds.push_back(
          std::chrono::duration<double, std::nano>(stop - start).count());
      }
    }
    std::sort(nanoseconds.begin(), nanoseconds.end());
    const double mean =
      std::accumulate(nanoseconds.begin(), nanoseconds.end(), 0.0) / nanoseconds.size();
    const double median = percentile(nanoseconds, 0.5);
    const double p95 = percentile(nanoseconds, 0.95);
    const double audio_nanoseconds = 1.0e9 * block_size / sample_rate;
    std::cout << std::setprecision(12) << "{\"format\":\"fssr-r1-benchmark-v1\""
              << ",\"block_size\":" << block_size << ",\"sample_rate_hz\":" << sample_rate
              << ",\"iterations\":" << iterations << ",\"timed_blocks\":" << nanoseconds.size()
              << ",\"mean_block_ns\":" << mean << ",\"median_block_ns\":" << median
              << ",\"p95_block_ns\":" << p95
              << ",\"median_ns_per_sample\":" << median / block_size
              << ",\"p95_ns_per_sample\":" << p95 / block_size
              << ",\"median_realtime_factor\":" << audio_nanoseconds / median
              << ",\"p95_realtime_factor\":" << audio_nanoseconds / p95
              << ",\"latency_samples\":" << model.latency_samples()
              << ",\"state_size_bytes\":" << model.state_size_bytes() << "}\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
