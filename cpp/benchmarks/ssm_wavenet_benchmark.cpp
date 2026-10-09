#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ssm_wavenet_native.hpp"

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
  if (argc != 2 && argc != 4)
  {
    std::cerr << "Usage: ssm_wavenet_benchmark MODEL [GAIN CONTOUR]\n";
    return 2;
  }
  try
  {
    // Same stimulus, warmup, timing and median as nam_benchmark MODEL BLOCK 2 5.
    for (const int block_size : {32, 64, 128, 256})
    {
      const int seconds = 2;
      const int iterations = 5;
      fssr::ssm::Model model(argv[1]);
      if (argc == 4)
        model.set_controls(std::vector<float>{std::stof(argv[2]), std::stof(argv[3])});
      constexpr double sample_rate = 48000.0;
      const int total_samples = seconds * static_cast<int>(sample_rate);
      if (total_samples % block_size != 0)
        throw std::runtime_error("audio length must be divisible by block size");
      std::vector<float> input(block_size);
      std::vector<float> output(block_size);
      for (int index = 0; index < block_size; ++index)
      {
        const double time = static_cast<double>(index) / sample_rate;
        input[index] = static_cast<float>(0.25 * std::sin(2.0 * M_PI * 220.0 * time)
                                              + 0.1 * std::sin(2.0 * M_PI * 1230.0 * time));
      }
      model.set_sample_rate(sample_rate);
      model.reset();
      const int blocks_per_iteration = total_samples / block_size;
      for (int warmup = 0; warmup < 3; ++warmup)
        for (int block = 0; block < blocks_per_iteration; ++block)
          model.process(input.data(), output.data(), block_size);
      std::vector<double> nanoseconds;
      nanoseconds.reserve(static_cast<std::size_t>(iterations) * blocks_per_iteration);
      for (int iteration = 0; iteration < iterations; ++iteration)
      {
        for (int block = 0; block < blocks_per_iteration; ++block)
        {
          const auto start = std::chrono::steady_clock::now();
          model.process(input.data(), output.data(), block_size);
          const auto stop = std::chrono::steady_clock::now();
          nanoseconds.push_back(
            std::chrono::duration<double, std::nano>(stop - start).count());
        }
      }
      std::sort(nanoseconds.begin(), nanoseconds.end());
      const double mean = std::accumulate(nanoseconds.begin(), nanoseconds.end(), 0.0) / nanoseconds.size();
      const double median = percentile(nanoseconds, 0.5);
      const double p95 = percentile(nanoseconds, 0.95);
      const double audio_ns = 1.0e9 * block_size / sample_rate;
      std::cout << std::setprecision(12)
                << "{\"block_size\":" << block_size << ",\"iterations\":" << iterations
                << ",\"timed_blocks\":" << nanoseconds.size() << ",\"mean_block_ns\":" << mean
                << ",\"median_block_ns\":" << median << ",\"p95_block_ns\":" << p95
                << ",\"median_ns_per_sample\":" << median / block_size
                << ",\"p95_ns_per_sample\":" << p95 / block_size
                << ",\"median_realtime_factor\":" << audio_ns / median
                << ",\"p95_realtime_factor\":" << audio_ns / p95 << "}\n";
    }
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
