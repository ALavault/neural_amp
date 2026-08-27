#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

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
  if (argc != 5)
  {
    std::cerr << "Usage: nam_benchmark MODEL BLOCK_SIZE SECONDS ITERATIONS\n";
    return 2;
  }
  try
  {
    const int block_size = std::stoi(argv[2]);
    const int seconds = std::stoi(argv[3]);
    const int iterations = std::stoi(argv[4]);
    if (block_size < 1 || block_size > 4096 || seconds < 1 || iterations < 1)
      throw std::runtime_error("invalid benchmark dimensions");
    auto model = nam::get_dsp(std::filesystem::path(argv[1]));
    if (!model)
      throw std::runtime_error("model failed to load");
    constexpr double sample_rate = 48000.0;
    const int total_samples = seconds * static_cast<int>(sample_rate);
    if (total_samples % block_size != 0)
      throw std::runtime_error("audio length must be divisible by block size");
    std::vector<NAM_SAMPLE> input(block_size);
    std::vector<NAM_SAMPLE> output(block_size);
    NAM_SAMPLE* input_ptrs[] = {input.data()};
    NAM_SAMPLE* output_ptrs[] = {output.data()};
    for (int index = 0; index < block_size; ++index)
    {
      const double time = static_cast<double>(index) / sample_rate;
      input[index] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * time)
                                            + 0.1 * std::sin(2.0 * M_PI * 1230.0 * time));
    }
    model->Reset(sample_rate, block_size);
    const int blocks_per_iteration = total_samples / block_size;
    for (int warmup = 0; warmup < 3; ++warmup)
      for (int block = 0; block < blocks_per_iteration; ++block)
        model->process(input_ptrs, output_ptrs, block_size);
    std::vector<double> nanoseconds;
    nanoseconds.reserve(static_cast<std::size_t>(iterations) * blocks_per_iteration);
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
      for (int block = 0; block < blocks_per_iteration; ++block)
      {
        const auto start = std::chrono::steady_clock::now();
        model->process(input_ptrs, output_ptrs, block_size);
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
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
