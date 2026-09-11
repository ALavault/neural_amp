#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "fssr_r1_native.hpp"
#include "json.hpp"

namespace
{
using json = nlohmann::json;

struct Summary
{
  double median = 0.0;
  double p95 = 0.0;
  std::size_t blocks = 0;
};

double percentile(const std::vector<double>& sorted, double fraction)
{
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Summary summarize(std::vector<double> values)
{
  if (values.empty())
    throw std::runtime_error("benchmark collected no timing values");
  std::sort(values.begin(), values.end());
  return {percentile(values, 0.5), percentile(values, 0.95), values.size()};
}

template <typename Callback>
double timed(Callback&& callback)
{
  const auto start = std::chrono::steady_clock::now();
  callback();
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(stop - start).count();
}

json block_result(const Summary& summary, int block_size)
{
  constexpr double sample_rate = 48000.0;
  const double audio_nanoseconds = 1.0e9 * static_cast<double>(block_size) / sample_rate;
  return {
    {"median_ns_per_sample", summary.median / block_size},
    {"p95_ns_per_sample", summary.p95 / block_size},
    {"rtf", summary.median / audio_nanoseconds},
    {"timed_blocks", summary.blocks}};
}

std::size_t nonnegative_size(const char* text, const char* label)
{
  const long long value = std::stoll(text);
  if (value < 0)
    throw std::runtime_error(std::string(label) + " must be non-negative");
  return static_cast<std::size_t>(value);
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 10)
  {
    std::cerr << "Usage: r2_benchmark A2_MODEL R2_MODEL A2_PARAMETERS A2_WEIGHT_BYTES "
                 "A2_STATE_BYTES A2_SCRATCH_BYTES PARITY_MAX_ERROR SECONDS REPETITIONS\n";
    return 2;
  }
  try
  {
    const std::filesystem::path a2_path(argv[1]);
    const std::filesystem::path r2_path(argv[2]);
    const auto a2_parameters = nonnegative_size(argv[3], "A2 parameters");
    const auto a2_weight_bytes = nonnegative_size(argv[4], "A2 weight bytes");
    const auto a2_state_bytes = nonnegative_size(argv[5], "A2 state bytes");
    const auto a2_scratch_bytes = nonnegative_size(argv[6], "A2 scratch bytes");
    const double parity_error = std::stod(argv[7]);
    const int seconds = std::stoi(argv[8]);
    const int repetitions = std::stoi(argv[9]);
    if (!std::isfinite(parity_error) || parity_error < 0.0 || seconds < 1
        || repetitions < 1)
      throw std::runtime_error("parity error, seconds, or repetitions are invalid");
    auto a2 = nam::get_dsp(a2_path);
    if (!a2 || a2->NumInputChannels() != 1 || a2->NumOutputChannels() != 1)
      throw std::runtime_error("A2 model must load as mono NAM DSP");
    fssr::r1::Model r2(r2_path);
    std::ifstream r2_stream(r2_path);
    json r2_payload;
    r2_stream >> r2_payload;
    const auto& r2_sizes = r2_payload.at("sizes");
    constexpr int sample_rate = 48000;
    constexpr int block_sizes[] = {1, 16, 64, 128};
    json a2_blocks = json::object();
    json r2_blocks = json::object();
    double checksum = 0.0;
    for (const int block_size : block_sizes)
    {
      const int total_samples = seconds * sample_rate;
      if (total_samples % block_size != 0)
        throw std::runtime_error("benchmark duration must divide every block size");
      const int blocks_per_repetition = total_samples / block_size;
      std::vector<NAM_SAMPLE> input(block_size);
      std::vector<float> r2_input(block_size);
      std::vector<NAM_SAMPLE> a2_output(block_size, 0.0f);
      std::vector<float> r2_output(block_size, 0.0f);
      NAM_SAMPLE* input_ptrs[] = {input.data()};
      NAM_SAMPLE* a2_output_ptrs[] = {a2_output.data()};
      for (int index = 0; index < block_size; ++index)
      {
        const double time = static_cast<double>(index) / sample_rate;
        input[index] = static_cast<NAM_SAMPLE>(
          0.25 * std::sin(2.0 * M_PI * 220.0 * time)
          + 0.1 * std::sin(2.0 * M_PI * 1230.0 * time));
        r2_input[index] = static_cast<float>(input[index]);
      }
      a2->Reset(sample_rate, block_size);
      r2.reset();
      for (int warmup = 0; warmup < 3; ++warmup)
      {
        a2->process(input_ptrs, a2_output_ptrs, block_size);
        r2.process(
          r2_input.data(), r2_output.data(), static_cast<std::size_t>(block_size));
      }
      std::vector<double> a2_timings;
      std::vector<double> r2_timings;
      a2_timings.reserve(static_cast<std::size_t>(repetitions) * blocks_per_repetition);
      r2_timings.reserve(static_cast<std::size_t>(repetitions) * blocks_per_repetition);
      for (int repetition = 0; repetition < repetitions; ++repetition)
      {
        a2->Reset(sample_rate, block_size);
        r2.reset();
        for (int block = 0; block < blocks_per_repetition; ++block)
        {
          const auto measure_a2 = [&] {
            a2_timings.push_back(timed(
              [&] { a2->process(input_ptrs, a2_output_ptrs, block_size); }));
          };
          const auto measure_r2 = [&] {
            r2_timings.push_back(timed([&] {
              r2.process(
                r2_input.data(), r2_output.data(), static_cast<std::size_t>(block_size));
            }));
          };
          if ((repetition + block) % 2 == 0)
          {
            measure_a2();
            measure_r2();
          }
          else
          {
            measure_r2();
            measure_a2();
          }
        }
        checksum += a2_output.back() + r2_output.back();
      }
      a2_blocks[std::to_string(block_size)] = block_result(summarize(a2_timings), block_size);
      r2_blocks[std::to_string(block_size)] = block_result(summarize(r2_timings), block_size);
    }
    json report = {
      {"format", "fssr-r2-interleaved-benchmark-v1"},
      {"same_binary", true},
      {"abi", "float32"},
      {"compiler_optimization", "-Ofast"},
      {"lto_ipo", true},
      {"same_isa", true},
      {"repetitions", repetitions},
      {"scientific_eligible", repetitions == 30},
      {"schedule", "interleaved_ab"},
      {"allocations_outside_timing", true},
      {"python_cpp_max_abs_error", parity_error},
      {"checksum", checksum},
      {"models",
       {{"a2",
         {{"parameters", a2_parameters},
          {"weight_bytes", a2_weight_bytes},
          {"persistent_state_bytes", a2_state_bytes},
          {"scratch_bytes", a2_scratch_bytes},
          {"latency_samples", 0},
          {"blocks", a2_blocks}}},
        {"candidate",
         {{"parameters", r2_sizes.at("parameters")},
          {"weight_bytes", r2_sizes.at("weight_bytes")},
          {"persistent_state_bytes", r2_sizes.at("persistent_state_bytes")},
          {"scratch_bytes", r2_sizes.at("scratch_bytes")},
          {"latency_samples", r2.latency_samples()},
          {"native_state_allocation_bytes", r2.state_size_bytes()},
          {"blocks", r2_blocks}}}}}};
    std::cout << std::setprecision(12) << report.dump() << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
