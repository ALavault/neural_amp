#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "amp_arch_skeleton.hpp"
#include "json.hpp"

namespace
{
using json = nlohmann::json;

struct Summary
{
  double median_ns;
  double p95_ns;
  std::size_t blocks;
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
    throw std::runtime_error("benchmark collected no timings");
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

json timing_report(const Summary& summary)
{
  constexpr double block_audio_ns = 1.0e9 * 64.0 / 48000.0;
  return {
    {"block_size", 64},
    {"timed_blocks", summary.blocks},
    {"median_block_ns", summary.median_ns},
    {"p95_block_ns", summary.p95_ns},
    {"median_ns_per_sample", summary.median_ns / 64.0},
    {"p95_ns_per_sample", summary.p95_ns / 64.0},
    {"median_rtf", summary.median_ns / block_audio_ns},
    {"p95_rtf", summary.p95_ns / block_audio_ns},
  };
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 4)
  {
    std::cerr << "Usage: amp_arch_skeleton_benchmark A2_MODEL SECONDS REPETITIONS\n";
    return 2;
  }
  try
  {
    const std::filesystem::path a2_path(argv[1]);
    const int seconds = std::stoi(argv[2]);
    const int repetitions = std::stoi(argv[3]);
    if (seconds < 1 || repetitions < 1)
      throw std::runtime_error("seconds and repetitions must be positive");
    auto a2 = nam::get_dsp(a2_path);
    if (!a2 || a2->NumInputChannels() != 1 || a2->NumOutputChannels() != 1)
      throw std::runtime_error("A2 skeleton baseline must be a mono NAM model");
    constexpr std::array<const char*, 6> families = {
      "selective_s6_x2",
      "micro_tcn_x2",
      "phys_s6_tcn_x2",
      "phys_det_tcn_x2",
      "rf2047_tfilm_x2",
      "cascade_rf2047_tfilm_x2"};
    constexpr std::array<const char*, 4> profiles = {
      "slim", "balanced", "full", "max"};
    constexpr int block_size = 64;
    constexpr int sample_rate = 48000;
    const int blocks_per_repetition = seconds * sample_rate / block_size;
    std::vector<NAM_SAMPLE> a2_input(block_size);
    std::vector<NAM_SAMPLE> a2_output(block_size, 0.0f);
    std::vector<float> candidate_input(block_size);
    std::vector<float> candidate_output(block_size, 0.0f);
    for (int index = 0; index < block_size; ++index)
    {
      const double time = static_cast<double>(index) / sample_rate;
      const float sample = static_cast<float>(
        0.25 * std::sin(2.0 * 3.141592653589793 * 220.0 * time)
        + 0.1 * std::sin(2.0 * 3.141592653589793 * 1230.0 * time));
      a2_input[index] = sample;
      candidate_input[index] = sample;
    }
    NAM_SAMPLE* a2_input_ptrs[] = {a2_input.data()};
    NAM_SAMPLE* a2_output_ptrs[] = {a2_output.data()};
    json results = json::array();
    double checksum = 0.0;
    for (const char* family : families)
    {
      for (const char* profile : profiles)
      {
        fssr::arch::Skeleton candidate(family, profile);
        a2->Reset(sample_rate, block_size);
        candidate.reset();
        for (int warmup = 0; warmup < 3; ++warmup)
        {
          a2->process(a2_input_ptrs, a2_output_ptrs, block_size);
          candidate.process(candidate_input.data(), candidate_output.data(), block_size);
        }
        std::vector<double> a2_timings;
        std::vector<double> candidate_timings;
        a2_timings.reserve(static_cast<std::size_t>(repetitions) * blocks_per_repetition);
        candidate_timings.reserve(static_cast<std::size_t>(repetitions) * blocks_per_repetition);
        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
          a2->Reset(sample_rate, block_size);
          candidate.reset();
          for (int block = 0; block < blocks_per_repetition; ++block)
          {
            const auto measure_a2 = [&] {
              const double elapsed = timed(
                [&] { a2->process(a2_input_ptrs, a2_output_ptrs, block_size); });
              a2_timings.push_back(elapsed);
            };
            const auto measure_candidate = [&] {
              const double elapsed = timed([&] {
                candidate.process(
                  candidate_input.data(), candidate_output.data(), block_size);
              });
              candidate_timings.push_back(elapsed);
            };
            if ((repetition + block) % 2 == 0)
            {
              measure_a2();
              measure_candidate();
            }
            else
            {
              measure_candidate();
              measure_a2();
            }
          }
          checksum += a2_output.back() + candidate_output.back();
        }
        results.push_back({
          {"family", candidate.family()},
          {"profile", candidate.profile()},
          {"parameters", candidate.parameters()},
          {"weight_bytes", candidate.weight_bytes()},
          {"persistent_state_bytes", candidate.persistent_state_bytes()},
          {"scratch_bytes", candidate.scratch_bytes()},
          {"latency_samples", candidate.latency_samples()},
          {"estimated_macs_per_sample", candidate.estimated_macs_per_sample()},
          {"a2", timing_report(summarize(std::move(a2_timings)))},
          {"candidate", timing_report(summarize(std::move(candidate_timings)))},
        });
      }
    }
    json report = {
      {"format", "fssr-amp-arch-native-skeleton-benchmark-v1"},
      {"campaign_version", "AMP-QUALITY-ARCH-v1"},
      {"blind", true},
      {"selection_uses_audio_or_esr", false},
      {"input_source", "deterministic_synthetic_probe"},
      {"trained_weights", false},
      {"same_binary", true},
      {"abi", "float32"},
      {"compiler_optimization", "-Ofast"},
      {"lto_ipo", true},
      {"native_isa", true},
      {"schedule", "interleaved_ab"},
      {"repetitions", repetitions},
      {"scientific_eligible", repetitions == 30},
      {"block_size", block_size},
      {"sample_rate_hz", sample_rate},
      {"checksum", checksum},
      {"results", results},
    };
    std::cout << std::setprecision(12) << report.dump() << '\n';
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
