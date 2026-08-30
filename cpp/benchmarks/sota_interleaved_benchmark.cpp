#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "sota_prototype.hpp"

namespace
{
constexpr std::size_t block_size = 128;
constexpr std::size_t sample_rate = 48000;
std::atomic<std::size_t> allocation_count{0};

struct Summary
{
  double median_block_ns;
  double p95_block_ns;
  std::size_t timed_blocks;
};

double percentile(const std::vector<double>& sorted, double fraction)
{
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Summary summarize(std::vector<double>& values)
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

std::string cpu_governor()
{
  std::ifstream stream(
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
  std::string governor;
  if (stream && std::getline(stream, governor) && !governor.empty())
    return governor;
  return "unavailable";
}

void print_summary(const Summary& summary)
{
  const double audio_duration_ns =
    1.0e9 * static_cast<double>(block_size) / static_cast<double>(sample_rate);
  std::cout << "{\"timed_blocks\":" << summary.timed_blocks
            << ",\"median_block_ns\":" << summary.median_block_ns
            << ",\"p95_block_ns\":" << summary.p95_block_ns
            << ",\"median_ns_per_sample\":"
            << summary.median_block_ns / static_cast<double>(block_size)
            << ",\"p95_ns_per_sample\":"
            << summary.p95_block_ns / static_cast<double>(block_size)
            << ",\"median_rtf\":" << summary.median_block_ns / audio_duration_ns
            << ",\"p95_rtf\":" << summary.p95_block_ns / audio_duration_ns << "}";
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

int main(int argc, char** argv)
{
  if (argc > 3)
  {
    std::cerr << "Usage: sota_interleaved_benchmark [SECONDS [REPETITIONS]]\n";
    return 2;
  }
  try
  {
    const int seconds = argc >= 2 ? std::stoi(argv[1]) : 1;
    const int repetitions = argc >= 3 ? std::stoi(argv[2]) : 30;
    if (seconds < 1 || seconds > 3600 || repetitions < 1 || repetitions > 1000)
      throw std::runtime_error("seconds or repetitions are outside supported bounds");
    const auto blocks_per_repetition =
      static_cast<std::size_t>(seconds) * sample_rate / block_size;
    const auto timing_count =
      blocks_per_repetition * static_cast<std::size_t>(repetitions);

    fssr::sota::Prototype control(fssr::sota::Variant::control_delay32);
    fssr::sota::Prototype candidate(fssr::sota::Variant::candidate_delay24);
    std::array<float, block_size> input{};
    std::array<float, block_size> control_output{};
    std::array<float, block_size> candidate_output{};
    for (std::size_t index = 0; index < block_size; ++index)
    {
      const double time = static_cast<double>(index) / static_cast<double>(sample_rate);
      input[index] = static_cast<float>(
        0.25 * std::sin(2.0 * 3.14159265358979323846 * 220.0 * time)
        + 0.1 * std::sin(2.0 * 3.14159265358979323846 * 1230.0 * time));
    }

    std::vector<double> control_timings;
    std::vector<double> candidate_timings;
    control_timings.reserve(timing_count);
    candidate_timings.reserve(timing_count);
    for (int warmup = 0; warmup < 8; ++warmup)
    {
      control.process(input.data(), control_output.data(), block_size);
      candidate.process(input.data(), candidate_output.data(), block_size);
    }
    control.reset();
    candidate.reset();

    double checksum = 0.0;
    const auto allocations_before = allocation_count.load(std::memory_order_relaxed);
    for (int repetition = 0; repetition < repetitions; ++repetition)
    {
      control.reset();
      candidate.reset();
      for (std::size_t block = 0; block < blocks_per_repetition; ++block)
      {
        const auto measure_control = [&] {
          control_timings.push_back(timed([&] {
            control.process(input.data(), control_output.data(), block_size);
          }));
          checksum += control_output[
            (static_cast<std::size_t>(repetition) + block) % block_size];
        };
        const auto measure_candidate = [&] {
          candidate_timings.push_back(timed([&] {
            candidate.process(input.data(), candidate_output.data(), block_size);
          }));
          checksum += candidate_output[
            (static_cast<std::size_t>(repetition) + block) % block_size];
        };
        if ((static_cast<std::size_t>(repetition) + block) % 2 == 0)
        {
          measure_control();
          measure_candidate();
        }
        else
        {
          measure_candidate();
          measure_control();
        }
      }
    }
    const auto allocations_after = allocation_count.load(std::memory_order_relaxed);
    const auto audio_loop_allocations = allocations_after - allocations_before;
    const auto control_summary = summarize(control_timings);
    const auto candidate_summary = summarize(candidate_timings);
    const auto governor = cpu_governor();
    const bool benchmark_contract_eligible =
      repetitions == 30 && seconds >= 1 && audio_loop_allocations == 0;

    std::cout << std::setprecision(12)
              << "{\"format\":\"fssr-sota-interleaved-benchmark-v1\""
              << ",\"implementation\":\"delayed_identity_scaffold\""
              << ",\"learned_weights_supported\":false"
              << ",\"abi\":\"float32\""
              << ",\"same_binary\":true"
              << ",\"same_isa\":true"
              << ",\"threads\":1"
              << ",\"sample_rate_hz\":" << sample_rate
              << ",\"block_size\":" << block_size
              << ",\"seconds_per_repetition\":" << seconds
              << ",\"repetitions\":" << repetitions
              << ",\"schedule\":\"interleaved_ab\""
              << ",\"rtf_definition\":\"compute_time_over_audio_duration\""
              << ",\"cpu_governor\":\"" << governor << "\""
              << ",\"audio_loop_allocations\":" << audio_loop_allocations
              << ",\"benchmark_contract_eligible\":"
              << (benchmark_contract_eligible ? "true" : "false")
              << ",\"scientific_eligible\":false"
              << ",\"scientific_ineligibility_reason\":"
                 "\"delayed_identity_scaffold_has_no_learned_weights\""
              << ",\"checksum\":" << checksum
              << ",\"models\":{\"control\":{\"identifier\":\""
              << control.identifier() << "\",\"latency_samples\":"
              << control.latency_samples() << ",\"timing\":";
    print_summary(control_summary);
    std::cout << "},\"candidate\":{\"identifier\":\"" << candidate.identifier()
              << "\",\"latency_samples\":" << candidate.latency_samples()
              << ",\"timing\":";
    print_summary(candidate_summary);
    std::cout << "}}}\n";
    return audio_loop_allocations == 0 ? 0 : 1;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
