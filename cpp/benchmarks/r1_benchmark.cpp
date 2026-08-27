#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "fssr_r1_native.hpp"

namespace
{
constexpr std::size_t block_size = 64;
constexpr int confirmation_sample_rate = 48000;
constexpr double pi = 3.14159265358979323846;

double percentile(const std::vector<double>& sorted, double fraction)
{
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

struct Measurement
{
  std::size_t timed_blocks = 0;
  double mean_block_ns = 0.0;
  double median_block_ns = 0.0;
  double p95_block_ns = 0.0;
  float checksum = 0.0f;
};

template <typename Reset, typename Process>
Measurement measure(
  int sample_rate,
  int seconds,
  int iterations,
  Reset&& reset,
  Process&& process)
{
  if (seconds < 1 || iterations < 1)
    throw std::runtime_error("seconds and iterations must be positive");
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
  reset();
  for (int warmup = 0; warmup < 3; ++warmup)
    for (std::size_t block = 0; block < blocks_per_iteration; ++block)
      process(input.data(), output.data(), block_size);
  std::vector<double> nanoseconds;
  nanoseconds.reserve(static_cast<std::size_t>(iterations) * blocks_per_iteration);
  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    for (std::size_t block = 0; block < blocks_per_iteration; ++block)
    {
      const auto start = std::chrono::steady_clock::now();
      process(input.data(), output.data(), block_size);
      const auto stop = std::chrono::steady_clock::now();
      nanoseconds.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
    }
  }
  std::sort(nanoseconds.begin(), nanoseconds.end());
  Measurement result;
  result.timed_blocks = nanoseconds.size();
  result.mean_block_ns =
    std::accumulate(nanoseconds.begin(), nanoseconds.end(), 0.0) / nanoseconds.size();
  result.median_block_ns = percentile(nanoseconds, 0.5);
  result.p95_block_ns = percentile(nanoseconds, 0.95);
  result.checksum = std::accumulate(output.begin(), output.end(), 0.0f);
  return result;
}

void print_measurement(const Measurement& result, int sample_rate)
{
  const double audio_nanoseconds = 1.0e9 * block_size / sample_rate;
  std::cout << ",\"block_size\":" << block_size << ",\"sample_rate_hz\":" << sample_rate
            << ",\"timed_blocks\":" << result.timed_blocks
            << ",\"mean_block_ns\":" << result.mean_block_ns
            << ",\"median_block_ns\":" << result.median_block_ns
            << ",\"p95_block_ns\":" << result.p95_block_ns
            << ",\"median_ns_per_sample\":" << result.median_block_ns / block_size
            << ",\"p95_ns_per_sample\":" << result.p95_block_ns / block_size
            << ",\"median_realtime_factor\":" << audio_nanoseconds / result.median_block_ns
            << ",\"p95_realtime_factor\":" << audio_nanoseconds / result.p95_block_ns
            << ",\"checksum\":" << result.checksum;
}

class DeterministicLstm
{
public:
  explicit DeterministicLstm(std::size_t width)
    : width_(width),
      weight_ih_(4 * width),
      weight_hh_(4 * width * width),
      bias_ih_(4 * width),
      bias_hh_(4 * width),
      head_(width),
      hidden_(width, 0.0f),
      cell_(width, 0.0f),
      gates_(4 * width, 0.0f)
  {
    for (std::size_t index = 0; index < weight_ih_.size(); ++index)
      weight_ih_[index] = deterministic_value(index, 11);
    for (std::size_t index = 0; index < weight_hh_.size(); ++index)
      weight_hh_[index] = deterministic_value(index, 23);
    for (std::size_t index = 0; index < bias_ih_.size(); ++index)
      bias_ih_[index] = deterministic_value(index, 37);
    for (std::size_t index = 0; index < bias_hh_.size(); ++index)
      bias_hh_[index] = deterministic_value(index, 43);
    for (std::size_t index = 0; index < head_.size(); ++index)
      head_[index] = deterministic_value(index, 53);
    head_bias_ = deterministic_value(0, 67);
  }

  void reset() noexcept
  {
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    std::fill(cell_.begin(), cell_.end(), 0.0f);
  }

  void process(const float* input, float* output, std::size_t sample_count) noexcept
  {
    for (std::size_t sample = 0; sample < sample_count; ++sample)
    {
      for (std::size_t gate = 0; gate < 4 * width_; ++gate)
      {
        float value = bias_ih_[gate] + weight_ih_[gate] * input[sample]
                      + bias_hh_[gate];
        const auto row = gate * width_;
        for (std::size_t unit = 0; unit < width_; ++unit)
          value += weight_hh_[row + unit] * hidden_[unit];
        gates_[gate] = value;
      }
      for (std::size_t unit = 0; unit < width_; ++unit)
      {
        const float input_gate = sigmoid(gates_[unit]);
        const float forget_gate = sigmoid(gates_[width_ + unit]);
        const float candidate = std::tanh(gates_[2 * width_ + unit]);
        const float output_gate = sigmoid(gates_[3 * width_ + unit]);
        cell_[unit] = forget_gate * cell_[unit] + input_gate * candidate;
        hidden_[unit] = output_gate * std::tanh(cell_[unit]);
      }
      float value = input[sample] + head_bias_;
      for (std::size_t unit = 0; unit < width_; ++unit)
        value += head_[unit] * hidden_[unit];
      output[sample] = value;
    }
  }

  [[nodiscard]] std::size_t estimated_macs_per_sample() const noexcept
  {
    return 4 * (width_ + width_ * width_) + width_;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return 2 * width_ * sizeof(float);
  }

private:
  float deterministic_value(std::size_t index, std::uint32_t salt) const noexcept
  {
    std::uint32_t value = static_cast<std::uint32_t>(index + 1)
                          ^ static_cast<std::uint32_t>(width_ * 2654435761U) ^ salt;
    value ^= value >> 16U;
    value *= 2246822519U;
    value ^= value >> 13U;
    const int centered = static_cast<int>(value % 2001U) - 1000;
    return 1.0e-5f * static_cast<float>(centered);
  }

  static float sigmoid(float value) noexcept
  {
    return 1.0f / (1.0f + std::exp(-value));
  }

  std::size_t width_;
  std::vector<float> weight_ih_;
  std::vector<float> weight_hh_;
  std::vector<float> bias_ih_;
  std::vector<float> bias_hh_;
  std::vector<float> head_;
  float head_bias_ = 0.0f;
  std::vector<float> hidden_;
  std::vector<float> cell_;
  std::vector<float> gates_;
};

bool registered_width(int width)
{
  constexpr int widths[] = {16, 32, 48, 64, 96};
  return std::find(std::begin(widths), std::end(widths), width) != std::end(widths);
}

void print_lstm_result(int width, int seconds, int iterations)
{
  if (!registered_width(width))
    throw std::runtime_error("LSTM width must be one of 16,32,48,64,96");
  DeterministicLstm model(static_cast<std::size_t>(width));
  const auto result = measure(
    confirmation_sample_rate,
    seconds,
    iterations,
    [&model] { model.reset(); },
    [&model](const float* input, float* output, std::size_t count) {
      model.process(input, output, count);
    });
  std::cout << "{\"format\":\"fssr-r1-lstm-cost-benchmark-v1\""
            << ",\"blind\":true,\"selection_uses_audio_or_esr\":false"
            << ",\"input_source\":\"deterministic-synthetic-probe\""
            << ",\"trained_weights\":false,\"family\":\"wright-lstm-skip\""
            << ",\"width\":" << width << ",\"iterations\":" << iterations
            << ",\"estimated_macs_per_sample\":" << model.estimated_macs_per_sample()
            << ",\"latency_samples\":0,\"state_size_bytes\":" << model.state_size_bytes();
  print_measurement(result, confirmation_sample_rate);
  std::cout << '}';
}
} // namespace

int main(int argc, char** argv)
{
  try
  {
    std::cout << std::setprecision(12);
    if (argc == 5 && std::string(argv[1]) == "--lstm-width")
    {
      print_lstm_result(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
      std::cout << '\n';
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--lstm-sweep")
    {
      const int seconds = std::stoi(argv[2]);
      const int iterations = std::stoi(argv[3]);
      std::cout << "{\"format\":\"fssr-r1-lstm-width-sweep-v1\",\"blind\":true"
                << ",\"selection_uses_audio_or_esr\":false,\"widths\":[16,32,48,64,96]"
                << ",\"results\":[";
      constexpr int widths[] = {16, 32, 48, 64, 96};
      for (std::size_t index = 0; index < std::size(widths); ++index)
      {
        if (index != 0)
          std::cout << ',';
        print_lstm_result(widths[index], seconds, iterations);
      }
      std::cout << "]}\n";
      return 0;
    }
    if (argc != 4)
    {
      std::cerr << "Usage: r1_benchmark MODEL SECONDS ITERATIONS\n"
                << "       r1_benchmark --lstm-width WIDTH SECONDS ITERATIONS\n"
                << "       r1_benchmark --lstm-sweep SECONDS ITERATIONS\n";
      return 2;
    }
    const int seconds = std::stoi(argv[2]);
    const int iterations = std::stoi(argv[3]);
    fssr::r1::Model model(argv[1]);
    const auto sample_rate = model.sample_rate_hz();
    const auto result = measure(
      sample_rate,
      seconds,
      iterations,
      [&model] { model.reset(); },
      [&model](const float* input, float* output, std::size_t count) {
        model.process(input, output, count);
      });
    std::cout << "{\"format\":\"fssr-r1-benchmark-v1\",\"iterations\":" << iterations
              << ",\"latency_samples\":" << model.latency_samples()
              << ",\"state_size_bytes\":" << model.state_size_bytes();
    print_measurement(result, sample_rate);
    std::cout << "}\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
