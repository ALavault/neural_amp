#include "amp_arch_skeleton.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fssr::arch
{
namespace
{
constexpr std::size_t kLatency = 32;
constexpr std::size_t kControlPeriod = 64;
constexpr std::size_t kModulationDelay = 16;
constexpr std::array<std::size_t, 5> kDilations = {2, 8, 32, 128, 512};

enum class Family
{
  selective_s6,
  micro_tcn,
  physics_s6_tcn,
  physics_deterministic_tcn,
  latent_s6_tcn,
  cascade_s6_tcn,
};

struct Profile
{
  std::size_t channels;
  std::size_t observer_state;
  std::size_t s6_state;
};

Family parse_family(const std::string& name)
{
  if (name == "selective_s6_x2")
    return Family::selective_s6;
  if (name == "micro_tcn_x2")
    return Family::micro_tcn;
  if (name == "phys_s6_tcn_x2")
    return Family::physics_s6_tcn;
  if (name == "phys_det_tcn_x2")
    return Family::physics_deterministic_tcn;
  if (name == "rf2047_tfilm_x2")
    return Family::latent_s6_tcn;
  if (name == "cascade_rf2047_tfilm_x2")
    return Family::cascade_s6_tcn;
  throw std::runtime_error("unknown AMP-QUALITY-ARCH-v1 family");
}

Profile parse_profile(const std::string& name)
{
  if (name == "slim")
    return {12, 32, 48};
  if (name == "balanced")
    return {16, 48, 64};
  if (name == "full")
    return {24, 64, 96};
  if (name == "max")
    return {32, 96, 128};
  throw std::runtime_error("unknown AMP-QUALITY-ARCH-v1 profile");
}

float deterministic_weight(std::size_t index, std::uint32_t salt) noexcept
{
  std::uint32_t value = static_cast<std::uint32_t>(index + 1) ^ salt;
  value ^= value >> 16U;
  value *= 2246822519U;
  value ^= value >> 13U;
  const int centered = static_cast<int>(value % 2001U) - 1000;
  return 2.0e-5f * static_cast<float>(centered);
}

void initialize(std::vector<float>& values, std::uint32_t salt) noexcept
{
  for (std::size_t index = 0; index < values.size(); ++index)
    values[index] = deterministic_weight(index, salt);
}

float sigmoid(float value) noexcept
{
  return 1.0f / (1.0f + std::exp(-value));
}

float softplus(float value) noexcept
{
  return std::max(value, 0.0f) + std::log1p(std::exp(-std::abs(value)));
}

class Fir65
{
public:
  explicit Fir65(bool interpolation)
  {
    constexpr double beta = 8.6;
    constexpr double center = 32.0;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double denominator = std::cyl_bessel_i(0.0, beta);
    double total = 0.0;
    for (std::size_t index = 0; index < coefficients_.size(); ++index)
    {
      const double offset = static_cast<double>(index) - center;
      const double sinc = offset == 0.0 ? 1.0 : std::sin(0.5 * pi * offset) / (0.5 * pi * offset);
      const double ratio = offset / center;
      const double window = std::cyl_bessel_i(0.0, beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio)))
                            / denominator;
      coefficients_[index] = static_cast<float>(0.5 * sinc * window);
      total += coefficients_[index];
    }
    const double scale = (interpolation ? 2.0 : 1.0) / total;
    for (auto& coefficient : coefficients_)
      coefficient = static_cast<float>(coefficient * scale);
  }

  void reset() noexcept
  {
    history_.fill(0.0f);
    position_ = 0;
  }

  float process(float input) noexcept
  {
    float output = coefficients_.back() * input;
    for (std::size_t coefficient = 0; coefficient + 1 < coefficients_.size(); ++coefficient)
    {
      const auto lag = coefficients_.size() - 1 - coefficient;
      const auto index = (position_ + history_.size() - lag) % history_.size();
      output += coefficients_[coefficient] * history_[index];
    }
    history_[position_] = input;
    position_ = (position_ + 1) % history_.size();
    return output;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(position_);
  }

private:
  std::array<float, 65> coefficients_{};
  std::array<float, 64> history_{};
  std::size_t position_ = 0;
};

template <std::size_t Samples>
class Delay
{
public:
  void reset() noexcept
  {
    history_.fill(0.0f);
    position_ = 0;
  }

  float process(float input) noexcept
  {
    const float output = history_[position_];
    history_[position_] = input;
    position_ = (position_ + 1) % Samples;
    return output;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(position_);
  }

private:
  std::array<float, Samples> history_{};
  std::size_t position_ = 0;
};

class FeatureBus
{
public:
  void reset() noexcept
  {
    sums_.fill(0.0f);
    current_.fill(0.0f);
    slow_energy_ = 0.0f;
    peak_ = 0.0f;
    count_ = 0;
  }

  [[nodiscard]] const std::array<float, 6>& current() const noexcept
  {
    return current_;
  }

  void consume(float input) noexcept
  {
    const float absolute = std::abs(input);
    sums_[0] += absolute;
    sums_[1] += input * input;
    sums_[2] += input;
    peak_ = std::max(peak_, absolute);
    ++count_;
    if (count_ != kControlPeriod)
      return;
    const float mean_abs = sums_[0] / static_cast<float>(kControlPeriod);
    const float energy = sums_[1] / static_cast<float>(kControlPeriod);
    const float mean = sums_[2] / static_cast<float>(kControlPeriod);
    slow_energy_ = 0.95f * slow_energy_ + 0.05f * energy;
    const float low_energy = mean * mean;
    current_ = {
      mean_abs,
      std::sqrt(slow_energy_ + 1.0e-8f),
      mean,
      low_energy,
      std::max(energy - low_energy, 0.0f),
      peak_ / std::sqrt(energy + 1.0e-8f)};
    sums_.fill(0.0f);
    peak_ = 0.0f;
    count_ = 0;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    return (sums_.size() + current_.size() + 2) * sizeof(float) + sizeof(count_);
  }

private:
  std::array<float, 3> sums_{};
  std::array<float, 6> current_{};
  float slow_energy_ = 0.0f;
  float peak_ = 0.0f;
  std::size_t count_ = 0;
};

class Control
{
public:
  Control(Family family, std::size_t channels, std::size_t state)
    : family_(family),
      channels_(channels),
      state_(state)
  {
    if (family_ != Family::micro_tcn)
    {
      modulation_.resize(2 * channels_, 0.0f);
      delayed_.resize(2 * channels_, 0.0f);
      modulation_delay_.resize(kModulationDelay * 2 * channels_, 0.0f);
    }
    if (family_ == Family::physics_deterministic_tcn)
    {
      deterministic_weights_.resize(2 * channels_ * 6);
      deterministic_bias_.resize(2 * channels_);
      initialize(deterministic_weights_, 101U);
      initialize(deterministic_bias_, 103U);
    }
    else if (uses_observer())
    {
      hidden_.resize(state_, 0.0f);
      work_.resize(state_, 0.0f);
      input_weights_.resize(state_ * 6);
      input_bias_.resize(state_);
      delta_weights_.resize(state_ * 6);
      delta_bias_.resize(state_);
      read_weights_.resize(state_ * 6);
      read_bias_.resize(state_);
      log_a_.resize(state_);
      modulation_weights_.resize(2 * channels_ * state_);
      modulation_bias_.resize(2 * channels_);
      initialize(input_weights_, 107U);
      initialize(input_bias_, 109U);
      initialize(delta_weights_, 113U);
      initialize(delta_bias_, 127U);
      initialize(read_weights_, 131U);
      initialize(read_bias_, 137U);
      initialize(modulation_weights_, 139U);
      initialize(modulation_bias_, 149U);
      for (std::size_t index = 0; index < state_; ++index)
        log_a_[index] = -5.0f + 4.0f * static_cast<float>(index) / static_cast<float>(std::max<std::size_t>(state_ - 1, 1));
    }
  }

  void reset() noexcept
  {
    bus_.reset();
    std::fill(modulation_.begin(), modulation_.end(), 0.0f);
    std::fill(delayed_.begin(), delayed_.end(), 0.0f);
    std::fill(modulation_delay_.begin(), modulation_delay_.end(), 0.0f);
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    std::fill(work_.begin(), work_.end(), 0.0f);
    delay_position_ = 0;
    count_ = 0;
  }

  void process(float input, float* gamma, float* beta) noexcept
  {
    if (family_ == Family::micro_tcn)
    {
      for (std::size_t channel = 0; channel < channels_; ++channel)
      {
        gamma[channel] = 1.0f;
        beta[channel] = 0.0f;
      }
      return;
    }
    if (count_ == 0)
      update(bus_.current());
    const std::size_t offset = delay_position_ * 2 * channels_;
    for (std::size_t index = 0; index < 2 * channels_; ++index)
    {
      delayed_[index] = modulation_delay_[offset + index];
      modulation_delay_[offset + index] = modulation_[index];
    }
    delay_position_ = (delay_position_ + 1) % kModulationDelay;
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      gamma[channel] = 1.0f + 0.25f * std::tanh(delayed_[channel]);
      beta[channel] = 0.10f * std::tanh(delayed_[channels_ + channel]);
    }
    bus_.consume(input);
    count_ = (count_ + 1) % kControlPeriod;
  }

  [[nodiscard]] std::size_t parameter_count() const noexcept
  {
    if (family_ == Family::micro_tcn)
      return 0;
    if (family_ == Family::physics_deterministic_tcn)
      return 14 * channels_;
    if (!uses_observer())
      return 0;
    return 28 * state_ + 2 * channels_ * state_ + 2 * channels_ + 6;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    if (family_ == Family::micro_tcn)
      return 0;
    return bus_.state_bytes() + (modulation_.size() + modulation_delay_.size() + hidden_.size()) * sizeof(float)
           + 3 * sizeof(std::size_t);
  }

  [[nodiscard]] std::size_t scratch_bytes() const noexcept
  {
    return (delayed_.size() + work_.size()) * sizeof(float);
  }

private:
  [[nodiscard]] bool uses_observer() const noexcept
  {
    return family_ == Family::physics_s6_tcn || family_ == Family::latent_s6_tcn
           || family_ == Family::cascade_s6_tcn;
  }

  static float affine_row(
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::size_t row,
    const std::array<float, 6>& features) noexcept
  {
    float value = bias[row];
    for (std::size_t feature = 0; feature < features.size(); ++feature)
      value += weights[row * features.size() + feature] * features[feature];
    return value;
  }

  void update(const std::array<float, 6>& features) noexcept
  {
    if (family_ == Family::physics_deterministic_tcn)
    {
      for (std::size_t row = 0; row < modulation_.size(); ++row)
        modulation_[row] = affine_row(deterministic_weights_, deterministic_bias_, row, features);
      return;
    }
    if (!uses_observer())
      return;
    for (std::size_t unit = 0; unit < state_; ++unit)
    {
      const float delta = softplus(affine_row(delta_weights_, delta_bias_, unit, features)) + 1.0e-4f;
      const float decay = std::exp(-delta * std::exp(log_a_[unit]));
      const float candidate = std::tanh(affine_row(input_weights_, input_bias_, unit, features));
      hidden_[unit] = decay * hidden_[unit] + (1.0f - decay) * candidate;
      work_[unit] = sigmoid(affine_row(read_weights_, read_bias_, unit, features)) * hidden_[unit];
    }
    for (std::size_t row = 0; row < modulation_.size(); ++row)
    {
      float value = modulation_bias_[row];
      for (std::size_t unit = 0; unit < state_; ++unit)
        value += modulation_weights_[row * state_ + unit] * work_[unit];
      modulation_[row] = value;
    }
  }

  Family family_;
  std::size_t channels_;
  std::size_t state_;
  FeatureBus bus_;
  std::vector<float> modulation_;
  std::vector<float> delayed_;
  std::vector<float> modulation_delay_;
  std::vector<float> hidden_;
  std::vector<float> work_;
  std::vector<float> deterministic_weights_;
  std::vector<float> deterministic_bias_;
  std::vector<float> input_weights_;
  std::vector<float> input_bias_;
  std::vector<float> delta_weights_;
  std::vector<float> delta_bias_;
  std::vector<float> read_weights_;
  std::vector<float> read_bias_;
  std::vector<float> log_a_;
  std::vector<float> modulation_weights_;
  std::vector<float> modulation_bias_;
  std::size_t delay_position_ = 0;
  std::size_t count_ = 0;
};

class TcnBlock
{
public:
  TcnBlock(std::size_t channels, std::size_t dilation, std::uint32_t salt)
    : channels_(channels),
      dilation_(dilation),
      history_length_(6 * dilation),
      depth_weights_(7 * channels),
      depth_bias_(channels),
      point_weights_(channels * channels),
      point_bias_(channels),
      history_(channels * history_length_, 0.0f),
      depth_(channels, 0.0f),
      mixed_(channels, 0.0f)
  {
    initialize(depth_weights_, salt);
    initialize(depth_bias_, salt + 1U);
    initialize(point_weights_, salt + 2U);
    initialize(point_bias_, salt + 3U);
  }

  void reset() noexcept
  {
    std::fill(history_.begin(), history_.end(), 0.0f);
    std::fill(depth_.begin(), depth_.end(), 0.0f);
    std::fill(mixed_.begin(), mixed_.end(), 0.0f);
    position_ = 0;
  }

  void process(float* hidden, const float* gamma, const float* beta) noexcept
  {
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      float value = depth_bias_[channel] + depth_weights_[channel * 7 + 6] * hidden[channel];
      const std::size_t history_offset = channel * history_length_;
      for (std::size_t tap = 0; tap < 6; ++tap)
      {
        const std::size_t lag = (6 - tap) * dilation_;
        const std::size_t index = (position_ + history_length_ - lag) % history_length_;
        value += depth_weights_[channel * 7 + tap] * history_[history_offset + index];
      }
      depth_[channel] = value;
    }
    for (std::size_t output = 0; output < channels_; ++output)
    {
      float value = point_bias_[output];
      for (std::size_t input = 0; input < channels_; ++input)
        value += point_weights_[output * channels_ + input] * depth_[input];
      mixed_[output] = value;
    }
    for (std::size_t channel = 0; channel < channels_; ++channel)
    {
      history_[channel * history_length_ + position_] = hidden[channel];
      hidden[channel] += std::tanh(gamma[channel] * mixed_[channel] + beta[channel]);
    }
    position_ = (position_ + 1) % history_length_;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(position_);
  }

  [[nodiscard]] std::size_t scratch_bytes() const noexcept
  {
    return (depth_.size() + mixed_.size()) * sizeof(float);
  }

private:
  std::size_t channels_;
  std::size_t dilation_;
  std::size_t history_length_;
  std::vector<float> depth_weights_;
  std::vector<float> depth_bias_;
  std::vector<float> point_weights_;
  std::vector<float> point_bias_;
  std::vector<float> history_;
  std::vector<float> depth_;
  std::vector<float> mixed_;
  std::size_t position_ = 0;
};

class TcnBranch
{
public:
  TcnBranch(std::size_t channels, std::uint32_t salt)
    : channels_(channels),
      input_weights_(channels),
      input_bias_(channels),
      output_weights_(channels),
      hidden_(channels, 0.0f)
  {
    initialize(input_weights_, salt);
    initialize(input_bias_, salt + 1U);
    initialize(output_weights_, salt + 2U);
    output_bias_ = deterministic_weight(0, salt + 3U);
    for (std::size_t index = 0; index < kDilations.size(); ++index)
      blocks_.emplace_back(channels_, kDilations[index], salt + 11U + static_cast<std::uint32_t>(7 * index));
  }

  void reset() noexcept
  {
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    for (auto& block : blocks_)
      block.reset();
  }

  float process(float input, const float* gamma, const float* beta) noexcept
  {
    for (std::size_t channel = 0; channel < channels_; ++channel)
      hidden_[channel] = input_bias_[channel] + input_weights_[channel] * input;
    for (auto& block : blocks_)
      block.process(hidden_.data(), gamma, beta);
    float residual = output_bias_;
    for (std::size_t channel = 0; channel < channels_; ++channel)
      residual += output_weights_[channel] * hidden_[channel];
    return input + 0.05f * std::tanh(residual);
  }

  [[nodiscard]] std::size_t parameter_count() const noexcept
  {
    return 5 * channels_ * channels_ + 48 * channels_ + 2;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    std::size_t result = 0;
    for (const auto& block : blocks_)
      result += block.state_bytes();
    return result;
  }

  [[nodiscard]] std::size_t scratch_bytes() const noexcept
  {
    std::size_t result = hidden_.size() * sizeof(float);
    for (const auto& block : blocks_)
      result += block.scratch_bytes();
    return result;
  }

  [[nodiscard]] std::size_t macs_per_internal_sample() const noexcept
  {
    return 2 * channels_ + 5 * (7 * channels_ + channels_ * channels_);
  }

private:
  std::size_t channels_;
  std::vector<float> input_weights_;
  std::vector<float> input_bias_;
  std::vector<TcnBlock> blocks_;
  std::vector<float> output_weights_;
  float output_bias_ = 0.0f;
  std::vector<float> hidden_;
};

class SelectiveS6
{
public:
  explicit SelectiveS6(std::size_t state)
    : state_(state),
      input_weights_(state),
      input_bias_(state),
      delta_weights_(state),
      delta_bias_(state),
      input_gate_weights_(state),
      input_gate_bias_(state),
      read_gate_weights_(state),
      read_gate_bias_(state),
      log_a_(state),
      output_weights_(state),
      hidden_(state, 0.0f)
  {
    initialize(input_weights_, 211U);
    initialize(input_bias_, 223U);
    initialize(delta_weights_, 227U);
    initialize(delta_bias_, 229U);
    initialize(input_gate_weights_, 233U);
    initialize(input_gate_bias_, 239U);
    initialize(read_gate_weights_, 241U);
    initialize(read_gate_bias_, 251U);
    initialize(output_weights_, 257U);
    output_bias_ = deterministic_weight(0, 263U);
    for (std::size_t index = 0; index < state_; ++index)
      log_a_[index] = -5.0f + 4.0f * static_cast<float>(index) / static_cast<float>(std::max<std::size_t>(state_ - 1, 1));
  }

  void reset() noexcept
  {
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
  }

  float process(float input) noexcept
  {
    float residual = output_bias_;
    for (std::size_t unit = 0; unit < state_; ++unit)
    {
      const float delta = softplus(delta_bias_[unit] + delta_weights_[unit] * input) + 1.0e-4f;
      const float decay = std::exp(-delta * std::exp(log_a_[unit]));
      const float driven = std::tanh(input_bias_[unit] + input_weights_[unit] * input)
                           * sigmoid(input_gate_bias_[unit] + input_gate_weights_[unit] * input);
      hidden_[unit] = decay * hidden_[unit] + (1.0f - decay) * driven;
      residual += output_weights_[unit]
                  * sigmoid(read_gate_bias_[unit] + read_gate_weights_[unit] * input) * hidden_[unit];
    }
    return input + 0.05f * std::tanh(residual);
  }

  [[nodiscard]] std::size_t parameter_count() const noexcept
  {
    return 10 * state_ + 2;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    return hidden_.size() * sizeof(float);
  }

  [[nodiscard]] std::size_t macs_per_internal_sample() const noexcept
  {
    return 5 * state_;
  }

private:
  std::size_t state_;
  std::vector<float> input_weights_;
  std::vector<float> input_bias_;
  std::vector<float> delta_weights_;
  std::vector<float> delta_bias_;
  std::vector<float> input_gate_weights_;
  std::vector<float> input_gate_bias_;
  std::vector<float> read_gate_weights_;
  std::vector<float> read_gate_bias_;
  std::vector<float> log_a_;
  std::vector<float> output_weights_;
  float output_bias_ = 0.0f;
  std::vector<float> hidden_;
};
} // namespace

struct Skeleton::Impl
{
  Impl(std::string family_name, std::string profile_name)
    : family_text(std::move(family_name)),
      profile_text(std::move(profile_name)),
      family(parse_family(family_text)),
      profile(parse_profile(profile_text)),
      control(family, profile.channels, profile.observer_state),
      gamma(profile.channels, 1.0f),
      beta(profile.channels, 0.0f)
  {
    if (family == Family::selective_s6)
      s6 = std::make_unique<SelectiveS6>(profile.s6_state);
    else
    {
      first = std::make_unique<TcnBranch>(profile.channels, 307U);
      if (family == Family::cascade_s6_tcn)
        second = std::make_unique<TcnBranch>(profile.channels, 401U);
    }
  }

  void reset() noexcept
  {
    upsample.reset();
    downsample.reset();
    linear_delay.reset();
    control.reset();
    if (first)
      first->reset();
    if (second)
      second->reset();
    if (s6)
      s6->reset();
  }

  float branch(float input) noexcept
  {
    if (s6)
      return s6->process(input);
    float output = first->process(input, gamma.data(), beta.data());
    if (second)
      output = second->process(output, gamma.data(), beta.data());
    return output;
  }

  void process(const float* input, float* output, std::size_t sample_count) noexcept
  {
    for (std::size_t sample = 0; sample < sample_count; ++sample)
    {
      if (family != Family::selective_s6)
        control.process(input[sample], gamma.data(), beta.data());
      const float high_even = upsample.process(2.0f * input[sample]);
      const float residual_even = branch(high_even) - high_even;
      const float down_even = downsample.process(residual_even);
      output[sample] = linear_delay.process(input[sample]) + down_even;
      const float high_odd = upsample.process(0.0f);
      const float residual_odd = branch(high_odd) - high_odd;
      static_cast<void>(downsample.process(residual_odd));
    }
  }

  [[nodiscard]] std::size_t parameters() const noexcept
  {
    if (s6)
      return s6->parameter_count();
    std::size_t result = control.parameter_count() + first->parameter_count();
    if (second)
      result += second->parameter_count();
    return result;
  }

  [[nodiscard]] std::size_t state_bytes() const noexcept
  {
    std::size_t result = upsample.state_bytes() + downsample.state_bytes()
                         + linear_delay.state_bytes() + control.state_bytes();
    if (first)
      result += first->state_bytes();
    if (second)
      result += second->state_bytes();
    if (s6)
      result += s6->state_bytes();
    return result;
  }

  [[nodiscard]] std::size_t scratch_bytes() const noexcept
  {
    std::size_t result = (gamma.size() + beta.size()) * sizeof(float) + control.scratch_bytes();
    if (first)
      result += first->scratch_bytes();
    if (second)
      result += second->scratch_bytes();
    return result;
  }

  [[nodiscard]] std::size_t macs_per_sample() const noexcept
  {
    constexpr std::size_t resampling_macs = 4 * 65;
    if (s6)
      return resampling_macs + 2 * s6->macs_per_internal_sample();
    std::size_t result = resampling_macs + 2 * first->macs_per_internal_sample();
    if (second)
      result += 2 * second->macs_per_internal_sample();
    if (family == Family::physics_deterministic_tcn)
      result += (14 * profile.channels) / kControlPeriod;
    else if (family != Family::micro_tcn)
      result += (18 * profile.observer_state + 2 * profile.channels * profile.observer_state)
                / kControlPeriod;
    return result;
  }

  std::string family_text;
  std::string profile_text;
  Family family;
  Profile profile;
  Fir65 upsample{true};
  Fir65 downsample{false};
  Delay<kLatency> linear_delay;
  Control control;
  std::unique_ptr<TcnBranch> first;
  std::unique_ptr<TcnBranch> second;
  std::unique_ptr<SelectiveS6> s6;
  std::vector<float> gamma;
  std::vector<float> beta;
};

Skeleton::Skeleton(const std::string& family, const std::string& profile)
  : impl_(std::make_unique<Impl>(family, profile))
{
}

Skeleton::~Skeleton() = default;
Skeleton::Skeleton(Skeleton&&) noexcept = default;
Skeleton& Skeleton::operator=(Skeleton&&) noexcept = default;

void Skeleton::reset() noexcept
{
  impl_->reset();
}

void Skeleton::process(const float* input, float* output, std::size_t sample_count) noexcept
{
  impl_->process(input, output, sample_count);
}

const std::string& Skeleton::family() const noexcept
{
  return impl_->family_text;
}

const std::string& Skeleton::profile() const noexcept
{
  return impl_->profile_text;
}

int Skeleton::latency_samples() const noexcept
{
  return static_cast<int>(kLatency);
}

std::size_t Skeleton::parameters() const noexcept
{
  return impl_->parameters();
}

std::size_t Skeleton::weight_bytes() const noexcept
{
  return parameters() * sizeof(float);
}

std::size_t Skeleton::persistent_state_bytes() const noexcept
{
  return impl_->state_bytes();
}

std::size_t Skeleton::scratch_bytes() const noexcept
{
  return impl_->scratch_bytes();
}

std::size_t Skeleton::estimated_macs_per_sample() const noexcept
{
  return impl_->macs_per_sample();
}
} // namespace fssr::arch
