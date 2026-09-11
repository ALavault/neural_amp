#include "fssr_r1_native.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

namespace fssr::r1
{
namespace
{
using json = nlohmann::json;

float finite_float(const json& value, const std::string& name)
{
  if (!value.is_number())
    throw std::runtime_error(name + " must be numeric");
  const auto converted = value.get<float>();
  if (!std::isfinite(converted))
    throw std::runtime_error(name + " must be finite float32");
  return converted;
}

int positive_int(const json& value, const std::string& name)
{
  if (!value.is_number_integer())
    throw std::runtime_error(name + " must be an integer");
  const auto converted = value.get<int>();
  if (converted <= 0)
    throw std::runtime_error(name + " must be positive");
  return converted;
}

std::vector<float> float_vector(const json& value, const std::string& name)
{
  if (!value.is_array())
    throw std::runtime_error(name + " must be an array");
  std::vector<float> result;
  result.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index)
    result.push_back(finite_float(value[index], name + "[" + std::to_string(index) + "]"));
  return result;
}

std::vector<std::vector<float>> float_matrix(
  const json& value,
  std::size_t rows,
  std::size_t columns,
  const std::string& name)
{
  if (!value.is_array() || value.size() != rows)
    throw std::runtime_error(name + " has an invalid row count");
  std::vector<std::vector<float>> result;
  result.reserve(rows);
  for (std::size_t row = 0; row < rows; ++row)
  {
    auto values = float_vector(value[row], name + "[" + std::to_string(row) + "]");
    if (values.size() != columns)
      throw std::runtime_error(name + " has an invalid column count");
    result.push_back(std::move(values));
  }
  return result;
}

std::vector<std::vector<std::vector<float>>> float_tensor3(
  const json& value,
  std::size_t dimension0,
  std::size_t dimension1,
  std::size_t dimension2,
  const std::string& name)
{
  if (!value.is_array() || value.size() != dimension0)
    throw std::runtime_error(name + " has an invalid outer dimension");
  std::vector<std::vector<std::vector<float>>> result;
  result.reserve(dimension0);
  for (std::size_t index = 0; index < dimension0; ++index)
    result.push_back(float_matrix(
      value[index], dimension1, dimension2, name + "[" + std::to_string(index) + "]"));
  return result;
}

void require_equal(const json& document, const char* key, const json& expected)
{
  if (!document.contains(key) || document.at(key) != expected)
    throw std::runtime_error(std::string(key) + " has an unsupported value");
}

class CausalFir
{
public:
  explicit CausalFir(const json& payload)
    : coefficients_(float_vector(payload.at("coefficients"), "FIR coefficients")),
      history_(coefficients_.empty() ? 0 : coefficients_.size() - 1, 0.0f)
  {
    if (coefficients_.empty())
      throw std::runtime_error("FIR filters cannot be empty");
  }

  void reset() noexcept
  {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_index_ = 0;
  }

  float process(float input) noexcept
  {
    float output = coefficients_.back() * input;
    const auto history_size = history_.size();
    for (std::size_t coefficient = 0; coefficient + 1 < coefficients_.size(); ++coefficient)
    {
      const auto lag = coefficients_.size() - 1 - coefficient;
      const auto index = (write_index_ + history_size - lag) % history_size;
      output += coefficients_[coefficient] * history_[index];
    }
    if (history_size != 0)
    {
      history_[write_index_] = input;
      write_index_ = (write_index_ + 1) % history_size;
    }
    return output;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(write_index_);
  }

private:
  std::vector<float> coefficients_;
  std::vector<float> history_;
  std::size_t write_index_ = 0;
};

class CausalDelay
{
public:
  explicit CausalDelay(std::size_t samples) : history_(samples, 0.0f) {}

  void reset() noexcept
  {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_index_ = 0;
  }

  float process(float input) noexcept
  {
    if (history_.empty())
      return input;
    const float output = history_[write_index_];
    history_[write_index_] = input;
    write_index_ = (write_index_ + 1) % history_.size();
    return output;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(write_index_);
  }

private:
  std::vector<float> history_;
  std::size_t write_index_ = 0;
};

class HermiteSpline
{
public:
  explicit HermiteSpline(const json& payload)
    : knots_(float_vector(payload.at("knots"), "spline knots")),
      values_(float_vector(payload.at("values"), "spline values")),
      slopes_(float_vector(payload.at("slopes"), "spline slopes")),
      drive_(finite_float(payload.at("drive"), "spline drive")),
      offset_(finite_float(payload.at("offset"), "spline offset")),
      adaa1_(payload.value("adaa1", false)),
      threshold_(adaa1_
                   ? finite_float(payload.at("difference_limit_threshold"), "ADAA threshold")
                   : 1.0e-4f)
  {
    if (knots_.size() < 4 || values_.size() != knots_.size() || slopes_.size() != knots_.size())
      throw std::runtime_error("spline arrays have incompatible dimensions");
    for (std::size_t index = 1; index < knots_.size(); ++index)
      if (!(knots_[index - 1] < knots_[index]))
        throw std::runtime_error("spline knots must be strictly increasing");
    spacing_ = (knots_.back() - knots_.front()) / static_cast<float>(knots_.size() - 1);
    for (std::size_t index = 1; index + 1 < knots_.size(); ++index)
    {
      const float expected = knots_.front() + spacing_ * static_cast<float>(index);
      const float tolerance = 32.0f * std::numeric_limits<float>::epsilon()
                              * std::max(1.0f, std::abs(expected));
      if (std::abs(knots_[index] - expected) > tolerance)
        throw std::runtime_error("spline knots must use the registered uniform grid");
    }
    if (adaa1_ && threshold_ != 1.0e-4f)
      throw std::runtime_error("ADAA threshold must equal 1e-4");
    cumulative_.assign(knots_.size(), 0.0f);
    for (std::size_t index = 0; index + 1 < knots_.size(); ++index)
    {
      const float integral = spacing_
                             * (0.5f * values_[index] + 0.5f * values_[index + 1]
                                + spacing_ * (slopes_[index] - slopes_[index + 1]) / 12.0f);
      cumulative_[index + 1] = cumulative_[index] + integral;
    }
  }

  void reset() noexcept { previous_ = 0.0f; }

  float process(
    float input,
    float drive_factor = 1.0f,
    float offset_delta = 0.0f) noexcept
  {
    const float value = drive_ * drive_factor * input + offset_ + offset_delta;
    if (!adaa1_)
      return plain(value);
    const float difference = value - previous_;
    const float output = std::abs(difference) < threshold_
                           ? plain(0.5f * (value + previous_))
                           : (primitive(value) - primitive(previous_)) / difference;
    previous_ = value;
    return output;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return adaa1_ ? sizeof(previous_) : 0;
  }

private:
  float plain(float value) const noexcept
  {
    if (value < knots_.front())
      return values_.front() + slopes_.front() * (value - knots_.front());
    if (value > knots_.back())
      return values_.back() + slopes_.back() * (value - knots_.back());
    const float coordinate = (value - knots_.front()) / spacing_;
    const auto raw_index = static_cast<std::size_t>(std::max(0.0f, std::floor(coordinate)));
    const auto index = std::min(raw_index, knots_.size() - 2);
    const float local = coordinate - static_cast<float>(index);
    const float local2 = local * local;
    const float local3 = local2 * local;
    const float h00 = 2.0f * local3 - 3.0f * local2 + 1.0f;
    const float h10 = local3 - 2.0f * local2 + local;
    const float h01 = -2.0f * local3 + 3.0f * local2;
    const float h11 = local3 - local2;
    return h00 * values_[index] + h10 * spacing_ * slopes_[index]
           + h01 * values_[index + 1] + h11 * spacing_ * slopes_[index + 1];
  }

  float primitive(float value) const noexcept
  {
    if (value < knots_.front())
    {
      const float distance = value - knots_.front();
      return values_.front() * distance + 0.5f * slopes_.front() * distance * distance;
    }
    if (value > knots_.back())
    {
      const float distance = value - knots_.back();
      return cumulative_.back() + values_.back() * distance
             + 0.5f * slopes_.back() * distance * distance;
    }
    const float coordinate = (value - knots_.front()) / spacing_;
    const auto raw_index = static_cast<std::size_t>(std::max(0.0f, std::floor(coordinate)));
    const auto index = std::min(raw_index, knots_.size() - 2);
    const float local = coordinate - static_cast<float>(index);
    const float local2 = local * local;
    const float local3 = local2 * local;
    const float local4 = local2 * local2;
    const float integral = spacing_
                           * ((0.5f * local4 - local3 + local) * values_[index]
                              + (0.25f * local4 - (2.0f / 3.0f) * local3
                                 + 0.5f * local2)
                                  * spacing_ * slopes_[index]
                              + (-0.5f * local4 + local3) * values_[index + 1]
                              + (0.25f * local4 - (1.0f / 3.0f) * local3)
                                  * spacing_ * slopes_[index + 1]);
    return cumulative_[index] + integral;
  }

  std::vector<float> knots_;
  std::vector<float> values_;
  std::vector<float> slopes_;
  std::vector<float> cumulative_;
  float drive_;
  float offset_;
  float spacing_ = 0.0f;
  bool adaa1_ = false;
  float threshold_ = 1.0e-4f;
  float previous_ = 0.0f;
};

class SlowController
{
public:
  SlowController(const json& payload, const std::string& core_kind)
    : hidden_size_(
        static_cast<std::size_t>(positive_int(payload.at("hidden_size"), "slow hidden size"))),
      decimation_(
        static_cast<std::size_t>(positive_int(payload.at("decimation"), "slow decimation"))),
      weight_ih_(float_matrix(payload.at("weight_ih"), 3 * hidden_size_, 3, "slow weight_ih")),
      weight_hh_(float_matrix(
        payload.at("weight_hh"), 3 * hidden_size_, hidden_size_, "slow weight_hh")),
      bias_ih_(float_vector(payload.at("bias_ih"), "slow bias_ih")),
      bias_hh_(float_vector(payload.at("bias_hh"), "slow bias_hh")),
      projection_weight_(
        float_matrix(payload.at("projection_weight"), 3, hidden_size_, "slow projection weight")),
      projection_bias_(float_vector(payload.at("projection_bias"), "slow projection bias")),
      hidden_(hidden_size_, 0.0f),
      input_gates_(3 * hidden_size_, 0.0f),
      hidden_gates_(3 * hidden_size_, 0.0f)
  {
    require_equal(payload, "kind", "fssr-slow-gru-v1");
    require_equal(payload, "feature_order", json::array({"mean_abs", "mean_square", "mean"}));
    require_equal(payload, "gate_order", json::array({"reset", "update", "new"}));
    const std::string application = core_kind == "cascade"
                                      ? "first-shaper-drive-offset-and-output-gain"
                                      : "shaper-drive-offset-and-output-gain";
    require_equal(payload, "application", application);
    if (finite_float(payload.at("drive_log_range"), "slow drive range") != 0.25f
        || finite_float(payload.at("offset_range"), "slow offset range") != 0.1f
        || finite_float(payload.at("gain_log_range"), "slow gain range") != 0.25f)
      throw std::runtime_error("slow modulation constants are unsupported");
    if (bias_ih_.size() != 3 * hidden_size_ || bias_hh_.size() != 3 * hidden_size_
        || projection_bias_.size() != 3)
      throw std::runtime_error("slow controller bias has an invalid dimension");
  }

  void reset() noexcept
  {
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
    accumulator_.fill(0.0f);
    count_ = 0;
  }

  [[nodiscard]] std::array<float, 3> modulation() const noexcept
  {
    std::array<float, 3> raw{};
    for (std::size_t output = 0; output < 3; ++output)
    {
      raw[output] = projection_bias_[output];
      for (std::size_t input = 0; input < hidden_size_; ++input)
        raw[output] += projection_weight_[output][input] * hidden_[input];
    }
    return {
      std::exp(0.25f * std::tanh(raw[0])),
      0.1f * std::tanh(raw[1]),
      std::exp(0.25f * std::tanh(raw[2]))};
  }

  void observe(float sample) noexcept
  {
    accumulator_[0] += std::abs(sample);
    accumulator_[1] += sample * sample;
    accumulator_[2] += sample;
    ++count_;
    if (count_ != decimation_)
      return;
    const std::array<float, 3> features = {
      accumulator_[0] / static_cast<float>(decimation_),
      accumulator_[1] / static_cast<float>(decimation_),
      accumulator_[2] / static_cast<float>(decimation_)};
    for (std::size_t output = 0; output < 3 * hidden_size_; ++output)
    {
      input_gates_[output] = bias_ih_[output];
      for (std::size_t input = 0; input < 3; ++input)
        input_gates_[output] += weight_ih_[output][input] * features[input];
      hidden_gates_[output] = bias_hh_[output];
      for (std::size_t input = 0; input < hidden_size_; ++input)
        hidden_gates_[output] += weight_hh_[output][input] * hidden_[input];
    }
    for (std::size_t unit = 0; unit < hidden_size_; ++unit)
    {
      const float reset = sigmoid(input_gates_[unit] + hidden_gates_[unit]);
      const float update = sigmoid(
        input_gates_[hidden_size_ + unit] + hidden_gates_[hidden_size_ + unit]);
      const float candidate = std::tanh(
        input_gates_[2 * hidden_size_ + unit]
        + reset * hidden_gates_[2 * hidden_size_ + unit]);
      hidden_[unit] = candidate + update * (hidden_[unit] - candidate);
    }
    accumulator_.fill(0.0f);
    count_ = 0;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return hidden_.size() * sizeof(float) + accumulator_.size() * sizeof(float)
           + sizeof(count_);
  }

private:
  static float sigmoid(float value) noexcept
  {
    return 1.0f / (1.0f + std::exp(-value));
  }

  std::size_t hidden_size_;
  std::size_t decimation_;
  std::vector<std::vector<float>> weight_ih_;
  std::vector<std::vector<float>> weight_hh_;
  std::vector<float> bias_ih_;
  std::vector<float> bias_hh_;
  std::vector<std::vector<float>> projection_weight_;
  std::vector<float> projection_bias_;
  std::vector<float> hidden_;
  std::array<float, 3> accumulator_{};
  std::size_t count_ = 0;
  std::vector<float> input_gates_;
  std::vector<float> hidden_gates_;
};

class ResidualLayer
{
public:
  ResidualLayer(
    const json& payload,
    int dilation,
    std::size_t channels,
    std::size_t kernel_size,
    bool separable)
    : channels_(channels),
      kernel_size_(kernel_size),
      dilation_(static_cast<std::size_t>(dilation)),
      history_size_((kernel_size - 1) * dilation_),
      history_(channels * history_size_, 0.0f),
      depthwise_(channels, 0.0f),
      update_(channels, 0.0f),
      separable_(separable)
  {
    if (separable_)
    {
      depthwise_weight_ =
        float_matrix(payload.at("depthwise_weight"), channels, kernel_size, "depthwise weight");
      depthwise_bias_ = float_vector(payload.at("depthwise_bias"), "depthwise bias");
      pointwise_weight_ =
        float_matrix(payload.at("pointwise_weight"), channels, channels, "pointwise weight");
      pointwise_bias_ = float_vector(payload.at("pointwise_bias"), "pointwise bias");
      if (depthwise_bias_.size() != channels || pointwise_bias_.size() != channels)
        throw std::runtime_error("separable residual layer bias has an invalid dimension");
    }
    else
    {
      convolution_weight_ = float_tensor3(
        payload.at("convolution_weight"), channels, channels, kernel_size, "convolution weight");
      convolution_bias_ = float_vector(payload.at("convolution_bias"), "convolution bias");
      if (convolution_bias_.size() != channels)
        throw std::runtime_error("full residual layer bias has an invalid dimension");
    }
  }

  void reset() noexcept
  {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_index_ = 0;
  }

  const std::vector<float>& process(const std::vector<float>& input) noexcept
  {
    if (separable_)
    {
      for (std::size_t channel = 0; channel < channels_; ++channel)
      {
        float value = depthwise_bias_[channel];
        for (std::size_t tap = 0; tap < kernel_size_; ++tap)
        {
          const auto lag = (kernel_size_ - 1 - tap) * dilation_;
          const float sample = lag == 0
                                 ? input[channel]
                                 : history_[channel * history_size_
                                            + (write_index_ + history_size_ - lag) % history_size_];
          value += depthwise_weight_[channel][tap] * sample;
        }
        depthwise_[channel] = value;
      }
      for (std::size_t output = 0; output < channels_; ++output)
      {
        float value = pointwise_bias_[output];
        for (std::size_t input_channel = 0; input_channel < channels_; ++input_channel)
          value += pointwise_weight_[output][input_channel] * depthwise_[input_channel];
        update_[output] = value;
      }
    }
    else
    {
      for (std::size_t output = 0; output < channels_; ++output)
      {
        float value = convolution_bias_[output];
        for (std::size_t input_channel = 0; input_channel < channels_; ++input_channel)
        {
          for (std::size_t tap = 0; tap < kernel_size_; ++tap)
          {
            const auto lag = (kernel_size_ - 1 - tap) * dilation_;
            const float sample = lag == 0
                                   ? input[input_channel]
                                   : history_[input_channel * history_size_
                                              + (write_index_ + history_size_ - lag) % history_size_];
            value += convolution_weight_[output][input_channel][tap] * sample;
          }
        }
        update_[output] = value;
      }
    }
    for (std::size_t channel = 0; channel < channels_; ++channel)
      history_[channel * history_size_ + write_index_] = input[channel];
    write_index_ = (write_index_ + 1) % history_size_;
    return update_;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    return history_.size() * sizeof(float) + sizeof(write_index_);
  }

private:
  std::vector<std::vector<float>> depthwise_weight_;
  std::vector<float> depthwise_bias_;
  std::vector<std::vector<float>> pointwise_weight_;
  std::vector<float> pointwise_bias_;
  std::vector<std::vector<std::vector<float>>> convolution_weight_;
  std::vector<float> convolution_bias_;
  std::size_t channels_;
  std::size_t kernel_size_;
  std::size_t dilation_;
  std::size_t history_size_;
  std::vector<float> history_;
  std::size_t write_index_ = 0;
  std::vector<float> depthwise_;
  std::vector<float> update_;
  bool separable_;
};

class Residual
{
public:
  explicit Residual(const json& payload, bool r2)
    : channels_(static_cast<std::size_t>(positive_int(payload.at("channels"), "residual channels"))),
      kernel_size_(
        static_cast<std::size_t>(positive_int(payload.at("kernel_size"), "residual kernel size"))),
      negative_slope_(finite_float(payload.at("negative_slope"), "negative slope")),
      scale_(finite_float(payload.at("scale"), "residual scale")),
      input_weight_(float_matrix(
        payload.at("input_projection").at("weight"), channels_, 2, "input projection weight")),
      input_bias_(float_vector(payload.at("input_projection").at("bias"), "input projection bias")),
      output_weight_(
        float_vector(payload.at("output_projection").at("weight"), "output projection weight")),
      output_bias_(
        finite_float(payload.at("output_projection").at("bias"), "output projection bias")),
      hidden_(channels_, 0.0f)
  {
    const auto kind = payload.at("kind").get<std::string>();
    const bool separable = kind == "causal-depthwise-separable-tcn";
    const bool full = kind == "causal-full-convolution-tcn";
    if (!separable && !full)
      throw std::runtime_error("unsupported residual operator");
    require_equal(payload, "input_features", json::array({"input", "core"}));
    const std::size_t required_channels = r2 ? 16 : 8;
    if (channels_ != required_channels || kernel_size_ != 3)
      throw std::runtime_error("residual channel count or kernel size is unsupported");
    if (negative_slope_ < 0.0f || scale_ < 0.0f || scale_ > 1.0f)
      throw std::runtime_error("residual activation parameters are out of range");
    if (input_bias_.size() != channels_ || output_weight_.size() != channels_)
      throw std::runtime_error("residual projection has an invalid dimension");
    const int receptive_field = positive_int(payload.at("receptive_field"), "receptive field");
    std::vector<int> expected;
    if (r2)
    {
      if (!payload.at("dilations").is_array() || payload.at("dilations").empty())
        throw std::runtime_error("R2 residual dilation schedule is missing");
      const int scale = positive_int(payload.at("dilations")[0], "R2 dilation scale");
      if (scale != 1 && scale != 2 && scale != 4)
        throw std::runtime_error("R2 dilation scale is unsupported");
      constexpr int base[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
      for (const int dilation : base)
        expected.push_back(scale * dilation);
      if (receptive_field != 1 + 2 * 1023 * scale)
        throw std::runtime_error("R2 residual receptive field is inconsistent");
    }
    else
    {
      expected = receptive_field == 31
                   ? std::vector<int>{1, 2, 4, 8}
                   : receptive_field == 2047
                       ? std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
                       : std::vector<int>{};
    }
    if (expected.empty() || !payload.at("dilations").is_array()
        || payload.at("dilations").size() != expected.size()
        || payload.at("layers").size() != expected.size())
      throw std::runtime_error("unsupported residual receptive field");
    if (full && (r2 || receptive_field != 31))
      throw std::runtime_error("the registered full-convolution ablation is RF31 only");
    for (std::size_t index = 0; index < expected.size(); ++index)
    {
      if (positive_int(payload.at("dilations")[index], "residual dilation") != expected[index])
        throw std::runtime_error("residual dilation schedule does not match its receptive field");
      layers_.emplace_back(
        payload.at("layers")[index], expected[index], channels_, kernel_size_, separable);
    }
  }

  void reset() noexcept
  {
    for (auto& layer : layers_)
      layer.reset();
    std::fill(hidden_.begin(), hidden_.end(), 0.0f);
  }

  float process(float input, float core) noexcept
  {
    for (std::size_t output = 0; output < channels_; ++output)
      hidden_[output] = input_bias_[output] + input_weight_[output][0] * input
                        + input_weight_[output][1] * core;
    for (auto& layer : layers_)
    {
      const auto& update = layer.process(hidden_);
      for (std::size_t channel = 0; channel < channels_; ++channel)
      {
        const float activated = update[channel] >= 0.0f
                                  ? update[channel]
                                  : negative_slope_ * update[channel];
        hidden_[channel] += activated;
      }
    }
    float raw = output_bias_;
    for (std::size_t channel = 0; channel < channels_; ++channel)
      raw += output_weight_[channel] * hidden_[channel];
    return scale_ * std::tanh(raw);
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    std::size_t result = hidden_.size() * sizeof(float);
    for (const auto& layer : layers_)
      result += layer.state_size_bytes();
    return result;
  }

private:
  std::size_t channels_;
  std::size_t kernel_size_;
  float negative_slope_;
  float scale_;
  std::vector<std::vector<float>> input_weight_;
  std::vector<float> input_bias_;
  std::vector<float> output_weight_;
  float output_bias_;
  std::vector<ResidualLayer> layers_;
  std::vector<float> hidden_;
};

class AANAMLayer
{
public:
  explicit AANAMLayer(const json& payload)
    : kernel_size_(static_cast<std::size_t>(positive_int(payload.at("kernel_size"), "AA-NAM kernel"))),
      dilation_(static_cast<std::size_t>(positive_int(payload.at("dilation"), "AA-NAM dilation"))),
      conv_weight_(float_matrix(payload.at("conv_weight"), 8, 8 * kernel_size_, "AA-NAM convolution")),
      conv_bias_(float_vector(payload.at("conv_bias"), "AA-NAM convolution bias")),
      condition_weight_(float_vector(payload.at("condition_weight"), "AA-NAM condition weight")),
      residual_weight_(float_matrix(payload.at("residual_weight"), 8, 8, "AA-NAM residual weight")),
      residual_bias_(float_vector(payload.at("residual_bias"), "AA-NAM residual bias")),
      history_(8 * (kernel_size_ - 1) * dilation_, 0.0f)
  {
    if (conv_bias_.size() != 8 || condition_weight_.size() != 8
        || residual_bias_.size() != 8)
      throw std::runtime_error("AA-NAM layer vectors must contain eight channels");
    activations_.reserve(8);
    for (std::size_t channel = 0; channel < 8; ++channel)
      activations_.emplace_back(payload.at("activation"));
  }

  void reset() noexcept
  {
    std::fill(history_.begin(), history_.end(), 0.0f);
    write_index_ = 0;
    for (auto& activation : activations_)
      activation.reset();
  }

  void process(float condition, std::array<float, 8>& hidden, std::array<float, 8>& head) noexcept
  {
    std::array<float, 8> activated{};
    const auto history_samples = (kernel_size_ - 1) * dilation_;
    for (std::size_t output = 0; output < 8; ++output)
    {
      float sum = conv_bias_[output] + condition_weight_[output] * condition;
      for (std::size_t tap = 0; tap < kernel_size_; ++tap)
      {
        const auto lag = (kernel_size_ - 1 - tap) * dilation_;
        for (std::size_t input = 0; input < 8; ++input)
        {
          const float source = lag == 0
                                 ? hidden[input]
                                 : history_[((write_index_ + history_samples - lag)
                                             % history_samples)
                                              * 8
                                            + input];
          sum += conv_weight_[output][input * kernel_size_ + tap] * source;
        }
      }
      activated[output] = activations_[output].process(sum);
    }
    if (history_samples != 0)
    {
      for (std::size_t channel = 0; channel < 8; ++channel)
        history_[write_index_ * 8 + channel] = hidden[channel];
      write_index_ = (write_index_ + 1) % history_samples;
    }
    std::array<float, 8> update{};
    for (std::size_t output = 0; output < 8; ++output)
    {
      update[output] = residual_bias_[output];
      for (std::size_t input = 0; input < 8; ++input)
        update[output] += residual_weight_[output][input] * activated[input];
    }
    for (std::size_t channel = 0; channel < 8; ++channel)
    {
      hidden[channel] += update[channel];
      head[channel] += activated[channel];
    }
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    std::size_t result = history_.size() * sizeof(float) + sizeof(write_index_);
    for (const auto& activation : activations_)
      result += activation.state_size_bytes();
    return result;
  }

private:
  std::size_t kernel_size_;
  std::size_t dilation_;
  std::vector<std::vector<float>> conv_weight_;
  std::vector<float> conv_bias_;
  std::vector<float> condition_weight_;
  std::vector<std::vector<float>> residual_weight_;
  std::vector<float> residual_bias_;
  std::vector<HermiteSpline> activations_;
  std::vector<float> history_;
  std::size_t write_index_ = 0;
};

class AANAMWaveNet
{
public:
  AANAMWaveNet(const json& payload, bool adaa1)
    : input_projection_(float_matrix(payload.at("input_projection"), 8, 1, "AA-NAM input projection")),
      head_scale_(finite_float(payload.at("head_scale"), "AA-NAM head scale")),
      receptive_field_(static_cast<std::size_t>(positive_int(payload.at("receptive_field"), "AA-NAM receptive field")))
  {
    require_equal(payload, "kind", "pinned-a2-full-hermite-v1");
    require_equal(payload, "channels", 8);
    require_equal(payload, "layer_count", 23);
    if (!payload.at("layers").is_array() || payload.at("layers").size() != 23)
      throw std::runtime_error("AA-NAM requires 23 layers");
    layers_.reserve(23);
    for (const auto& layer : payload.at("layers"))
      layers_.emplace_back(layer);
    const auto& head = payload.at("head");
    head_kernel_ = static_cast<std::size_t>(positive_int(head.at("kernel_size"), "AA-NAM head kernel"));
    head_weight_ = float_matrix(head.at("weight"), 8, head_kernel_, "AA-NAM head weight");
    head_bias_ = finite_float(head.at("bias"), "AA-NAM head bias");
    head_history_.assign(8 * (head_kernel_ - 1), 0.0f);
    warmup_samples_ = receptive_field_ - 1 + (adaa1 ? 23 : 0);
    reset();
  }

  void reset() noexcept
  {
    for (auto& layer : layers_)
      layer.reset();
    std::fill(head_history_.begin(), head_history_.end(), 0.0f);
    head_write_index_ = 0;
    for (std::size_t sample = 0; sample < warmup_samples_; ++sample)
      process_unwarmed(0.0f);
  }

  float process(float input) noexcept { return process_unwarmed(input); }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    std::size_t result = head_history_.size() * sizeof(float) + sizeof(head_write_index_);
    for (const auto& layer : layers_)
      result += layer.state_size_bytes();
    return result;
  }

private:
  float process_unwarmed(float input) noexcept
  {
    std::array<float, 8> hidden{};
    std::array<float, 8> head{};
    for (std::size_t channel = 0; channel < 8; ++channel)
      hidden[channel] = input_projection_[channel][0] * input;
    for (auto& layer : layers_)
      layer.process(input, hidden, head);
    float output = head_bias_;
    const auto history_samples = head_kernel_ - 1;
    for (std::size_t channel = 0; channel < 8; ++channel)
      for (std::size_t tap = 0; tap < head_kernel_; ++tap)
      {
        const auto lag = head_kernel_ - 1 - tap;
        const float source = lag == 0
                               ? head[channel]
                               : head_history_[((head_write_index_ + history_samples - lag)
                                                % history_samples)
                                                 * 8
                                               + channel];
        output += head_weight_[channel][tap] * source;
      }
    if (history_samples != 0)
    {
      for (std::size_t channel = 0; channel < 8; ++channel)
        head_history_[head_write_index_ * 8 + channel] = head[channel];
      head_write_index_ = (head_write_index_ + 1) % history_samples;
    }
    return head_scale_ * output;
  }

  std::vector<std::vector<float>> input_projection_;
  std::vector<AANAMLayer> layers_;
  float head_scale_;
  std::size_t receptive_field_;
  std::size_t warmup_samples_ = 0;
  std::size_t head_kernel_ = 0;
  std::vector<std::vector<float>> head_weight_;
  float head_bias_ = 0.0f;
  std::vector<float> head_history_;
  std::size_t head_write_index_ = 0;
};
} // namespace

struct Model::Impl
{
  explicit Impl(const std::filesystem::path& model_path)
  {
    std::ifstream stream(model_path);
    if (!stream)
      throw std::runtime_error("cannot open R1 model: " + model_path.string());
    json document;
    stream >> document;
    if (!document.is_object())
      throw std::runtime_error("R1 model root must be a JSON object");
    const auto format = document.at("format").get<std::string>();
    is_r2 = format == "fssr-r2-native-v1";
    if (!is_r2 && format != "fssr-r1-native-v1")
      throw std::runtime_error("native model format is unsupported");
    require_equal(document, "version", 1);
    require_equal(document, "campaign_version", is_r2 ? "FSSR-R2-v1" : "FSSR-R1");
    require_equal(document, "precision", "float32");
    sample_rate_hz = positive_int(document.at("sample_rate_hz"), "sample rate");
    latency_samples = document.at("latency_samples").get<int>();
    if (latency_samples < 0)
      throw std::runtime_error("latency_samples must be non-negative");
    if (is_r2)
    {
      factor = positive_int(document.at("dilation_scale"), "dilation scale");
      if (factor != 1 && factor != 2 && factor != 4)
        throw std::runtime_error("R2 dilation scale is unsupported");
      if (positive_int(document.at("internal_sample_rate"), "internal sample rate")
          != sample_rate_hz * factor)
        throw std::runtime_error("R2 internal sample rate is inconsistent");
      const auto aa_mode = document.at("aa_mode").get<std::string>();
      const int expected_factor = aa_mode == "full_island_x2" ? 2
                                  : aa_mode == "teacher_x4"   ? 4
                                  : aa_mode == "off" || aa_mode == "adaa1" ? 1
                                                                               : 0;
      const int expected_latency = factor > 1 ? 16 : aa_mode == "adaa1" ? 1 : 0;
      if (factor != expected_factor || latency_samples != expected_latency)
        throw std::runtime_error("R2 AA mode, factor, and latency are inconsistent");
      const auto family = document.at("family").get<std::string>();
      is_aa_nam = family == "aa-nam";
      if (!is_aa_nam && family != "aa-fssr")
        throw std::runtime_error("R2 family is unsupported");
      if (is_aa_nam)
        aanam = std::make_unique<AANAMWaveNet>(document.at("wavenet"), aa_mode == "adaa1");
    }
    else if (latency_samples != 0)
    {
      throw std::runtime_error("R1 latency must be zero");
    }
    if (!is_aa_nam)
    {
    const auto& core = document.at("core");
    const auto kind = core.at("kind").get<std::string>();
    const std::size_t shaper_count = kind == "mono" ? 1 : kind == "cascade" ? 2 : 0;
    if (shaper_count == 0 || !core.at("filters").is_array()
        || core.at("filters").size() != shaper_count + 1 || !core.at("shapers").is_array()
        || core.at("shapers").size() != shaper_count)
      throw std::runtime_error("core topology does not match mono or cascade");
    for (const auto& filter : core.at("filters"))
      filters.emplace_back(filter);
    for (const auto& shaper : core.at("shapers"))
      shapers.emplace_back(shaper);
    output_gain = finite_float(core.at("output_gain"), "core output gain");
    if (document.at("slow_controller").is_string())
    {
      require_equal(document, "slow_controller", "none");
    }
    else if (document.at("slow_controller").is_object())
    {
      slow = std::make_unique<SlowController>(document.at("slow_controller"), kind);
    }
    else
    {
      throw std::runtime_error("slow_controller must be none or an object");
    }
    if (!document.at("residual").is_null())
      residual = std::make_unique<Residual>(document.at("residual"), is_r2);
    }
    if (is_r2 && factor > 1)
    {
      const auto& resampling = document.at("resampling");
      if (!resampling.is_object()
          || positive_int(resampling.at("factor"), "resampling factor") != factor
          || positive_int(resampling.at("linear_delay_samples"), "linear delay") != 16)
        throw std::runtime_error("R2 resampling declaration is invalid");
      upsample = std::make_unique<CausalFir>(
        json{{"coefficients", resampling.at("upsample_coefficients")}});
      downsample = std::make_unique<CausalFir>(
        json{{"coefficients", resampling.at("downsample_coefficients")}});
      linear_delay = std::make_unique<CausalDelay>(16);
    }
    else if (is_r2 && !document.at("resampling").is_null())
    {
      throw std::runtime_error("base-rate R2 model cannot include resampling");
    }
    if (is_r2 && factor == 1 && latency_samples > 0)
      linear_delay = std::make_unique<CausalDelay>(static_cast<std::size_t>(latency_samples));
  }

  void reset() noexcept
  {
    for (auto& filter : filters)
      filter.reset();
    for (auto& shaper : shapers)
      shaper.reset();
    if (slow)
      slow->reset();
    if (residual)
      residual->reset();
    if (upsample)
      upsample->reset();
    if (downsample)
      downsample->reset();
    if (linear_delay)
      linear_delay->reset();
    if (aanam)
      aanam->reset();
  }

  float process_internal(float input) noexcept
  {
    if (aanam)
      return aanam->process(input);
    const std::array<float, 3> modulation = slow
                                              ? slow->modulation()
                                              : std::array<float, 3>{1.0f, 0.0f, 1.0f};
    float core = filters.front().process(input);
    for (std::size_t index = 0; index < shapers.size(); ++index)
      core = filters[index + 1].process(
        index == 0
          ? shapers[index].process(core, modulation[0], modulation[1])
          : shapers[index].process(core));
    core *= output_gain * modulation[2];
    if (slow)
      slow->observe(input);
    return residual ? core + residual->process(input, core) : core;
  }

  float process(float input) noexcept
  {
    if (factor == 1)
    {
      const float internal = process_internal(input);
      return linear_delay ? linear_delay->process(internal) : internal;
    }
    float selected = 0.0f;
    for (int phase = 0; phase < factor; ++phase)
    {
      const float inserted = phase == 0 ? input : 0.0f;
      const float interpolated = upsample->process(inserted);
      const float branch = process_internal(interpolated);
      const float filtered = downsample->process(branch - interpolated);
      if (phase == 0)
        selected = filtered;
    }
    return linear_delay->process(input) + selected;
  }

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    std::size_t result = 0;
    for (const auto& filter : filters)
      result += filter.state_size_bytes();
    for (const auto& shaper : shapers)
      result += shaper.state_size_bytes();
    if (slow)
      result += slow->state_size_bytes();
    if (residual)
      result += residual->state_size_bytes();
    if (upsample)
      result += upsample->state_size_bytes();
    if (downsample)
      result += downsample->state_size_bytes();
    if (linear_delay)
      result += linear_delay->state_size_bytes();
    if (aanam)
      result += aanam->state_size_bytes();
    return result;
  }

  bool is_r2 = false;
  bool is_aa_nam = false;
  int factor = 1;
  int sample_rate_hz = 0;
  int latency_samples = 0;
  float output_gain = 1.0f;
  std::vector<CausalFir> filters;
  std::vector<HermiteSpline> shapers;
  std::unique_ptr<SlowController> slow;
  std::unique_ptr<Residual> residual;
  std::unique_ptr<CausalFir> upsample;
  std::unique_ptr<CausalFir> downsample;
  std::unique_ptr<CausalDelay> linear_delay;
  std::unique_ptr<AANAMWaveNet> aanam;
};

Model::Model(const std::filesystem::path& model_path) : impl_(std::make_unique<Impl>(model_path)) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

void Model::reset() noexcept
{
  impl_->reset();
}

void Model::process(const float* input, float* output, std::size_t sample_count) noexcept
{
  for (std::size_t index = 0; index < sample_count; ++index)
    output[index] = impl_->process(input[index]);
}

int Model::sample_rate_hz() const noexcept
{
  return impl_->sample_rate_hz;
}

int Model::latency_samples() const noexcept
{
  return impl_->latency_samples;
}

std::size_t Model::state_size_bytes() const noexcept
{
  return impl_->state_size_bytes();
}
} // namespace fssr::r1
