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

class HermiteSpline
{
public:
  explicit HermiteSpline(const json& payload)
    : knots_(float_vector(payload.at("knots"), "spline knots")),
      values_(float_vector(payload.at("values"), "spline values")),
      slopes_(float_vector(payload.at("slopes"), "spline slopes")),
      drive_(finite_float(payload.at("drive"), "spline drive")),
      offset_(finite_float(payload.at("offset"), "spline offset"))
  {
    if (knots_.size() < 4 || values_.size() != knots_.size() || slopes_.size() != knots_.size())
      throw std::runtime_error("spline arrays have incompatible dimensions");
    for (std::size_t index = 1; index < knots_.size(); ++index)
      if (!(knots_[index - 1] < knots_[index]))
        throw std::runtime_error("spline knots must be strictly increasing");
  }

  float process(
    float input,
    float drive_factor = 1.0f,
    float offset_delta = 0.0f) const noexcept
  {
    const float value = drive_ * drive_factor * input + offset_ + offset_delta;
    if (value < knots_.front())
      return values_.front() + slopes_.front() * (value - knots_.front());
    if (value > knots_.back())
      return values_.back() + slopes_.back() * (value - knots_.back());
    const auto upper = std::upper_bound(knots_.begin(), knots_.end(), value);
    const auto raw_index = static_cast<std::size_t>(std::distance(knots_.begin(), upper));
    const auto index = std::min(std::max<std::size_t>(raw_index, 1) - 1, knots_.size() - 2);
    const float spacing = knots_[index + 1] - knots_[index];
    const float local = (value - knots_[index]) / spacing;
    const float local2 = local * local;
    const float local3 = local2 * local;
    const float h00 = 2.0f * local3 - 3.0f * local2 + 1.0f;
    const float h10 = local3 - 2.0f * local2 + local;
    const float h01 = -2.0f * local3 + 3.0f * local2;
    const float h11 = local3 - local2;
    return h00 * values_[index] + h10 * spacing * slopes_[index]
           + h01 * values_[index + 1] + h11 * spacing * slopes_[index + 1];
  }

private:
  std::vector<float> knots_;
  std::vector<float> values_;
  std::vector<float> slopes_;
  float drive_;
  float offset_;
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
  explicit Residual(const json& payload)
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
    if (channels_ != 8 || kernel_size_ != 3)
      throw std::runtime_error("R1 residual requires eight channels and kernel size three");
    if (negative_slope_ < 0.0f || scale_ < 0.0f || scale_ > 1.0f)
      throw std::runtime_error("residual activation parameters are out of range");
    if (input_bias_.size() != channels_ || output_weight_.size() != channels_)
      throw std::runtime_error("residual projection has an invalid dimension");
    const int receptive_field = positive_int(payload.at("receptive_field"), "receptive field");
    const std::vector<int> expected = receptive_field == 31
                                        ? std::vector<int>{1, 2, 4, 8}
                                        : receptive_field == 2047
                                            ? std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
                                            : std::vector<int>{};
    if (expected.empty() || !payload.at("dilations").is_array()
        || payload.at("dilations").size() != expected.size()
        || payload.at("layers").size() != expected.size())
      throw std::runtime_error("unsupported residual receptive field");
    if (full && receptive_field != 31)
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
    require_equal(document, "format", "fssr-r1-native-v1");
    require_equal(document, "version", 1);
    require_equal(document, "campaign_version", "FSSR-R1");
    require_equal(document, "precision", "float32");
    require_equal(document, "latency_samples", 0);
    sample_rate_hz = positive_int(document.at("sample_rate_hz"), "sample rate");
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
      residual = std::make_unique<Residual>(document.at("residual"));
  }

  void reset() noexcept
  {
    for (auto& filter : filters)
      filter.reset();
    if (slow)
      slow->reset();
    if (residual)
      residual->reset();
  }

  float process(float input) noexcept
  {
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

  [[nodiscard]] std::size_t state_size_bytes() const noexcept
  {
    std::size_t result = 0;
    for (const auto& filter : filters)
      result += filter.state_size_bytes();
    if (slow)
      result += slow->state_size_bytes();
    if (residual)
      result += residual->state_size_bytes();
    return result;
  }

  int sample_rate_hz = 0;
  float output_gain = 1.0f;
  std::vector<CausalFir> filters;
  std::vector<HermiteSpline> shapers;
  std::unique_ptr<SlowController> slow;
  std::unique_ptr<Residual> residual;
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
  return 0;
}

std::size_t Model::state_size_bytes() const noexcept
{
  return impl_->state_size_bytes();
}
} // namespace fssr::r1
