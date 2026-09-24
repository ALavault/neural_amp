#include "ssm_wavenet_native.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "json.hpp"

namespace fssr::ssm
{
namespace
{
using json = nlohmann::json;
using Complex = std::complex<double>;
Complex phi2(Complex z)
{
  if (std::abs(z) < 1e-4)
    return 0.5 + z / 6.0 + z * z / 24.0;
  return (std::exp(z) - 1.0 - z) / (z * z);
}
std::vector<double> values(const json& j, int count)
{
  auto result = j.get<std::vector<double>>();
  if (result.size() != static_cast<std::size_t>(count))
    throw std::runtime_error("invalid vector dimension");
  for (double v : result)
    if (!std::isfinite(v)) throw std::runtime_error("non-finite parameter");
  return result;
}
std::vector<double> matrix(const json& j, int rows, int cols)
{
  if (!j.is_array() || j.size() != static_cast<std::size_t>(rows))
    throw std::runtime_error("invalid matrix dimension");
  std::vector<double> result;
  for (const auto& row : j)
  {
    const auto v = values(row, cols);
    result.insert(result.end(), v.begin(), v.end());
  }
  return result;
}
}

Model::Model(const std::filesystem::path& path)
{
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open model");
  json j;
  file >> j;
  const auto& arch = j.at("architecture");
  if (j.at("format") != "ssm-wavenet-native-v1" || j.at("version") != 1 ||
      j.at("precision") != "float32" || j.at("latency_samples") != 0 ||
      arch.at("discretization") != "foh" || arch.at("act_type") != "gated")
    throw std::runtime_error("unsupported native SSM format");
  channels_ = arch.at("channels").get<int>();
  modes_ = arch.at("state_dim").get<int>();
  const int count = arch.at("num_blocks").get<int>();
  trained_rate_ = j.at("trained_sample_rate_hz").get<double>();
  output_act_ = arch.at("output_act").get<std::string>();
  if (channels_ <= 0 || modes_ <= 0 || count <= 0 ||
      !std::isfinite(trained_rate_) || trained_rate_ <= 0 ||
      (output_act_ != "tanh" && output_act_ != "softsign" && output_act_ != "none") ||
      j.at("blocks").size() != static_cast<std::size_t>(count))
    throw std::runtime_error("invalid architecture");
  auto conv = [](const json& data, int rows, int cols, bool flat = false) {
    Conv c;
    auto w = flat ? values(data.at("weight"), rows) : matrix(data.at("weight"), rows, cols);
    auto b = values(data.at("bias"), rows);
    c.weight.resize(w.size());  // transposed: weight[k * rows + r] = W[r][k]
    for (int r = 0; r < rows; ++r)
      for (int k = 0; k < cols; ++k) c.weight[k * rows + r] = static_cast<float>(w[r * cols + k]);
    c.bias.assign(b.begin(), b.end());
    return c;
  };
  input_ = conv(j.at("input_conv"), channels_, 1, true);
  conv1_ = conv(j.at("output_net").at("conv1"), channels_, channels_);
  conv2_ = conv(j.at("output_net").at("conv2"), 1, channels_);
  for (const auto& data : j.at("blocks"))
  {
    Block b;
    const auto& s = data.at("ssm");
    b.log_dt = values(s.at("log_dt"), channels_);
    auto re = matrix(s.at("lambda_re"), channels_, modes_);
    auto im = matrix(s.at("lambda_im"), channels_, modes_);
    auto cr = matrix(s.at("c_re"), channels_, modes_);
    auto ci = matrix(s.at("c_im"), channels_, modes_);
    for (std::size_t k = 0; k < re.size(); ++k)
    {
      if (re[k] >= 0) throw std::runtime_error("SSM poles must be stable");
      b.lambda.emplace_back(re[k], im[k]);
      b.c.emplace_back(cr[k], ci[k]);
    }
    auto d = values(s.at("d"), channels_);
    b.d.assign(d.begin(), d.end());
    b.previous.resize(channels_);
    for (auto* v : {&b.a_re, &b.a_im, &b.cb1_re, &b.cb1_im, &b.cb0_re, &b.cb0_im,
                    &b.s_re, &b.s_im})
      v->resize(re.size());
    b.gate = conv(data.at("pre_gate"), 2 * channels_, channels_);
    b.residual = conv(data.at("res_conv"), channels_, channels_);
    b.skip = conv(data.at("skip_conv"), channels_, channels_);
    blocks_.push_back(std::move(b));
  }
  h_.resize(channels_); sum_.resize(channels_); y_.resize(channels_);
  v_.resize(2 * channels_); g_.resize(channels_); scratch_.resize(channels_);
  output_.resize(1);
  u_.resize(channels_ * modes_); u_prev_.resize(channels_ * modes_);
  set_sample_rate(48000);
  reset();
}

void Model::set_sample_rate(double fs)
{
  if (!std::isfinite(fs) || fs <= 0) throw std::runtime_error("invalid sample rate");
  for (auto& b : blocks_)
    for (int k = 0; k < channels_; ++k)
    {
      const double dt = std::exp(b.log_dt[k]) * trained_rate_ / fs;
      for (int m = 0; m < modes_; ++m)
      {
        const int i = k * modes_ + m;
        const Complex z = b.lambda[i] * dt, a = std::exp(z);
        const Complex cb1 = b.c[i] * dt * phi2(z), cb0 = b.c[i] * dt * a * phi2(-z);
        b.a_re[i] = static_cast<float>(a.real());
        b.a_im[i] = static_cast<float>(a.imag());
        b.cb1_re[i] = static_cast<float>(cb1.real());
        b.cb1_im[i] = static_cast<float>(cb1.imag());
        b.cb0_re[i] = static_cast<float>(cb0.real());
        b.cb0_im[i] = static_cast<float>(cb0.imag());
      }
    }
}

void Model::reset()
{
  for (auto& b : blocks_)
  {
    std::fill(b.s_re.begin(), b.s_re.end(), 0.0f);
    std::fill(b.s_im.begin(), b.s_im.end(), 0.0f);
    std::fill(b.previous.begin(), b.previous.end(), 0.0f);
  }
}

void Model::Conv::apply(const std::vector<float>& in, std::vector<float>& out) const
{
  // weight is stored input-major (transposed at load): each output keeps its own
  // accumulator, so the inner loop vectorises without reassociating any sum (IEEE kept).
  const std::size_t rows = bias.size();
  std::copy(bias.begin(), bias.end(), out.begin());
  for (std::size_t k = 0; k < in.size(); ++k)
  {
    const float x = in[k];
    const float* w = weight.data() + k * rows;
    for (std::size_t r = 0; r < rows; ++r)
      out[r] += w[r] * x;
  }
}

void Model::process(const float* in, float* out, int n)
{
  for (int t = 0; t < n; ++t)
  {
    for (int k = 0; k < channels_; ++k)
      h_[k] = input_.weight[k] * in[t] + input_.bias[k];
    std::fill(sum_.begin(), sum_.end(), 0.0f);
    for (auto& b : blocks_)
    {
      const int n_modes = channels_ * modes_;
      for (int k = 0; k < channels_; ++k)
        for (int m = 0; m < modes_; ++m)
        {
          u_[k * modes_ + m] = h_[k];
          u_prev_[k * modes_ + m] = b.previous[k];
        }
      // s = a s + cb1 u + cb0 u_prev, complex, written out over all modes at once.
      for (int i = 0; i < n_modes; ++i)
      {
        const float re = b.a_re[i] * b.s_re[i] - b.a_im[i] * b.s_im[i] +
                         b.cb1_re[i] * u_[i] + b.cb0_re[i] * u_prev_[i];
        const float im = b.a_re[i] * b.s_im[i] + b.a_im[i] * b.s_re[i] +
                         b.cb1_im[i] * u_[i] + b.cb0_im[i] * u_prev_[i];
        b.s_re[i] = re;
        b.s_im[i] = im;
      }
      for (int k = 0; k < channels_; ++k)
      {
        float total = 0;
        for (int m = 0; m < modes_; ++m) total += b.s_re[k * modes_ + m];
        y_[k] = 2 * total + b.d[k] * h_[k];
        b.previous[k] = h_[k];
      }
      b.gate.apply(y_, v_);
      for (int k = 0; k < channels_; ++k)
        g_[k] = std::tanh(v_[k]) / (1.0f + std::exp(-v_[channels_ + k]));
      b.skip.apply(g_, scratch_);
      for (int k = 0; k < channels_; ++k) sum_[k] += scratch_[k];
      b.residual.apply(g_, scratch_);
      for (int k = 0; k < channels_; ++k) h_[k] += scratch_[k];
    }
    for (auto& x : sum_) x = std::max(x, 0.0f);
    conv1_.apply(sum_, scratch_);
    for (auto& x : scratch_) x = std::max(x, 0.0f);
    conv2_.apply(scratch_, output_);
    float x = output_[0];
    if (output_act_ == "tanh") x = std::tanh(x);
    else if (output_act_ == "softsign") x /= 1.0f + std::abs(x);
    out[t] = x;
  }
}
} // namespace fssr::ssm
