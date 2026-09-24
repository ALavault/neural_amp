#pragma once

#include <complex>
#include <filesystem>
#include <string>
#include <vector>

namespace fssr::ssm
{
// Native v1, mono FOH. process() allocates no memory; rate changes retain state.
class Model
{
public:
  explicit Model(const std::filesystem::path& path);
  void set_sample_rate(double fs);
  void process(const float* in, float* out, int n);
  void reset();

private:
  struct Conv
  {
    std::vector<float> weight, bias;
    void apply(const std::vector<float>& in, std::vector<float>& out) const;
  };
  struct Block
  {
    std::vector<double> log_dt;
    std::vector<std::complex<double>> lambda, c;
    // Structure of arrays, one entry per (channel, mode): the recurrence vectorises.
    std::vector<float> a_re, a_im, cb1_re, cb1_im, cb0_re, cb0_im, s_re, s_im;
    std::vector<float> d, previous;
    Conv gate, residual, skip;
  };
  int channels_, modes_;
  double trained_rate_;
  std::string output_act_;
  Conv input_, conv1_, conv2_;
  std::vector<Block> blocks_;
  std::vector<float> h_, sum_, y_, v_, g_, scratch_, output_, u_, u_prev_;
};
} // namespace fssr::ssm
