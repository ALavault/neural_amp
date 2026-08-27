#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

namespace
{
std::vector<float> read_f32(const std::filesystem::path& path)
{
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open input: " + path.string());
  const auto bytes = stream.tellg();
  if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0)
    throw std::runtime_error("invalid float32 input length");
  std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(samples.data()), bytes);
  return samples;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& samples)
{
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open output: " + path.string());
  stream.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(float)));
}

std::vector<int> parse_blocks(const std::string& text)
{
  std::vector<int> blocks;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ','))
  {
    const int block = std::stoi(token);
    if (block < 1 || block > 4096)
      throw std::runtime_error("block sizes must be in [1,4096]");
    blocks.push_back(block);
  }
  if (blocks.empty())
    throw std::runtime_error("at least one block size is required");
  return blocks;
}

std::vector<float> process(nam::DSP& model, const std::vector<float>& input, const std::vector<int>& blocks)
{
  const int maximum = *std::max_element(blocks.begin(), blocks.end());
  model.Reset(48000.0, maximum);
  std::vector<NAM_SAMPLE> input_buffer(maximum, 0.0);
  std::vector<NAM_SAMPLE> output_buffer(maximum, 0.0);
  NAM_SAMPLE* input_ptrs[] = {input_buffer.data()};
  NAM_SAMPLE* output_ptrs[] = {output_buffer.data()};
  std::vector<float> output(input.size());
  std::size_t position = 0;
  std::size_t block_index = 0;
  while (position < input.size())
  {
    const int count = std::min<int>(blocks[block_index % blocks.size()], input.size() - position);
    for (int index = 0; index < count; ++index)
      input_buffer[index] = static_cast<NAM_SAMPLE>(input[position + index]);
    model.process(input_ptrs, output_ptrs, count);
    for (int index = 0; index < count; ++index)
      output[position + index] = static_cast<float>(output_buffer[index]);
    position += count;
    ++block_index;
  }
  return output;
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6)
  {
    std::cerr << "Usage: nam_block_runner MODEL INPUT.f32 OUTPUT.f32 BLOCKS [RESET_OUTPUT.f32]\n";
    return 2;
  }
  try
  {
    auto model = nam::get_dsp(std::filesystem::path(argv[1]));
    if (!model)
      throw std::runtime_error("model failed to load");
    if (model->NumInputChannels() != 1 || model->NumOutputChannels() != 1)
      throw std::runtime_error("only mono models are supported");
    const auto input = read_f32(argv[2]);
    const auto blocks = parse_blocks(argv[4]);
    write_f32(argv[3], process(*model, input, blocks));
    if (argc == 6)
      write_f32(argv[5], process(*model, input, blocks));
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
