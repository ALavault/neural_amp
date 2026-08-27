#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fssr_r1_native.hpp"

namespace
{
std::vector<float> read_f32(const std::filesystem::path& path)
{
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open input file");
  const auto byte_count = stream.tellg();
  if (byte_count < 0 || byte_count % static_cast<std::streamoff>(sizeof(float)) != 0)
    throw std::runtime_error("input is not a raw float32 file");
  std::vector<float> values(static_cast<std::size_t>(byte_count) / sizeof(float));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(values.data()), byte_count);
  if (!stream && !values.empty())
    throw std::runtime_error("failed to read input file");
  return values;
}

void write_f32(const std::filesystem::path& path, const std::vector<float>& values)
{
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    throw std::runtime_error("cannot open output file");
  stream.write(
    reinterpret_cast<const char*>(values.data()),
    static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream)
    throw std::runtime_error("failed to write output file");
}

std::vector<std::size_t> parse_blocks(const std::string& text)
{
  std::vector<std::size_t> blocks;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ','))
  {
    const auto block = std::stoul(token);
    if (block < 1 || block > 4096)
      throw std::runtime_error("block sizes must be in [1,4096]");
    blocks.push_back(block);
  }
  if (blocks.empty())
    throw std::runtime_error("at least one block size is required");
  return blocks;
}

std::vector<float> process(
  fssr::r1::Model& model,
  const std::vector<float>& input,
  const std::vector<std::size_t>& blocks)
{
  model.reset();
  std::vector<float> output(input.size(), 0.0f);
  std::size_t position = 0;
  std::size_t block_index = 0;
  while (position < input.size())
  {
    const auto count = std::min(blocks[block_index % blocks.size()], input.size() - position);
    model.process(input.data() + position, output.data() + position, count);
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
    std::cerr << "Usage: r1_block_runner MODEL INPUT.f32 OUTPUT.f32 BLOCKS [RESET_OUTPUT.f32]\n";
    return 2;
  }
  try
  {
    fssr::r1::Model model(argv[1]);
    const auto input = read_f32(argv[2]);
    const auto blocks = parse_blocks(argv[4]);
    write_f32(argv[3], process(model, input, blocks));
    if (argc == 6)
      write_f32(argv[5], process(model, input, blocks));
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
