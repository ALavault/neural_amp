#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ssm_wavenet_native.hpp"
#include "json.hpp"

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
    if (block < 1 || block > 2147483647)
      throw std::runtime_error("block sizes must be in [1,INT_MAX]");
    blocks.push_back(block);
  }
  if (blocks.empty())
    throw std::runtime_error("at least one block size is required");
  return blocks;
}

std::vector<float> process(
  fssr::ssm::Model& model,
  const std::vector<float>& input,
  const std::vector<std::size_t>& blocks,
  const nlohmann::json& events)
{
  model.reset();
  std::vector<float> output(input.size(), 0.0f);
  std::size_t position = 0;
  std::size_t block_index = 0, event_index = 0;
  while (position < input.size())
  {
    if (event_index < events.size() && events[event_index].at("sample") == position)
    {
      model.set_controls(events[event_index].at("controls").get<std::vector<float>>());
      ++event_index;
    }
    auto count = std::min(blocks[block_index % blocks.size()], input.size() - position);
    if (event_index < events.size())
      count = std::min(count, events[event_index].at("sample").get<std::size_t>() - position);
    model.process(input.data() + position, output.data() + position, count);
    position += count;
    ++block_index;
  }
  return output;
}
} // namespace

int main(int argc, char** argv)
{
  if (argc < 5 || argc > 7)
  {
    std::cerr << "Usage: ssm_wavenet_block_runner MODEL INPUT.f32 OUTPUT.f32 BLOCKS [SAMPLE_RATE [CONTROLS.json]]\n";
    return 2;
  }
  try
  {
    fssr::ssm::Model model(argv[1]);
    if (argc >= 6) model.set_sample_rate(std::stod(argv[5]));
    auto events = nlohmann::json::array();
    if (argc == 7)
    {
      std::ifstream file(argv[6]);
      file >> events;
      if (!events.is_array() || events.empty() || events[0].at("sample") != 0)
        throw std::runtime_error("control events must start at sample 0");
      for (std::size_t i = 0; i < events.size(); ++i)
      {
        if (!events[i].at("sample").is_number_unsigned() ||
            (i && events[i].at("sample") <= events[i - 1].at("sample")))
          throw std::runtime_error("control events must have increasing nonnegative sample indices");
        model.set_controls(events[i].at("controls").get<std::vector<float>>());
      }
    }
    const auto input = read_f32(argv[2]);
    const auto blocks = parse_blocks(argv[4]);
    write_f32(argv[3], process(model, input, blocks, events));
    // Always exercise reset; the companion file makes it observable to parity tests.
    write_f32(std::string(argv[3]) + ".reset.f32", process(model, input, blocks, events));
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
