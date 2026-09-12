// Render a raw float32 file through the plugin's own processBlock, so the demo
// chain can be compared against the native runner used by the fact sheet.
// Usage: offline_render <model.nam> <in.f32> <out.f32> <block> [channels]
// With 2 channels it also asserts both output channels are identical, which is
// what the standalone needs to fill both speakers from a mono model.

#include <cstdio>
#include <cstring>
#include <vector>

#include <juce_events/juce_events.h>

#include "../src/PluginProcessor.h"

int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6)
  {
    std::fprintf(stderr,
                 "usage: offline_render <model.nam> <in.f32> <out.f32> <block> [channels]\n");
    return 2;
  }
  juce::ScopedJuceInitialiser_GUI juceInit;

  const int block = std::atoi(argv[4]);
  const int channels = (argc == 6) ? std::atoi(argv[5]) : 1;
  juce::MemoryBlock raw;
  if (!juce::File(juce::String(argv[2])).loadFileAsData(raw))
  {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 2;
  }
  const int samples = static_cast<int>(raw.getSize() / sizeof(float));
  const auto* input = static_cast<const float*>(raw.getData());

  FssrAmpProcessor processor;
  processor.prepareToPlay(48000.0, block);
  const auto error = processor.loadModel(juce::File(juce::String(argv[1])));
  if (!error.isEmpty())
  {
    std::fprintf(stderr, "%s\n", error.toRawUTF8());
    return 2;
  }

  std::vector<float> output(static_cast<std::size_t>(samples), 0.0f);
  juce::AudioBuffer<float> buffer(channels, block);
  juce::MidiBuffer midi;
  for (int start = 0; start < samples; start += block)
  {
    const int count = juce::jmin(block, samples - start);
    buffer.setSize(channels, count, false, false, true);
    for (int channel = 0; channel < channels; ++channel)
      buffer.copyFrom(channel, 0, input + start, count);
    processor.processBlock(buffer, midi);
    for (int channel = 1; channel < channels; ++channel)
      if (std::memcmp(buffer.getReadPointer(0), buffer.getReadPointer(channel),
                      static_cast<std::size_t>(count) * sizeof(float))
          != 0)
      {
        std::fprintf(stderr, "channel %d differs from channel 0\n", channel);
        return 1;
      }
    std::copy_n(buffer.getReadPointer(0), count, output.begin() + start);
  }

  const juce::File out{juce::String(argv[3])};
  out.deleteFile();
  return out.replaceWithData(output.data(), output.size() * sizeof(float)) ? 0 : 2;
}
