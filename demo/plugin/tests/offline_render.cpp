// Render a raw float32 file through the plugin's own processBlock, so the demo
// chain can be compared against the native runner used by the fact sheet.
// Usage: offline_render <model.nam> <in.f32> <out.f32> <block>

#include <cstdio>
#include <vector>

#include <juce_events/juce_events.h>

#include "../src/PluginProcessor.h"

int main(int argc, char** argv)
{
  if (argc != 5)
  {
    std::fprintf(stderr, "usage: offline_render <model.nam> <in.f32> <out.f32> <block>\n");
    return 2;
  }
  juce::ScopedJuceInitialiser_GUI juceInit;

  const int block = std::atoi(argv[4]);
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
  juce::AudioBuffer<float> buffer(1, block);
  juce::MidiBuffer midi;
  for (int start = 0; start < samples; start += block)
  {
    const int count = juce::jmin(block, samples - start);
    buffer.setSize(1, count, false, false, true);
    buffer.copyFrom(0, 0, input + start, count);
    processor.processBlock(buffer, midi);
    std::copy_n(buffer.getReadPointer(0), count, output.begin() + start);
  }

  const juce::File out{juce::String(argv[3])};
  out.deleteFile();
  return out.replaceWithData(output.data(), output.size() * sizeof(float)) ? 0 : 2;
}
