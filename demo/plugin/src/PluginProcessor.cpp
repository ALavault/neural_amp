#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "NAM/get_dsp.h"

namespace
{
juce::AudioProcessorValueTreeState::ParameterLayout makeLayout()
{
  using juce::NormalisableRange;
  std::vector<std::unique_ptr<juce::RangedAudioParameter>> parameters;
  parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
    juce::ParameterID{"input_db", 1}, "Input", NormalisableRange<float>{-24.0f, 24.0f}, 0.0f));
  parameters.push_back(std::make_unique<juce::AudioParameterFloat>(
    juce::ParameterID{"output_db", 1}, "Output", NormalisableRange<float>{-24.0f, 24.0f}, 0.0f));
  parameters.push_back(std::make_unique<juce::AudioParameterBool>(
    juce::ParameterID{"bypass", 1}, "Bypass", false));
  // 0 = plugin input, 1 = reference DI through the model, 2 = reference capture.
  parameters.push_back(std::make_unique<juce::AudioParameterChoice>(
    juce::ParameterID{"source", 1}, "Source",
    juce::StringArray{"Entree", "A: modele", "B: reel"}, 0));
  return {parameters.begin(), parameters.end()};
}
} // namespace

FssrAmpProcessor::FssrAmpProcessor()
  : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::mono(), true)
                           .withOutput("Output", juce::AudioChannelSet::mono(), true))
  , parameters(*this, nullptr, "state", makeLayout())
{
}

void FssrAmpProcessor::prepareToPlay(double sampleRate, int maximumBlockSize)
{
  currentSampleRate.store(sampleRate);
  currentBlockSize = maximumBlockSize;
  inputScratch.assign(static_cast<std::size_t>(maximumBlockSize), 0.0);
  outputScratch.assign(static_cast<std::size_t>(maximumBlockSize), 0.0);
  const juce::SpinLock::ScopedLockType lock(modelLock);
  if (model != nullptr)
    model->Reset(sampleRate, maximumBlockSize);
  referencePosition = 0;
}

bool FssrAmpProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
  // The model is mono; stereo is accepted so the standalone fills both speakers
  // (JUCE zeroes any device channel a mono processor does not write).
  const auto accepted = [](const juce::AudioChannelSet& set) {
    return set == juce::AudioChannelSet::mono() || set == juce::AudioChannelSet::stereo();
  };
  return accepted(layouts.getMainInputChannelSet())
         && accepted(layouts.getMainOutputChannelSet());
}

juce::String FssrAmpProcessor::loadModel(const juce::File& file)
{
  std::unique_ptr<nam::DSP> loaded;
  try
  {
    loaded = nam::get_dsp(std::filesystem::path(file.getFullPathName().toStdString()));
  }
  catch (const std::exception& error)
  {
    return juce::String("modele illisible: ") + error.what();
  }
  if (loaded == nullptr)
    return "modele illisible";
  if (loaded->NumInputChannels() != 1 || loaded->NumOutputChannels() != 1)
    return "seuls les modeles mono sont supportes";
  modelSampleRate.store(loaded->GetExpectedSampleRate());
  loaded->Reset(currentSampleRate.load(), currentBlockSize);
  {
    const juce::SpinLock::ScopedLockType lock(modelLock);
    model = std::move(loaded);
  }
  modelName = file.getFileNameWithoutExtension();
  return {};
}

juce::String FssrAmpProcessor::sampleRateWarning() const
{
  const double expected = modelSampleRate.load();
  if (expected <= 0.0 || std::abs(expected - currentSampleRate.load()) < 1.0)
    return {};
  return "ATTENTION hote a " + juce::String(currentSampleRate.load(), 0) + " Hz, modele attendu a "
         + juce::String(expected, 0) + " Hz";
}

juce::String FssrAmpProcessor::loadReference(const juce::File& dryFile, const juce::File& wetFile)
{
  juce::AudioFormatManager formats;
  formats.registerBasicFormats();
  juce::AudioBuffer<float> dry, wet;
  for (auto* pair : {&dryFile, &wetFile})
  {
    std::unique_ptr<juce::AudioFormatReader> reader(formats.createReaderFor(*pair));
    if (reader == nullptr)
      return "fichier audio illisible: " + pair->getFileName();
    auto& destination = (pair == &dryFile) ? dry : wet;
    destination.setSize(1, static_cast<int>(reader->lengthInSamples));
    reader->read(&destination, 0, destination.getNumSamples(), 0, true, false);
  }
  if (dry.getNumSamples() != wet.getNumSamples())
    return "les deux fichiers doivent avoir la meme duree";
  {
    const juce::SpinLock::ScopedLockType lock(modelLock);
    referenceDry = std::move(dry);
    referenceWet = std::move(wet);
    referencePosition = 0;
  }
  referenceName = dryFile.getFileNameWithoutExtension();
  referenceLoaded = true;
  return {};
}

void FssrAmpProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
  processMono(buffer, midi);
  const int samples = buffer.getNumSamples();
  for (int channel = 1; channel < buffer.getNumChannels(); ++channel)
    buffer.copyFrom(channel, 0, buffer, 0, 0, samples);
}

void FssrAmpProcessor::processMono(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
  juce::ScopedNoDenormals noDenormals;
  const int samples = buffer.getNumSamples();
  auto* channel = buffer.getWritePointer(0);
  const float inputGain =
    juce::Decibels::decibelsToGain(parameters.getRawParameterValue("input_db")->load());
  const float outputGain =
    juce::Decibels::decibelsToGain(parameters.getRawParameterValue("output_db")->load());
  const bool bypass = parameters.getRawParameterValue("bypass")->load() > 0.5f;
  const int source = static_cast<int>(parameters.getRawParameterValue("source")->load());

  const juce::SpinLock::ScopedTryLockType lock(modelLock);
  if (!lock.isLocked())
  {
    buffer.clear();
    return;
  }

  const bool useReference = source != 0 && referenceWet.getNumSamples() > 0;
  if (useReference)
  {
    // Stream the stored capture, looping, so A and B stay sample-aligned.
    const auto& stored = (source == 1) ? referenceDry : referenceWet;
    const int length = stored.getNumSamples();
    int position = referencePosition.load();
    for (int index = 0; index < samples; ++index)
    {
      channel[index] = stored.getSample(0, position);
      position = (position + 1) % length;
    }
    referencePosition.store(position);
    if (source == 2)
    {
      buffer.applyGain(outputGain);
      return; // the real capture bypasses the model
    }
  }

  if (bypass || model == nullptr)
  {
    buffer.applyGain(inputGain * outputGain);
    return;
  }

  if (static_cast<int>(inputScratch.size()) < samples)
  {
    buffer.applyGain(inputGain * outputGain);
    return; // larger block than prepared for; stay silent-safe rather than allocate
  }

  for (int index = 0; index < samples; ++index)
    inputScratch[static_cast<std::size_t>(index)] =
      static_cast<double>(channel[index]) * inputGain;
  double* inputPointers[] = {inputScratch.data()};
  double* outputPointers[] = {outputScratch.data()};
  model->process(inputPointers, outputPointers, samples);
  for (int index = 0; index < samples; ++index)
    channel[index] =
      static_cast<float>(outputScratch[static_cast<std::size_t>(index)]) * outputGain;
}

juce::AudioProcessorEditor* FssrAmpProcessor::createEditor()
{
  return new FssrAmpEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
  return new FssrAmpProcessor();
}
