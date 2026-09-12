#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <juce_audio_processors/juce_audio_processors.h>

#include "NAM/dsp.h"

/// Demo amp: one NAM model, gain staging, and A/B against a reference capture.
class FssrAmpProcessor : public juce::AudioProcessor
{
public:
  FssrAmpProcessor();
  ~FssrAmpProcessor() override = default;

  void prepareToPlay(double sampleRate, int maximumBlockSize) override;
  void releaseResources() override {}
  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
  void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override;

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override { return true; }
  const juce::String getName() const override { return "FSSR Amp Demo"; }
  bool acceptsMidi() const override { return false; }
  bool producesMidi() const override { return false; }
  double getTailLengthSeconds() const override { return 0.0; }
  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String&) override {}
  void getStateInformation(juce::MemoryBlock&) override {}
  void setStateInformation(const void*, int) override {}

  /// Load a `.nam` model. Returns an error message, or an empty string on success.
  juce::String loadModel(const juce::File& file);
  /// Load a DI/target pair for blind A/B. Returns an error message, or empty.
  juce::String loadReference(const juce::File& dryFile, const juce::File& wetFile);

  juce::AudioProcessorValueTreeState parameters;
  juce::String modelName;
  juce::String referenceName;
  /// Non-empty when the host rate differs from the model's: NAM core does not resample.
  juce::String sampleRateWarning() const;
  std::atomic<bool> referenceLoaded{false};

private:
  /// The model is mono: channel 0 is processed, then copied to the others.
  void processMono(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&);

  /// Audio-thread state, swapped in from the message thread under a spin lock.
  juce::SpinLock modelLock;
  std::unique_ptr<nam::DSP> model;
  std::vector<double> inputScratch, outputScratch;
  juce::AudioBuffer<float> referenceDry, referenceWet;
  std::atomic<int> referencePosition{0};
  std::atomic<double> modelSampleRate{0.0};
  std::atomic<double> currentSampleRate{48000.0};
  int currentBlockSize{512};
};
