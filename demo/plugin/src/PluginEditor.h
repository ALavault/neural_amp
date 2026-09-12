#pragma once

#include <juce_audio_utils/juce_audio_utils.h>

#include "PluginProcessor.h"

class FssrAmpEditor : public juce::AudioProcessorEditor
{
public:
  explicit FssrAmpEditor(FssrAmpProcessor&);
  void paint(juce::Graphics&) override;
  void resized() override;

private:
  using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
  using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
  using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

  void chooseModel();
  void chooseReference();

  FssrAmpProcessor& processor;
  juce::TextButton modelButton{"Charger un modele .nam"};
  juce::TextButton referenceButton{"Charger DI + capture reelle"};
  juce::Label status;
  juce::Slider input{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
  juce::Slider output{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
  juce::ToggleButton bypass{"Bypass"};
  juce::ComboBox source;
  std::unique_ptr<SliderAttachment> inputAttachment, outputAttachment;
  std::unique_ptr<ButtonAttachment> bypassAttachment;
  std::unique_ptr<ComboAttachment> sourceAttachment;
  std::unique_ptr<juce::FileChooser> chooser;
};
