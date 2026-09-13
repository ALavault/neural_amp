#include "PluginEditor.h"

FssrAmpEditor::FssrAmpEditor(FssrAmpProcessor& owner)
  : juce::AudioProcessorEditor(&owner)
  , processor(owner)
{
  addAndMakeVisible(modelButton);
  addAndMakeVisible(referenceButton);
  addAndMakeVisible(status);
  addAndMakeVisible(input);
  addAndMakeVisible(output);
  addAndMakeVisible(bypass);
  addAndMakeVisible(source);

  source.addItemList({"Entree", "A: modele", "B: reel"}, 1);
  addAndMakeVisible(inputLabel);
  addAndMakeVisible(outputLabel);
  status.setJustificationType(juce::Justification::centredLeft);
  status.setText("Aucun modele charge", juce::dontSendNotification);
  modelButton.onClick = [this] { chooseModel(); };
  referenceButton.onClick = [this] { chooseReference(); };

  auto& state = processor.parameters;
  inputAttachment = std::make_unique<SliderAttachment>(state, "input_db", input);
  outputAttachment = std::make_unique<SliderAttachment>(state, "output_db", output);
  bypassAttachment = std::make_unique<ButtonAttachment>(state, "bypass", bypass);
  sourceAttachment = std::make_unique<ComboAttachment>(state, "source", source);


  setSize(560, 340);
  startTimerHz(2);
}

void FssrAmpEditor::showStatus(const juce::String& text, bool isWarning)
{
  status.setColour(juce::Label::textColourId,
                   isWarning ? juce::Colours::red : juce::Colours::white);
  status.setText(text, juce::dontSendNotification);
}

void FssrAmpEditor::timerCallback()
{
  const auto warning = processor.sampleRateWarning();
  if (!warning.isEmpty())
    showStatus(warning, true);
  else if (processor.modelName.isNotEmpty() && status.getText().startsWith("ATTENTION"))
    showStatus("Modele : " + processor.modelName, false);
}

void FssrAmpEditor::chooseModel()
{
  chooser = std::make_unique<juce::FileChooser>("Modele NAM", juce::File{}, "*.nam");
  chooser->launchAsync(juce::FileBrowserComponent::openMode
                         | juce::FileBrowserComponent::canSelectFiles,
                       [this](const juce::FileChooser& result) {
                         const auto file = result.getResult();
                         if (file == juce::File{})
                           return;
                         const auto error = processor.loadModel(file);
                         if (!error.isEmpty())
                         {
                           status.setText(error, juce::dontSendNotification);
                           return;
                         }
                         const auto warning = processor.sampleRateWarning();
                         showStatus(warning.isEmpty() ? "Modele : " + processor.modelName
                                                      : warning,
                                    !warning.isEmpty());
                       });
}

void FssrAmpEditor::chooseReference()
{
  chooser = std::make_unique<juce::FileChooser>("DI puis capture reelle (2 fichiers)",
                                                juce::File{}, "*.wav");
  chooser->launchAsync(juce::FileBrowserComponent::openMode
                         | juce::FileBrowserComponent::canSelectFiles
                         | juce::FileBrowserComponent::canSelectMultipleItems,
                       [this](const juce::FileChooser& result) {
                         const auto files = result.getResults();
                         if (files.size() != 2)
                         {
                           status.setText("Selectionner exactement DI et capture",
                                          juce::dontSendNotification);
                           return;
                         }
                         // A multi-select dialog returns files unordered: the DI is the
                         // one whose name says so, otherwise the first by name.
                         auto dry = files[0], wet = files[1];
                         if (wet.getFileName().containsIgnoreCase("input")
                             || dry.getFileName().containsIgnoreCase("target"))
                           std::swap(dry, wet);
                         const auto error = processor.loadReference(dry, wet);
                         status.setText(error.isEmpty()
                                          ? "Reference : " + processor.referenceName
                                          : error,
                                        juce::dontSendNotification);
                       });
}

void FssrAmpEditor::paint(juce::Graphics& graphics)
{
  graphics.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
  graphics.setColour(juce::Colours::white);
  graphics.setFont(18.0f);
  graphics.drawFittedText("FSSR Amp Demo", getLocalBounds().removeFromTop(34),
                          juce::Justification::centred, 1);
}

void FssrAmpEditor::resized()
{
  auto area = getLocalBounds().reduced(12);
  area.removeFromTop(34);
  modelButton.setBounds(area.removeFromTop(30));
  area.removeFromTop(6);
  referenceButton.setBounds(area.removeFromTop(30));
  area.removeFromTop(6);
  status.setBounds(area.removeFromTop(26));
  area.removeFromTop(6);
  source.setBounds(area.removeFromTop(28));
  area.removeFromTop(10);
  const std::pair<juce::Label*, juce::Slider*> rows[] = {
    {&inputLabel, &input},
    {&outputLabel, &output},
  };
  for (const auto& [label, slider] : rows)
  {
    auto row = area.removeFromTop(30);
    label->setBounds(row.removeFromLeft(64));
    slider->setBounds(row);
    area.removeFromTop(6);
  }
  bypass.setBounds(area.removeFromTop(30));
}
