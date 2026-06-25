#include "PluginEditor.h"

namespace {

constexpr auto outputGainId = "output_gain_db";

} // namespace

RTNeuralLoaderAudioProcessorEditor::RTNeuralLoaderAudioProcessorEditor(
    RTNeuralLoaderAudioProcessor& owner)
    : AudioProcessorEditor(&owner),
      processorRef(owner),
      outputAttachment(owner.apvts, outputGainId, outputSlider)
{
    titleLabel.setText("RTNeural Loader", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    titleLabel.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    outputSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    outputSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 84, 22);
    outputSlider.setName("Output");
    addAndMakeVisible(outputSlider);

    loadButton.onClick = [this] { openModelChooser(); };
    addAndMakeVisible(loadButton);

    statusLabel.setJustificationType(juce::Justification::centredLeft);
    statusLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(statusLabel);

    pathLabel.setJustificationType(juce::Justification::centredLeft);
    pathLabel.setFont(juce::FontOptions(12.0f));
    pathLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(pathLabel);

    updateModelLabels(processorRef.hasLoadedModel() ? "Loaded" : "No model loaded - volume-only passthrough");

    setResizable(false, false);
    setSize(430, 220);
}

void RTNeuralLoaderAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff101513));

    auto bounds = getLocalBounds().reduced(18);
    g.setColour(juce::Colour(0xff26322e));
    g.drawRoundedRectangle(bounds.toFloat(), 8.0f, 1.0f);
}

void RTNeuralLoaderAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(24);
    titleLabel.setBounds(bounds.removeFromTop(34));

    auto controls = bounds.removeFromTop(118);
    outputSlider.setBounds(controls.removeFromLeft(150).reduced(8));

    auto right = controls.reduced(8, 12);
    loadButton.setBounds(right.removeFromTop(38));
    right.removeFromTop(10);
    statusLabel.setBounds(right.removeFromTop(24));
    pathLabel.setBounds(right.removeFromTop(42));
}

void RTNeuralLoaderAudioProcessorEditor::openModelChooser()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Load RTNeural JSON model",
        juce::File(),
        "*.json;*.rtneural.json");

    constexpr auto flags = juce::FileBrowserComponent::openMode
        | juce::FileBrowserComponent::canSelectFiles;

    fileChooser->launchAsync(flags, [this](const juce::FileChooser& chooser) {
        const auto file = chooser.getResult();
        if(file == juce::File())
            return;

        juce::String error;
        if(processorRef.loadModelFromFile(file, error))
            updateModelLabels("Loaded");
        else
            updateModelLabels(error);
    });
}

void RTNeuralLoaderAudioProcessorEditor::updateModelLabels(const juce::String& status)
{
    statusLabel.setText(status + ": " + processorRef.getModelName(), juce::dontSendNotification);

    const auto path = processorRef.getModelPath();
    pathLabel.setText(path.isEmpty() ? "Choose an exported model.rtneural.json file."
                                    : path,
                      juce::dontSendNotification);
}
