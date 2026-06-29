#include "PluginEditor.h"

namespace {

constexpr auto inputGainId = "input_gain_db";
constexpr auto pedalOutputGainId = "pedal_output_gain_db";
constexpr auto outputGainId = "output_gain_db";
constexpr auto lowEqId = "eq_low_db";
constexpr auto midEqId = "eq_mid_db";
constexpr auto highEqId = "eq_high_db";
constexpr auto bypassId = "bypass";
constexpr auto pedalEnabledId = "pedal_enabled";
constexpr auto irEnabledId = "ir_enabled";

juce::String formatPeak(float peak)
{
    if(peak <= 0.000001f)
        return "Peak -inf dBFS";

    return "Peak " + juce::String(juce::Decibels::gainToDecibels(peak), 1) + " dBFS";
}

} // namespace

RTNeuralLoaderAudioProcessorEditor::RTNeuralLoaderAudioProcessorEditor(
    RTNeuralLoaderAudioProcessor& owner)
    : AudioProcessorEditor(&owner),
      processorRef(owner),
      inputAttachment(owner.apvts, inputGainId, inputSlider),
      pedalOutputAttachment(owner.apvts, pedalOutputGainId, pedalOutputSlider),
      outputAttachment(owner.apvts, outputGainId, outputSlider),
      lowAttachment(owner.apvts, lowEqId, lowSlider),
      midAttachment(owner.apvts, midEqId, midSlider),
      highAttachment(owner.apvts, highEqId, highSlider),
      pedalAttachment(owner.apvts, pedalEnabledId, pedalButton),
      irAttachment(owner.apvts, irEnabledId, irButton),
      bypassAttachment(owner.apvts, bypassId, bypassButton)
{
    titleLabel.setText("RTNeural Loader", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    titleLabel.setFont(juce::FontOptions(24.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    subtitleLabel.setText("Debug runtime for exported RTNeural package folders",
                          juce::dontSendNotification);
    subtitleLabel.setJustificationType(juce::Justification::centredLeft);
    subtitleLabel.setFont(juce::FontOptions(12.0f));
    subtitleLabel.setColour(juce::Label::textColourId, juce::Colour(0xffaeb9b4));
    addAndMakeVisible(subtitleLabel);

    configureGainSlider(inputSlider, "Input");
    configureGainSlider(pedalOutputSlider, "Pedal Out");
    configureGainSlider(outputSlider, "Output");
    configureGainSlider(lowSlider, "Low");
    configureGainSlider(midSlider, "Mid");
    configureGainSlider(highSlider, "High");

    loadButton.onClick = [this] { openModelChooser(); };
    addAndMakeVisible(loadButton);

    loadPedalButton.onClick = [this] { openPedalChooser(); };
    addAndMakeVisible(loadPedalButton);

    loadIrButton.onClick = [this] { openImpulseResponseChooser(); };
    addAndMakeVisible(loadIrButton);

    pedalButton.setClickingTogglesState(true);
    pedalButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff8ae78b));
    addAndMakeVisible(pedalButton);

    irButton.setClickingTogglesState(true);
    irButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff8ae78b));
    addAndMakeVisible(irButton);

    bypassButton.setClickingTogglesState(true);
    bypassButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff8ae78b));
    addAndMakeVisible(bypassButton);

    configureInfoLabel(statusLabel, 15.0f, true);
    configureInfoLabel(pathLabel, 11.0f);
    configureInfoLabel(pedalLabel, 11.0f);
    configureInfoLabel(irLabel, 11.0f);
    configureInfoLabel(safetyLabel, 12.0f, true);
    configureInfoLabel(peakLabel, 12.0f, true);

    pathLabel.setColour(juce::Label::textColourId, juce::Colour(0xffaeb9b4));
    pedalLabel.setColour(juce::Label::textColourId, juce::Colour(0xffaeb9b4));
    irLabel.setColour(juce::Label::textColourId, juce::Colour(0xffaeb9b4));
    safetyLabel.setColour(juce::Label::textColourId, juce::Colour(0xffffcf5f));
    peakLabel.setColour(juce::Label::textColourId, juce::Colour(0xff8ae78b));

    addAndMakeVisible(statusLabel);
    addAndMakeVisible(pathLabel);
    addAndMakeVisible(pedalLabel);
    addAndMakeVisible(irLabel);
    addAndMakeVisible(safetyLabel);
    addAndMakeVisible(peakLabel);

    for(auto& label : infoLabels)
    {
        configureInfoLabel(label, 12.0f);
        addAndMakeVisible(label);
    }

    updateModelLabels();
    startTimerHz(10);

    setResizable(false, false);
    setSize(840, 560);
}

void RTNeuralLoaderAudioProcessorEditor::configureGainSlider(juce::Slider& slider,
                                                            const juce::String& name)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 72, 20);
    slider.setName(name);
    slider.setTextValueSuffix(" dB");
    addAndMakeVisible(slider);
}

void RTNeuralLoaderAudioProcessorEditor::configureInfoLabel(juce::Label& label,
                                                           float fontSize,
                                                           bool bold)
{
    label.setJustificationType(juce::Justification::centredLeft);
    label.setFont(juce::FontOptions(fontSize, bold ? juce::Font::bold : juce::Font::plain));
    label.setColour(juce::Label::textColourId, juce::Colour(0xffe6ede8));
}

void RTNeuralLoaderAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff101513));

    auto bounds = getLocalBounds().reduced(18);
    g.setColour(juce::Colour(0xff26322e));
    g.drawRoundedRectangle(bounds.toFloat(), 8.0f, 1.0f);

    auto left = bounds.removeFromLeft(320).reduced(14);
    auto controls = left.withTrimmedTop(72);
    g.setColour(juce::Colour(0xff1b2420));
    g.fillRoundedRectangle(controls.toFloat(), 8.0f);

    auto right = bounds.reduced(14);
    g.setColour(juce::Colour(0xff131917));
    g.fillRoundedRectangle(right.toFloat(), 8.0f);
}

void RTNeuralLoaderAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(30);
    auto left = bounds.removeFromLeft(300);
    bounds.removeFromLeft(24);
    auto right = bounds;

    titleLabel.setBounds(left.removeFromTop(30));
    subtitleLabel.setBounds(left.removeFromTop(22));
    left.removeFromTop(18);

    auto firstRow = left.removeFromTop(106);
    inputSlider.setBounds(firstRow.removeFromLeft(96).reduced(6));
    pedalOutputSlider.setBounds(firstRow.removeFromLeft(96).reduced(6));
    outputSlider.setBounds(firstRow.removeFromLeft(96).reduced(6));

    auto eqRow = left.removeFromTop(106);
    lowSlider.setBounds(eqRow.removeFromLeft(96).reduced(6));
    midSlider.setBounds(eqRow.removeFromLeft(96).reduced(6));
    highSlider.setBounds(eqRow.removeFromLeft(96).reduced(6));

    left.removeFromTop(16);
    loadButton.setBounds(left.removeFromTop(40));
    left.removeFromTop(8);
    loadPedalButton.setBounds(left.removeFromTop(34));
    left.removeFromTop(8);
    loadIrButton.setBounds(left.removeFromTop(34));
    left.removeFromTop(8);
    auto toggles = left.removeFromTop(34);
    pedalButton.setBounds(toggles.removeFromLeft(90).reduced(0, 1));
    toggles.removeFromLeft(8);
    irButton.setBounds(toggles.removeFromLeft(90).reduced(0, 1));
    toggles.removeFromLeft(8);
    bypassButton.setBounds(toggles.reduced(0, 1));
    left.removeFromTop(8);
    peakLabel.setBounds(left.removeFromTop(24));

    statusLabel.setBounds(right.removeFromTop(28));
    pathLabel.setBounds(right.removeFromTop(42));
    pedalLabel.setBounds(right.removeFromTop(26));
    irLabel.setBounds(right.removeFromTop(26));
    safetyLabel.setBounds(right.removeFromTop(46));
    right.removeFromTop(8);

    for(auto& label : infoLabels)
        label.setBounds(right.removeFromTop(24));
}

void RTNeuralLoaderAudioProcessorEditor::timerCallback()
{
    const auto peak = processorRef.consumeOutputPeak();
    peakLabel.setText(formatPeak(peak), juce::dontSendNotification);
    peakLabel.setColour(juce::Label::textColourId,
                        peak >= 0.98f ? juce::Colour(0xffff726f) : juce::Colour(0xff8ae78b));
    safetyLabel.setText(processorRef.getSafetyStatus(), juce::dontSendNotification);
    irLabel.setText(processorRef.getImpulseResponseStatus() + ": " + processorRef.getImpulseResponseName(),
                    juce::dontSendNotification);
    pedalLabel.setText(processorRef.getPedalStatus() + ": " + processorRef.getPedalName(),
                       juce::dontSendNotification);
    updateControlEnablement();
}

void RTNeuralLoaderAudioProcessorEditor::openModelChooser()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Load RTNeural export folder or model JSON",
        juce::File(),
        "*.json;*.rtneural.json");

    constexpr auto flags = juce::FileBrowserComponent::openMode
        | juce::FileBrowserComponent::canSelectFiles
        | juce::FileBrowserComponent::canSelectDirectories;

    fileChooser->launchAsync(flags, [this](const juce::FileChooser& chooser) {
        const auto selection = chooser.getResult();
        if(selection == juce::File())
            return;

        juce::String error;
        if(! processorRef.loadModelFromSelection(selection, error))
            statusLabel.setText(error, juce::dontSendNotification);

        updateModelLabels();
    });
}

void RTNeuralLoaderAudioProcessorEditor::openPedalChooser()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Load RTNeural pedal export folder or model JSON",
        juce::File(),
        "*.json;*.rtneural.json");

    constexpr auto flags = juce::FileBrowserComponent::openMode
        | juce::FileBrowserComponent::canSelectFiles
        | juce::FileBrowserComponent::canSelectDirectories;

    fileChooser->launchAsync(flags, [this](const juce::FileChooser& chooser) {
        const auto selection = chooser.getResult();
        if(selection == juce::File())
            return;

        juce::String error;
        if(! processorRef.loadPedalFromSelection(selection, error))
            pedalLabel.setText(error, juce::dontSendNotification);

        updateModelLabels();
    });
}

void RTNeuralLoaderAudioProcessorEditor::openImpulseResponseChooser()
{
    fileChooser = std::make_unique<juce::FileChooser>(
        "Load cabinet impulse response",
        juce::File(),
        "*.wav;*.aif;*.aiff;*.flac");

    constexpr auto flags = juce::FileBrowserComponent::openMode
        | juce::FileBrowserComponent::canSelectFiles;

    fileChooser->launchAsync(flags, [this](const juce::FileChooser& chooser) {
        const auto selection = chooser.getResult();
        if(selection == juce::File())
            return;

        juce::String error;
        if(! processorRef.loadImpulseResponseFromFile(selection, error))
            irLabel.setText(error, juce::dontSendNotification);

        updateModelLabels();
    });
}

void RTNeuralLoaderAudioProcessorEditor::updateModelLabels()
{
    statusLabel.setText(processorRef.getLoadStatus() + ": " + processorRef.getModelName(),
                        juce::dontSendNotification);

    const auto packagePath = processorRef.getPackagePath();
    const auto modelPath = processorRef.getModelPath();
    const auto path = packagePath.isNotEmpty() ? packagePath : modelPath;
    pathLabel.setText(path.isEmpty() ? "Choose an export folder or model.rtneural.json."
                                    : path,
                      juce::dontSendNotification);

    const auto irPath = processorRef.getImpulseResponsePath();
    irLabel.setText(irPath.isEmpty()
                        ? processorRef.getImpulseResponseStatus() + ": choose a cab IR WAV/AIFF."
                        : processorRef.getImpulseResponseStatus() + ": " + processorRef.getImpulseResponseName(),
                    juce::dontSendNotification);

    const auto pedalPath = processorRef.getPedalPath();
    pedalLabel.setText(pedalPath.isEmpty()
                           ? processorRef.getPedalStatus() + ": choose a pedal export folder."
                           : processorRef.getPedalStatus() + ": " + processorRef.getPedalName(),
                       juce::dontSendNotification);

    safetyLabel.setText(processorRef.getSafetyStatus(), juce::dontSendNotification);
    updateControlEnablement();

    const auto lines = processorRef.getModelInfoLines();
    for(size_t i = 0; i < infoLabels.size(); ++i)
    {
        infoLabels[i].setText(static_cast<int>(i) < lines.size() ? lines[static_cast<int>(i)] : juce::String(),
                              juce::dontSendNotification);
    }
}

void RTNeuralLoaderAudioProcessorEditor::updateControlEnablement()
{
    const auto hasPedal = processorRef.hasLoadedPedal();
    pedalButton.setEnabled(hasPedal);
    pedalOutputSlider.setEnabled(hasPedal && pedalButton.getToggleState());
}
