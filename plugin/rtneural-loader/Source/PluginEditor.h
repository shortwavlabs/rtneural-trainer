#pragma once

#include <JuceHeader.h>

#include "PluginProcessor.h"

class RTNeuralLoaderAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit RTNeuralLoaderAudioProcessorEditor(RTNeuralLoaderAudioProcessor& processor);
    ~RTNeuralLoaderAudioProcessorEditor() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void openModelChooser();
    void updateModelLabels(const juce::String& status);

    RTNeuralLoaderAudioProcessor& processorRef;

    juce::Label titleLabel;
    juce::Slider outputSlider;
    juce::TextButton loadButton { "Load Model" };
    juce::Label statusLabel;
    juce::Label pathLabel;
    std::unique_ptr<juce::FileChooser> fileChooser;

    juce::AudioProcessorValueTreeState::SliderAttachment outputAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RTNeuralLoaderAudioProcessorEditor)
};

