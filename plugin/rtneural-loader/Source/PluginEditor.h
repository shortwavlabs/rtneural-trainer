#pragma once

#include <JuceHeader.h>

#include "PluginProcessor.h"

#include <array>

class RTNeuralLoaderAudioProcessorEditor final : public juce::AudioProcessorEditor,
                                                private juce::Timer
{
public:
    explicit RTNeuralLoaderAudioProcessorEditor(RTNeuralLoaderAudioProcessor& processor);
    ~RTNeuralLoaderAudioProcessorEditor() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    void timerCallback() override;
    void openModelChooser();
    void openPedalChooser();
    void openImpulseResponseChooser();
    void updateModelLabels();
    void updateControlEnablement();
    void configureGainSlider(juce::Slider& slider, const juce::String& name);
    void configureInfoLabel(juce::Label& label, float fontSize = 12.0f, bool bold = false);

    RTNeuralLoaderAudioProcessor& processorRef;

    juce::Label titleLabel;
    juce::Label subtitleLabel;
    juce::Slider inputSlider;
    juce::Slider pedalOutputSlider;
    juce::Slider outputSlider;
    juce::Slider lowSlider;
    juce::Slider midSlider;
    juce::Slider highSlider;
    juce::TextButton loadButton { "Load Export" };
    juce::TextButton loadPedalButton { "Load Pedal" };
    juce::TextButton loadIrButton { "Load IR" };
    juce::TextButton pedalButton { "Pedal On" };
    juce::TextButton irButton { "IR On" };
    juce::TextButton bypassButton { "Bypass" };
    juce::Label statusLabel;
    juce::Label pathLabel;
    juce::Label pedalLabel;
    juce::Label irLabel;
    juce::Label safetyLabel;
    juce::Label peakLabel;
    std::array<juce::Label, 10> infoLabels;
    std::unique_ptr<juce::FileChooser> fileChooser;

    juce::AudioProcessorValueTreeState::SliderAttachment inputAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment pedalOutputAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment outputAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment lowAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment midAttachment;
    juce::AudioProcessorValueTreeState::SliderAttachment highAttachment;
    juce::AudioProcessorValueTreeState::ButtonAttachment pedalAttachment;
    juce::AudioProcessorValueTreeState::ButtonAttachment irAttachment;
    juce::AudioProcessorValueTreeState::ButtonAttachment bypassAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RTNeuralLoaderAudioProcessorEditor)
};
