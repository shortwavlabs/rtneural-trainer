#pragma once

#include <JuceHeader.h>
#include <RTNeural/RTNeural.h>

#include <array>
#include <atomic>
#include <memory>
#include <vector>

class RTNeuralLoaderAudioProcessor final : public juce::AudioProcessor
{
public:
    RTNeuralLoaderAudioProcessor();
    ~RTNeuralLoaderAudioProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    bool loadModelFromFile(const juce::File& file, juce::String& errorMessage);
    bool loadModelFromSelection(const juce::File& selection, juce::String& errorMessage);
    bool loadPedalFromFile(const juce::File& file, juce::String& errorMessage);
    bool loadPedalFromSelection(const juce::File& selection, juce::String& errorMessage);
    juce::String getModelName() const;
    juce::String getModelPath() const;
    juce::String getPackagePath() const;
    juce::String getPedalName() const;
    juce::String getPedalPath() const;
    juce::String getPedalStatus() const;
    bool loadImpulseResponseFromFile(const juce::File& file, juce::String& errorMessage);
    juce::String getImpulseResponseName() const;
    juce::String getImpulseResponsePath() const;
    juce::String getImpulseResponseStatus() const;
    juce::StringArray getModelInfoLines() const;
    juce::String getLoadStatus() const;
    juce::String getSafetyStatus() const;
    float consumeOutputPeak();
    bool hasLoadedModel() const;
    bool hasLoadedPedal() const;
    bool hasLoadedImpulseResponse() const;

    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

private:
    struct Biquad
    {
        void reset();
        void setIdentity();
        void setLowShelf(double sampleRate, double frequencyHz, float gainDb);
        void setHighShelf(double sampleRate, double frequencyHz, float gainDb);
        void setPeak(double sampleRate, double frequencyHz, double q, float gainDb);
        float process(float input);

        double b0 = 1.0;
        double b1 = 0.0;
        double b2 = 0.0;
        double a1 = 0.0;
        double a2 = 0.0;
        double x1 = 0.0;
        double x2 = 0.0;
        double y1 = 0.0;
        double y2 = 0.0;
    };

    struct ChannelEq
    {
        void reset();
        float process(float input);

        Biquad low;
        Biquad mid;
        Biquad high;
    };

    struct ModelMetadata
    {
        juce::String preset;
        juce::String architecture;
        juce::String projectName;
        juce::String runId;
        juce::String runDevice;
        juce::String validationStatus;
        juce::String aliasingStatus;
        juce::String benchmarkStatus;
        int sampleRate = 0;
        int latencySamples = 0;
        int receptiveFieldSamples = 0;
        int conv1dLayers = 0;
        int modelSizeBytes = 0;
        double esr = -1.0;
        double averageAsr = -1.0;
        double worstAsr = -1.0;
        double validationMaxError = -1.0;
        double nativeRealtimeFactor = -1.0;
    };

    struct ModelSet
    {
        std::array<std::unique_ptr<RTNeural::Model<float>>, 2> channels;
        juce::String name;
        juce::String path;
        juce::String packagePath;
        ModelMetadata metadata;
        int inputSize = 1;
    };

    struct ResolvedSelection
    {
        juce::File modelFile;
        juce::File packageDirectory;
    };

    static bool resolveModelSelection(const juce::File& selection,
                                      ResolvedSelection& resolved,
                                      juce::String& errorMessage);
    static std::unique_ptr<ModelSet> loadModelSetFromSelection(const juce::File& selection,
                                                               juce::String& errorMessage);
    static ModelMetadata readMetadata(const juce::File& modelFile,
                                      const juce::File& packageDirectory);
    static juce::String formatMetric(double value, int decimals);

    void applyRestoredModelPath();
    void applyRestoredPedalPath();
    void applyRestoredImpulseResponsePath();
    void updateEqCoefficientsIfNeeded();

    std::atomic<float>* inputGainDb = nullptr;
    std::atomic<float>* pedalOutputGainDb = nullptr;
    std::atomic<float>* outputGainDb = nullptr;
    std::atomic<float>* lowEqDb = nullptr;
    std::atomic<float>* midEqDb = nullptr;
    std::atomic<float>* highEqDb = nullptr;
    std::atomic<float>* bypass = nullptr;
    std::atomic<float>* pedalEnabled = nullptr;
    std::atomic<float>* irEnabled = nullptr;
    std::atomic<ModelSet*> currentPedal { nullptr };
    std::atomic<ModelSet*> currentModel { nullptr };
    std::atomic<double> hostSampleRate { 0.0 };
    std::atomic<float> outputPeak { 0.0f };
    std::atomic<bool> pedalLoaded { false };
    std::atomic<bool> impulseResponseLoaded { false };
    std::atomic<double> impulseResponseSeconds { 0.0 };

    juce::String loadedModelName { "No model loaded" };
    juce::String loadedModelPath;
    juce::String loadedPackagePath;
    juce::String loadedPedalName { "No pedal loaded" };
    juce::String loadedPedalPath;
    juce::String loadedPedalPackagePath;
    juce::String pedalStatus { "No pedal loaded" };
    juce::String loadedIrName { "No IR loaded" };
    juce::String loadedIrPath;
    juce::String irStatus { "No cabinet IR loaded" };
    juce::String loadStatus { "No model loaded - utility passthrough" };
    ModelMetadata loadedMetadata;
    ModelMetadata loadedPedalMetadata;
    std::vector<std::unique_ptr<ModelSet>> retainedModels;
    std::vector<std::unique_ptr<ModelSet>> retainedPedalModels;
    std::array<ChannelEq, 2> eqChannels;
    juce::dsp::Convolution cabinetConvolution;

    double eqSampleRate = 0.0;
    float cachedLowEqDb = 999.0f;
    float cachedMidEqDb = 999.0f;
    float cachedHighEqDb = 999.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RTNeuralLoaderAudioProcessor)
};
