#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <algorithm>
#include <fstream>

namespace {

constexpr auto outputGainId = "output_gain_db";

float dbToGain(float db)
{
    return juce::Decibels::decibelsToGain(db);
}

} // namespace

juce::AudioProcessorValueTreeState::ParameterLayout
RTNeuralLoaderAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { outputGainId, 1 },
        "Output",
        juce::NormalisableRange<float> { -36.0f, 12.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    return layout;
}

RTNeuralLoaderAudioProcessor::RTNeuralLoaderAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    outputGainDb = apvts.getRawParameterValue(outputGainId);
    apvts.state.setProperty("modelPath", loadedModelPath, nullptr);
    apvts.state.setProperty("modelName", loadedModelName, nullptr);
}

void RTNeuralLoaderAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate, samplesPerBlock);

    if(auto* model = currentModel.load(std::memory_order_acquire))
    {
        for(auto& channelModel : model->channels)
        {
            if(channelModel != nullptr)
                channelModel->reset();
        }
    }
}

void RTNeuralLoaderAudioProcessor::releaseResources() {}

bool RTNeuralLoaderAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& input = layouts.getMainInputChannelSet();
    const auto& output = layouts.getMainOutputChannelSet();

    if(input != output)
        return false;

    return output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo();
}

void RTNeuralLoaderAudioProcessor::processBlock(
    juce::AudioBuffer<float>& buffer,
    juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    juce::ignoreUnused(midiMessages);

    const auto totalInputChannels = getTotalNumInputChannels();
    const auto totalOutputChannels = getTotalNumOutputChannels();

    for(auto channel = totalInputChannels; channel < totalOutputChannels; ++channel)
        buffer.clear(channel, 0, buffer.getNumSamples());

    auto* model = currentModel.load(std::memory_order_acquire);

    if(model != nullptr)
    {
        float input[1] {};

        for(auto channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            auto* channelData = buffer.getWritePointer(channel);
            auto* channelModel = model->channels[static_cast<size_t>(std::min(channel, 1))].get();

            if(channelModel == nullptr)
                continue;

            for(auto sample = 0; sample < buffer.getNumSamples(); ++sample)
            {
                input[0] = channelData[sample];
                channelData[sample] = channelModel->forward(input);
            }
        }
    }

    const auto gain = dbToGain(outputGainDb != nullptr ? outputGainDb->load(std::memory_order_relaxed) : 0.0f);
    buffer.applyGain(gain);
}

bool RTNeuralLoaderAudioProcessor::loadModelFromFile(const juce::File& file, juce::String& errorMessage)
{
    if(! file.existsAsFile())
    {
        errorMessage = "Model file does not exist.";
        return false;
    }

    try
    {
        std::ifstream stream(file.getFullPathName().toStdString(), std::ios::binary);
        if(! stream.good())
        {
            errorMessage = "Could not open model file.";
            return false;
        }

        auto next = std::make_unique<ModelSet>();
        next->channels[0] = RTNeural::json_parser::parseJson<float>(stream);

        stream.clear();
        stream.seekg(0, std::ios::beg);
        next->channels[1] = RTNeural::json_parser::parseJson<float>(stream);

        if(next->channels[0] == nullptr || next->channels[1] == nullptr)
        {
            errorMessage = "RTNeural could not parse this model.";
            return false;
        }

        next->channels[0]->reset();
        next->channels[1]->reset();
        next->inputSize = next->channels[0]->layers.empty() ? 1 : next->channels[0]->layers[0]->in_size;

        if(next->inputSize != 1)
        {
            errorMessage = "This test plugin only supports mono-input RTNeural models.";
            return false;
        }

        next->name = file.getFileName();
        next->path = file.getFullPathName();
        auto* raw = next.get();
        retainedModels.push_back(std::move(next));

        loadedModelName = raw->name;
        loadedModelPath = raw->path;
        apvts.state.setProperty("modelPath", loadedModelPath, nullptr);
        apvts.state.setProperty("modelName", loadedModelName, nullptr);
        currentModel.store(raw, std::memory_order_release);
        return true;
    }
    catch(const std::exception& e)
    {
        errorMessage = juce::String("Failed to load model: ") + e.what();
        return false;
    }
}

juce::String RTNeuralLoaderAudioProcessor::getModelName() const
{
    return loadedModelName;
}

juce::String RTNeuralLoaderAudioProcessor::getModelPath() const
{
    return loadedModelPath;
}

bool RTNeuralLoaderAudioProcessor::hasLoadedModel() const
{
    return currentModel.load(std::memory_order_acquire) != nullptr;
}

juce::AudioProcessorEditor* RTNeuralLoaderAudioProcessor::createEditor()
{
    return new RTNeuralLoaderAudioProcessorEditor(*this);
}

bool RTNeuralLoaderAudioProcessor::hasEditor() const
{
    return true;
}

const juce::String RTNeuralLoaderAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool RTNeuralLoaderAudioProcessor::acceptsMidi() const
{
    return false;
}

bool RTNeuralLoaderAudioProcessor::producesMidi() const
{
    return false;
}

bool RTNeuralLoaderAudioProcessor::isMidiEffect() const
{
    return false;
}

double RTNeuralLoaderAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int RTNeuralLoaderAudioProcessor::getNumPrograms()
{
    return 1;
}

int RTNeuralLoaderAudioProcessor::getCurrentProgram()
{
    return 0;
}

void RTNeuralLoaderAudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String RTNeuralLoaderAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void RTNeuralLoaderAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void RTNeuralLoaderAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    state.setProperty("modelPath", loadedModelPath, nullptr);
    state.setProperty("modelName", loadedModelName, nullptr);

    if(auto xml = state.createXml())
        copyXmlToBinary(*xml, destData);
}

void RTNeuralLoaderAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if(auto xmlState = getXmlFromBinary(data, sizeInBytes))
    {
        if(xmlState->hasTagName(apvts.state.getType()))
        {
            auto restoredState = juce::ValueTree::fromXml(*xmlState);
            loadedModelPath = restoredState.getProperty("modelPath", {}).toString();
            loadedModelName = restoredState.getProperty("modelName", "No model loaded").toString();
            apvts.replaceState(restoredState);
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RTNeuralLoaderAudioProcessor();
}
