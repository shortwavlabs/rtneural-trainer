#include "PluginProcessor.h"

#include "PluginEditor.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace {

constexpr auto inputGainId = "input_gain_db";
constexpr auto outputGainId = "output_gain_db";
constexpr auto lowEqId = "eq_low_db";
constexpr auto midEqId = "eq_mid_db";
constexpr auto highEqId = "eq_high_db";
constexpr auto bypassId = "bypass";
constexpr auto irEnabledId = "ir_enabled";

float dbToGain(float db)
{
    return juce::Decibels::decibelsToGain(db);
}

juce::var parseJsonFile(const juce::File& file)
{
    if(! file.existsAsFile())
        return {};

    return juce::JSON::parse(file);
}

juce::var readPath(const juce::var& root, std::initializer_list<const char*> keys)
{
    auto current = root;

    for(auto* key : keys)
    {
        if(auto* object = current.getDynamicObject())
            current = object->getProperty(key);
        else
            return {};
    }

    return current;
}

juce::String readStringPath(const juce::var& root,
                            std::initializer_list<const char*> keys,
                            const juce::String& fallback = {})
{
    auto value = readPath(root, keys);
    return value.isVoid() || value.isUndefined() ? fallback : value.toString();
}

double readDoublePath(const juce::var& root,
                      std::initializer_list<const char*> keys,
                      double fallback = -1.0)
{
    auto value = readPath(root, keys);

    if(value.isDouble() || value.isInt() || value.isInt64() || value.isBool())
        return static_cast<double>(value);

    if(value.isString())
        return value.toString().getDoubleValue();

    return fallback;
}

int readIntPath(const juce::var& root,
                std::initializer_list<const char*> keys,
                int fallback = 0)
{
    const auto value = readDoublePath(root, keys, static_cast<double>(fallback));
    return static_cast<int>(std::round(value));
}

template <typename BiquadType>
void setShelfCoefficients(BiquadType& biquad,
                          double sampleRate,
                          double frequencyHz,
                          float gainDb,
                          bool highShelf)
{
    if(sampleRate <= 0.0 || std::abs(gainDb) < 0.001f)
    {
        biquad.setIdentity();
        return;
    }

    const auto a = std::pow(10.0, static_cast<double>(gainDb) / 40.0);
    const auto omega = juce::MathConstants<double>::twoPi * frequencyHz / sampleRate;
    const auto sinOmega = std::sin(omega);
    const auto cosOmega = std::cos(omega);
    const auto sqrtA = std::sqrt(a);
    const auto shelfSlope = 1.0;
    const auto alpha = sinOmega / 2.0 * std::sqrt((a + 1.0 / a) * (1.0 / shelfSlope - 1.0) + 2.0);

    double b0 {};
    double b1 {};
    double b2 {};
    double a0 {};
    double a1 {};
    double a2 {};

    if(highShelf)
    {
        b0 = a * ((a + 1.0) + (a - 1.0) * cosOmega + 2.0 * sqrtA * alpha);
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cosOmega);
        b2 = a * ((a + 1.0) + (a - 1.0) * cosOmega - 2.0 * sqrtA * alpha);
        a0 = (a + 1.0) - (a - 1.0) * cosOmega + 2.0 * sqrtA * alpha;
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cosOmega);
        a2 = (a + 1.0) - (a - 1.0) * cosOmega - 2.0 * sqrtA * alpha;
    }
    else
    {
        b0 = a * ((a + 1.0) - (a - 1.0) * cosOmega + 2.0 * sqrtA * alpha);
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cosOmega);
        b2 = a * ((a + 1.0) - (a - 1.0) * cosOmega - 2.0 * sqrtA * alpha);
        a0 = (a + 1.0) + (a - 1.0) * cosOmega + 2.0 * sqrtA * alpha;
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cosOmega);
        a2 = (a + 1.0) + (a - 1.0) * cosOmega - 2.0 * sqrtA * alpha;
    }

    biquad.b0 = b0 / a0;
    biquad.b1 = b1 / a0;
    biquad.b2 = b2 / a0;
    biquad.a1 = a1 / a0;
    biquad.a2 = a2 / a0;
}

} // namespace

void RTNeuralLoaderAudioProcessor::Biquad::reset()
{
    x1 = 0.0;
    x2 = 0.0;
    y1 = 0.0;
    y2 = 0.0;
}

void RTNeuralLoaderAudioProcessor::Biquad::setIdentity()
{
    b0 = 1.0;
    b1 = 0.0;
    b2 = 0.0;
    a1 = 0.0;
    a2 = 0.0;
}

void RTNeuralLoaderAudioProcessor::Biquad::setLowShelf(double sampleRate,
                                                       double frequencyHz,
                                                       float gainDb)
{
    setShelfCoefficients(*this, sampleRate, frequencyHz, gainDb, false);
}

void RTNeuralLoaderAudioProcessor::Biquad::setHighShelf(double sampleRate,
                                                        double frequencyHz,
                                                        float gainDb)
{
    setShelfCoefficients(*this, sampleRate, frequencyHz, gainDb, true);
}

void RTNeuralLoaderAudioProcessor::Biquad::setPeak(double sampleRate,
                                                   double frequencyHz,
                                                   double q,
                                                   float gainDb)
{
    if(sampleRate <= 0.0 || q <= 0.0 || std::abs(gainDb) < 0.001f)
    {
        setIdentity();
        return;
    }

    const auto a = std::pow(10.0, static_cast<double>(gainDb) / 40.0);
    const auto omega = juce::MathConstants<double>::twoPi * frequencyHz / sampleRate;
    const auto alpha = std::sin(omega) / (2.0 * q);
    const auto cosOmega = std::cos(omega);

    const auto rawB0 = 1.0 + alpha * a;
    const auto rawB1 = -2.0 * cosOmega;
    const auto rawB2 = 1.0 - alpha * a;
    const auto rawA0 = 1.0 + alpha / a;
    const auto rawA1 = -2.0 * cosOmega;
    const auto rawA2 = 1.0 - alpha / a;

    b0 = rawB0 / rawA0;
    b1 = rawB1 / rawA0;
    b2 = rawB2 / rawA0;
    a1 = rawA1 / rawA0;
    a2 = rawA2 / rawA0;
}

float RTNeuralLoaderAudioProcessor::Biquad::process(float input)
{
    const auto output = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

    x2 = x1;
    x1 = input;
    y2 = y1;
    y1 = output;

    if(! std::isfinite(output))
        return 0.0f;

    return static_cast<float>(output);
}

void RTNeuralLoaderAudioProcessor::ChannelEq::reset()
{
    low.reset();
    mid.reset();
    high.reset();
}

float RTNeuralLoaderAudioProcessor::ChannelEq::process(float input)
{
    auto output = low.process(input);
    output = mid.process(output);
    output = high.process(output);
    return output;
}

juce::AudioProcessorValueTreeState::ParameterLayout
RTNeuralLoaderAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { inputGainId, 1 },
        "Input",
        juce::NormalisableRange<float> { -24.0f, 24.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { outputGainId, 1 },
        "Output",
        juce::NormalisableRange<float> { -36.0f, 12.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { lowEqId, 1 },
        "Low",
        juce::NormalisableRange<float> { -12.0f, 12.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { midEqId, 1 },
        "Mid",
        juce::NormalisableRange<float> { -12.0f, 12.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { highEqId, 1 },
        "High",
        juce::NormalisableRange<float> { -12.0f, 12.0f, 0.01f },
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { bypassId, 1 },
        "Bypass",
        false));
    layout.add(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID { irEnabledId, 1 },
        "Cab IR",
        true));
    return layout;
}

RTNeuralLoaderAudioProcessor::RTNeuralLoaderAudioProcessor()
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    inputGainDb = apvts.getRawParameterValue(inputGainId);
    outputGainDb = apvts.getRawParameterValue(outputGainId);
    lowEqDb = apvts.getRawParameterValue(lowEqId);
    midEqDb = apvts.getRawParameterValue(midEqId);
    highEqDb = apvts.getRawParameterValue(highEqId);
    bypass = apvts.getRawParameterValue(bypassId);
    irEnabled = apvts.getRawParameterValue(irEnabledId);
    apvts.state.setProperty("modelPath", loadedModelPath, nullptr);
    apvts.state.setProperty("modelName", loadedModelName, nullptr);
    apvts.state.setProperty("packagePath", loadedPackagePath, nullptr);
    apvts.state.setProperty("irPath", loadedIrPath, nullptr);
    apvts.state.setProperty("irName", loadedIrName, nullptr);
}

void RTNeuralLoaderAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    hostSampleRate.store(sampleRate, std::memory_order_release);
    cabinetConvolution.prepare({ sampleRate,
                                 static_cast<juce::uint32>(std::max(1, samplesPerBlock)),
                                 static_cast<juce::uint32>(std::max(1, getTotalNumOutputChannels())) });
    cabinetConvolution.reset();

    eqSampleRate = sampleRate;
    cachedLowEqDb = 999.0f;
    cachedMidEqDb = 999.0f;
    cachedHighEqDb = 999.0f;
    updateEqCoefficientsIfNeeded();

    for(auto& channelEq : eqChannels)
        channelEq.reset();

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

    const auto modelBypassed = bypass != nullptr && bypass->load(std::memory_order_relaxed) > 0.5f;
    auto* model = modelBypassed ? nullptr : currentModel.load(std::memory_order_acquire);

    if(! modelBypassed)
    {
        updateEqCoefficientsIfNeeded();

        const auto inputGain = dbToGain(inputGainDb != nullptr
                                            ? inputGainDb->load(std::memory_order_relaxed)
                                            : 0.0f);
        buffer.applyGain(inputGain);
    }

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

    if(! modelBypassed)
    {
        for(auto channel = 0; channel < buffer.getNumChannels(); ++channel)
        {
            auto* channelData = buffer.getWritePointer(channel);
            auto& channelEq = eqChannels[static_cast<size_t>(std::min(channel, 1))];

            for(auto sample = 0; sample < buffer.getNumSamples(); ++sample)
                channelData[sample] = channelEq.process(channelData[sample]);
        }
    }

    const auto processIr = ! modelBypassed
        && impulseResponseLoaded.load(std::memory_order_acquire)
        && irEnabled != nullptr
        && irEnabled->load(std::memory_order_relaxed) > 0.5f;
    if(processIr && buffer.getNumSamples() > 0)
    {
        juce::dsp::AudioBlock<float> block(buffer);
        juce::dsp::ProcessContextReplacing<float> context(block);
        cabinetConvolution.process(context);
    }

    const auto gain = dbToGain(outputGainDb != nullptr
                                   ? outputGainDb->load(std::memory_order_relaxed)
                                   : 0.0f);
    buffer.applyGain(gain);

    float peak = 0.0f;
    for(auto channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        const auto* channelData = buffer.getReadPointer(channel);
        for(auto sample = 0; sample < buffer.getNumSamples(); ++sample)
            peak = std::max(peak, std::abs(channelData[sample]));
    }
    auto heldPeak = outputPeak.load(std::memory_order_relaxed);
    while(peak > heldPeak
          && ! outputPeak.compare_exchange_weak(heldPeak,
                                                peak,
                                                std::memory_order_release,
                                                std::memory_order_relaxed))
    {
    }
}

bool RTNeuralLoaderAudioProcessor::resolveModelSelection(const juce::File& selection,
                                                         ResolvedSelection& resolved,
                                                         juce::String& errorMessage)
{
    if(selection == juce::File())
    {
        errorMessage = "No model selected.";
        return false;
    }

    if(selection.isDirectory())
    {
        const auto modelFile = selection.getChildFile("model.rtneural.json");
        if(! modelFile.existsAsFile())
        {
            errorMessage = "Export folder does not contain model.rtneural.json.";
            return false;
        }

        resolved.modelFile = modelFile;
        resolved.packageDirectory = selection;
        return true;
    }

    if(! selection.existsAsFile())
    {
        errorMessage = "Model file does not exist.";
        return false;
    }

    resolved.modelFile = selection;
    const auto parent = selection.getParentDirectory();
    resolved.packageDirectory = parent.getChildFile("package.json").existsAsFile() ? parent : juce::File();
    return true;
}

RTNeuralLoaderAudioProcessor::ModelMetadata RTNeuralLoaderAudioProcessor::readMetadata(
    const juce::File& modelFile,
    const juce::File& packageDirectory)
{
    ModelMetadata metadata;

    const auto modelJson = parseJsonFile(modelFile);
    metadata.preset = readStringPath(modelJson, { "metadata", "architecture" });
    metadata.architecture = metadata.preset;
    metadata.sampleRate = readIntPath(modelJson, { "metadata", "sample_rate" });
    metadata.latencySamples = readIntPath(modelJson, { "metadata", "latency_samples" });
    metadata.esr = readDoublePath(modelJson, { "metadata", "loss", "stream_val_esr" },
                                  readDoublePath(modelJson, { "metadata", "loss", "esr" }));

    if(packageDirectory == juce::File())
        return metadata;

    const auto packageJson = parseJsonFile(packageDirectory.getChildFile("package.json"));
    if(! packageJson.isVoid() && ! packageJson.isUndefined())
    {
        metadata.projectName = readStringPath(packageJson, { "project", "name" });
        metadata.runId = readStringPath(packageJson, { "run", "id" });
        metadata.runDevice = readStringPath(packageJson, { "run", "device" });
        metadata.preset = readStringPath(packageJson, { "preset" }, metadata.preset);
        metadata.esr = readDoublePath(packageJson, { "quality", "esr" }, metadata.esr);
        metadata.sampleRate = readIntPath(packageJson, { "sample_rate" }, metadata.sampleRate);
    }

    const auto validationJson = parseJsonFile(packageDirectory.getChildFile("validation-report.json"));
    metadata.validationStatus = readStringPath(validationJson, { "status" });
    metadata.validationMaxError = readDoublePath(validationJson, { "max_abs_error" });

    const auto aliasingJson = parseJsonFile(packageDirectory.getChildFile("aliasing-report.json"));
    metadata.aliasingStatus = readStringPath(aliasingJson, { "status" });
    metadata.averageAsr = readDoublePath(aliasingJson, { "average_asr" });
    metadata.worstAsr = readDoublePath(aliasingJson, { "worst_asr" });

    const auto benchmarkJson = parseJsonFile(packageDirectory.getChildFile("benchmark-report.json"));
    metadata.benchmarkStatus = readStringPath(benchmarkJson, { "status" });
    metadata.nativeRealtimeFactor = readDoublePath(benchmarkJson, { "summary", "realtime_factor_worst" },
                                                  readDoublePath(benchmarkJson, { "realtime_factor" }));
    metadata.architecture = readStringPath(benchmarkJson,
                                           { "model_info", "architecture" },
                                           metadata.architecture);
    metadata.latencySamples = readIntPath(benchmarkJson,
                                          { "model_info", "latency_samples" },
                                          metadata.latencySamples);
    metadata.receptiveFieldSamples = readIntPath(benchmarkJson,
                                                 { "model_info", "receptive_field_samples" });
    metadata.conv1dLayers = readIntPath(benchmarkJson, { "model_info", "conv1d_layers" });
    metadata.modelSizeBytes = readIntPath(benchmarkJson, { "model_info", "size_bytes" });

    return metadata;
}

bool RTNeuralLoaderAudioProcessor::loadModelFromSelection(const juce::File& selection,
                                                          juce::String& errorMessage)
{
    ResolvedSelection resolved;
    if(! resolveModelSelection(selection, resolved, errorMessage))
        return false;

    return loadModelFromFile(resolved.modelFile, errorMessage);
}

bool RTNeuralLoaderAudioProcessor::loadModelFromFile(const juce::File& file,
                                                     juce::String& errorMessage)
{
    ResolvedSelection resolved;
    if(! resolveModelSelection(file, resolved, errorMessage))
        return false;

    try
    {
        std::ifstream stream(resolved.modelFile.getFullPathName().toStdString(), std::ios::binary);
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

        next->name = resolved.packageDirectory == juce::File()
                         ? resolved.modelFile.getFileName()
                         : resolved.packageDirectory.getFileName();
        next->path = resolved.modelFile.getFullPathName();
        next->packagePath = resolved.packageDirectory.getFullPathName();
        next->metadata = readMetadata(resolved.modelFile, resolved.packageDirectory);
        auto* raw = next.get();
        retainedModels.push_back(std::move(next));

        loadedModelName = raw->name;
        loadedModelPath = raw->path;
        loadedPackagePath = raw->packagePath;
        loadedMetadata = raw->metadata;
        loadStatus = "Loaded";
        apvts.state.setProperty("modelPath", loadedModelPath, nullptr);
        apvts.state.setProperty("modelName", loadedModelName, nullptr);
        apvts.state.setProperty("packagePath", loadedPackagePath, nullptr);
        currentModel.store(raw, std::memory_order_release);
        return true;
    }
    catch(const std::exception& e)
    {
        errorMessage = juce::String("Failed to load model: ") + e.what();
        loadStatus = errorMessage;
        return false;
    }
}

juce::String RTNeuralLoaderAudioProcessor::formatMetric(double value, int decimals)
{
    if(value < 0.0 || ! std::isfinite(value))
        return "n/a";

    return juce::String(value, decimals);
}

juce::String RTNeuralLoaderAudioProcessor::getModelName() const
{
    return loadedModelName;
}

juce::String RTNeuralLoaderAudioProcessor::getModelPath() const
{
    return loadedModelPath;
}

juce::String RTNeuralLoaderAudioProcessor::getPackagePath() const
{
    return loadedPackagePath;
}

bool RTNeuralLoaderAudioProcessor::loadImpulseResponseFromFile(const juce::File& file,
                                                               juce::String& errorMessage)
{
    if(file == juce::File())
    {
        errorMessage = "No impulse response selected.";
        return false;
    }

    if(! file.existsAsFile())
    {
        errorMessage = "Impulse response file does not exist.";
        return false;
    }

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if(reader == nullptr)
    {
        errorMessage = "Could not read this impulse response file.";
        irStatus = errorMessage;
        return false;
    }

    try
    {
        cabinetConvolution.loadImpulseResponse(file,
                                               juce::dsp::Convolution::Stereo::yes,
                                               juce::dsp::Convolution::Trim::yes,
                                               0,
                                               juce::dsp::Convolution::Normalise::yes);
        cabinetConvolution.reset();

        loadedIrName = file.getFileName();
        loadedIrPath = file.getFullPathName();
        impulseResponseSeconds.store(reader->sampleRate > 0.0
                                         ? static_cast<double>(reader->lengthInSamples) / reader->sampleRate
                                         : 0.0,
                                     std::memory_order_release);
        impulseResponseLoaded.store(true, std::memory_order_release);
        irStatus = "IR loaded";
        apvts.state.setProperty("irPath", loadedIrPath, nullptr);
        apvts.state.setProperty("irName", loadedIrName, nullptr);
        return true;
    }
    catch(const std::exception& e)
    {
        errorMessage = juce::String("Failed to load IR: ") + e.what();
        irStatus = errorMessage;
        return false;
    }
}

juce::String RTNeuralLoaderAudioProcessor::getImpulseResponseName() const
{
    return loadedIrName;
}

juce::String RTNeuralLoaderAudioProcessor::getImpulseResponsePath() const
{
    return loadedIrPath;
}

juce::String RTNeuralLoaderAudioProcessor::getImpulseResponseStatus() const
{
    return irStatus;
}

juce::StringArray RTNeuralLoaderAudioProcessor::getModelInfoLines() const
{
    juce::StringArray lines;

    if(! hasLoadedModel())
    {
        lines.add("No model loaded.");
        lines.add("Load an export folder or model.rtneural.json.");
        lines.add("Audio is passthrough with output trim.");
        return lines;
    }

    lines.add("Preset: " + (loadedMetadata.preset.isNotEmpty() ? loadedMetadata.preset : "unknown"));
    lines.add("ESR: " + formatMetric(loadedMetadata.esr, 4)
              + "  ASR worst/avg: " + formatMetric(loadedMetadata.worstAsr, 4)
              + " / " + formatMetric(loadedMetadata.averageAsr, 4));
    lines.add("Native RTF: " + formatMetric(loadedMetadata.nativeRealtimeFactor, 2)
              + "x  Validation: "
              + (loadedMetadata.validationStatus.isNotEmpty() ? loadedMetadata.validationStatus : "n/a"));
    lines.add("Sample rate: " + (loadedMetadata.sampleRate > 0 ? juce::String(loadedMetadata.sampleRate) : "n/a")
              + " Hz  Latency: " + juce::String(loadedMetadata.latencySamples) + " samples");

    if(loadedMetadata.receptiveFieldSamples > 0 || loadedMetadata.conv1dLayers > 0)
        lines.add("Receptive field: " + juce::String(loadedMetadata.receptiveFieldSamples)
                  + " samples  Conv1D layers: " + juce::String(loadedMetadata.conv1dLayers));

    lines.add("Cab IR: " + (hasLoadedImpulseResponse() ? loadedIrName : "none"));

    if(loadedMetadata.projectName.isNotEmpty())
        lines.add("Project: " + loadedMetadata.projectName);

    if(loadedMetadata.runId.isNotEmpty())
        lines.add("Run: " + loadedMetadata.runId);

    return lines;
}

juce::String RTNeuralLoaderAudioProcessor::getLoadStatus() const
{
    return loadStatus;
}

juce::String RTNeuralLoaderAudioProcessor::getSafetyStatus() const
{
    if(! hasLoadedModel())
        return "No model loaded.";

    juce::StringArray warnings;
    const auto hostRate = hostSampleRate.load(std::memory_order_acquire);

    if(hostRate > 0.0 && loadedMetadata.sampleRate > 0
       && std::abs(hostRate - static_cast<double>(loadedMetadata.sampleRate)) > 1.0)
    {
        warnings.add("Session is " + juce::String(hostRate, 0) + " Hz; model is "
                     + juce::String(loadedMetadata.sampleRate) + " Hz.");
    }

    if(loadedMetadata.aliasingStatus.isNotEmpty() && loadedMetadata.aliasingStatus != "pass")
        warnings.add("Aliasing report: " + loadedMetadata.aliasingStatus + ".");

    if(loadedMetadata.nativeRealtimeFactor > 0.0 && loadedMetadata.nativeRealtimeFactor < 2.0)
        warnings.add("Low native runtime headroom: "
                     + formatMetric(loadedMetadata.nativeRealtimeFactor, 2) + "x.");

    const auto irOn = irEnabled != nullptr && irEnabled->load(std::memory_order_relaxed) > 0.5f;
    if(irOn && ! hasLoadedImpulseResponse())
        warnings.add("Cab IR is enabled but no IR is loaded.");

    if(warnings.isEmpty())
        return "Model metadata looks usable for this session.";

    return warnings.joinIntoString(" ");
}

float RTNeuralLoaderAudioProcessor::consumeOutputPeak()
{
    return outputPeak.exchange(0.0f, std::memory_order_acq_rel);
}

bool RTNeuralLoaderAudioProcessor::hasLoadedModel() const
{
    return currentModel.load(std::memory_order_acquire) != nullptr;
}

bool RTNeuralLoaderAudioProcessor::hasLoadedImpulseResponse() const
{
    return impulseResponseLoaded.load(std::memory_order_acquire);
}

void RTNeuralLoaderAudioProcessor::applyRestoredModelPath()
{
    if(loadedModelPath.isEmpty())
        return;

    juce::String error;
    if(! loadModelFromFile(juce::File(loadedModelPath), error))
        loadStatus = "Could not restore model: " + error;
}

void RTNeuralLoaderAudioProcessor::applyRestoredImpulseResponsePath()
{
    if(loadedIrPath.isEmpty())
        return;

    juce::String error;
    if(! loadImpulseResponseFromFile(juce::File(loadedIrPath), error))
        irStatus = "Could not restore IR: " + error;
}

void RTNeuralLoaderAudioProcessor::updateEqCoefficientsIfNeeded()
{
    const auto lowDb = lowEqDb != nullptr ? lowEqDb->load(std::memory_order_relaxed) : 0.0f;
    const auto midDb = midEqDb != nullptr ? midEqDb->load(std::memory_order_relaxed) : 0.0f;
    const auto highDb = highEqDb != nullptr ? highEqDb->load(std::memory_order_relaxed) : 0.0f;

    if(eqSampleRate <= 0.0
       || (std::abs(lowDb - cachedLowEqDb) < 0.001f
           && std::abs(midDb - cachedMidEqDb) < 0.001f
           && std::abs(highDb - cachedHighEqDb) < 0.001f))
        return;

    cachedLowEqDb = lowDb;
    cachedMidEqDb = midDb;
    cachedHighEqDb = highDb;

    for(auto& channelEq : eqChannels)
    {
        channelEq.low.setLowShelf(eqSampleRate, 120.0, lowDb);
        channelEq.mid.setPeak(eqSampleRate, 750.0, 0.85, midDb);
        channelEq.high.setHighShelf(eqSampleRate, 4000.0, highDb);
    }
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
    return hasLoadedImpulseResponse() ? impulseResponseSeconds.load(std::memory_order_acquire) : 0.0;
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
    state.setProperty("packagePath", loadedPackagePath, nullptr);
    state.setProperty("irPath", loadedIrPath, nullptr);
    state.setProperty("irName", loadedIrName, nullptr);

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
            loadedPackagePath = restoredState.getProperty("packagePath", {}).toString();
            loadedIrPath = restoredState.getProperty("irPath", {}).toString();
            loadedIrName = restoredState.getProperty("irName", "No IR loaded").toString();
            apvts.replaceState(restoredState);
            applyRestoredModelPath();
            applyRestoredImpulseResponsePath();
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RTNeuralLoaderAudioProcessor();
}
