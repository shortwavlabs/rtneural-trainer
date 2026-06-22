#include <RTNeural/RTNeural.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef RTNEURAL_VALIDATOR_BACKEND
#define RTNEURAL_VALIDATOR_BACKEND "rtneural-unknown"
#endif

#ifndef RTNEURAL_VALIDATOR_BUILD_TYPE
#define RTNEURAL_VALIDATOR_BUILD_TYPE "unknown"
#endif

namespace {

struct WavBuffer
{
    std::vector<float> samples;
    int sampleRate = 0;
    int channels = 0;
};

struct ValidationResult
{
    std::string status;
    std::size_t sampleCount = 0;
    float maxAbsError = 0.0f;
    float rmse = 0.0f;
    float peakOutput = 0.0f;
};

std::map<std::string, std::string> parseArgs(int argc, char** argv, int start)
{
    std::map<std::string, std::string> args;
    for(int i = start; i < argc; ++i)
    {
        std::string key = argv[i];
        if(key.rfind("--", 0) == 0 && i + 1 < argc)
            args[key.substr(2)] = argv[++i];
    }
    return args;
}

const std::string& requireArg(const std::map<std::string, std::string>& args, const std::string& key)
{
    const auto it = args.find(key);
    if(it == args.end() || it->second.empty())
        throw std::runtime_error("--" + key + " is required");
    return it->second;
}

std::string nowTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    return std::to_string(seconds);
}

std::uint16_t readU16(std::istream& input)
{
    unsigned char bytes[2] {};
    input.read(reinterpret_cast<char*>(bytes), 2);
    if(!input)
        throw std::runtime_error("Unexpected end of file");
    return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t readU32(std::istream& input)
{
    unsigned char bytes[4] {};
    input.read(reinterpret_cast<char*>(bytes), 4);
    if(!input)
        throw std::runtime_error("Unexpected end of file");
    return static_cast<std::uint32_t>(
        bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
}

std::string readFourCC(std::istream& input)
{
    char id[4] {};
    input.read(id, 4);
    if(!input)
        throw std::runtime_error("Unexpected end of file");
    return std::string(id, 4);
}

void skipBytes(std::istream& input, std::uint32_t count)
{
    input.seekg(count, std::ios::cur);
    if(count % 2 == 1)
        input.seekg(1, std::ios::cur);
    if(!input)
        throw std::runtime_error("Unexpected end of file while skipping WAV chunk");
}

float clampSample(float value)
{
    return std::max(-1.0f, std::min(1.0f, value));
}

float decodePcmSample(const unsigned char* bytes, int bitsPerSample)
{
    if(bitsPerSample == 8)
        return (static_cast<int>(bytes[0]) - 128) / 128.0f;

    if(bitsPerSample == 16)
    {
        const auto value = static_cast<std::int16_t>(bytes[0] | (bytes[1] << 8));
        return clampSample(static_cast<float>(value) / 32768.0f);
    }

    if(bitsPerSample == 24)
    {
        std::int32_t value = static_cast<std::int32_t>(
            bytes[0] | (bytes[1] << 8) | (bytes[2] << 16));
        if((value & 0x00800000) != 0)
            value |= static_cast<std::int32_t>(0xFF000000);
        return clampSample(static_cast<float>(value) / 8388608.0f);
    }

    if(bitsPerSample == 32)
    {
        const auto value = static_cast<std::int32_t>(
            bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24));
        return clampSample(static_cast<float>(value) / 2147483648.0f);
    }

    throw std::runtime_error("Unsupported PCM bit depth: " + std::to_string(bitsPerSample));
}

float decodeFloatSample(const unsigned char* bytes, int bitsPerSample)
{
    if(bitsPerSample == 32)
    {
        float value = 0.0f;
        std::memcpy(&value, bytes, sizeof(float));
        return std::isfinite(value) ? clampSample(value) : 0.0f;
    }

    if(bitsPerSample == 64)
    {
        double value = 0.0;
        std::memcpy(&value, bytes, sizeof(double));
        return std::isfinite(value) ? clampSample(static_cast<float>(value)) : 0.0f;
    }

    throw std::runtime_error("Unsupported IEEE float bit depth: " + std::to_string(bitsPerSample));
}

WavBuffer readWavMono(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if(!input)
        throw std::runtime_error("Could not open WAV file: " + path);

    if(readFourCC(input) != "RIFF")
        throw std::runtime_error("WAV file is missing RIFF header: " + path);
    (void) readU32(input);
    if(readFourCC(input) != "WAVE")
        throw std::runtime_error("WAV file is missing WAVE header: " + path);

    std::uint16_t audioFormat = 0;
    std::uint16_t channels = 0;
    std::uint32_t sampleRate = 0;
    std::uint16_t blockAlign = 0;
    std::uint16_t bitsPerSample = 0;
    std::vector<unsigned char> data;

    while(input && !input.eof())
    {
        const auto chunkId = readFourCC(input);
        const auto chunkSize = readU32(input);

        if(chunkId == "fmt ")
        {
            audioFormat = readU16(input);
            channels = readU16(input);
            sampleRate = readU32(input);
            (void) readU32(input);
            blockAlign = readU16(input);
            bitsPerSample = readU16(input);
            const auto consumed = static_cast<std::uint32_t>(16);
            if(chunkSize > consumed)
                skipBytes(input, chunkSize - consumed);
            else if(chunkSize % 2 == 1)
                skipBytes(input, 0);
        }
        else if(chunkId == "data")
        {
            data.resize(chunkSize);
            input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if(!input)
                throw std::runtime_error("Unexpected end of WAV data: " + path);
            if(chunkSize % 2 == 1)
                skipBytes(input, 0);
        }
        else
        {
            skipBytes(input, chunkSize);
        }

        if(audioFormat != 0 && !data.empty())
            break;
    }

    if(audioFormat == 0 || channels == 0 || sampleRate == 0 || blockAlign == 0 || bitsPerSample == 0)
        throw std::runtime_error("WAV file is missing a usable fmt chunk: " + path);
    if(data.empty())
        throw std::runtime_error("WAV file is missing audio data: " + path);

    const int bytesPerSample = bitsPerSample / 8;
    if(bytesPerSample <= 0)
        throw std::runtime_error("Unsupported WAV bit depth: " + std::to_string(bitsPerSample));
    if(blockAlign < channels * bytesPerSample)
        throw std::runtime_error("Invalid WAV block alignment: " + path);

    WavBuffer buffer;
    buffer.sampleRate = static_cast<int>(sampleRate);
    buffer.channels = static_cast<int>(channels);

    for(std::size_t frameOffset = 0; frameOffset + blockAlign <= data.size(); frameOffset += blockAlign)
    {
        float mono = 0.0f;
        for(std::uint16_t ch = 0; ch < channels; ++ch)
        {
            const auto* sampleBytes = data.data() + frameOffset + ch * bytesPerSample;
            if(audioFormat == 1)
                mono += decodePcmSample(sampleBytes, bitsPerSample);
            else if(audioFormat == 3)
                mono += decodeFloatSample(sampleBytes, bitsPerSample);
            else
                throw std::runtime_error("Unsupported WAV format code: " + std::to_string(audioFormat));
        }
        buffer.samples.push_back(mono / static_cast<float>(channels));
    }

    return buffer;
}

std::unique_ptr<RTNeural::Model<float>> loadModel(const std::string& path, bool debug)
{
    std::ifstream jsonStream(path);
    if(!jsonStream)
        throw std::runtime_error("Could not open model JSON: " + path);

    auto model = RTNeural::json_parser::parseJson<float>(jsonStream, debug);
    if(!model)
        throw std::runtime_error("RTNeural could not parse model JSON: " + path);
    if(model->layers.empty())
        throw std::runtime_error("RTNeural model contains no layers: " + path);
    if(model->getInSize() != 1)
        throw std::runtime_error("Validator currently supports mono/single-input models only");
    if(model->getOutSize() != 1)
        throw std::runtime_error("Validator currently supports single-output models only");

    return model;
}

std::vector<float> runModel(RTNeural::Model<float>& model, const std::vector<float>& inputSamples)
{
    model.reset();
    std::vector<float> output;
    output.reserve(inputSamples.size());

    float input[1] {};
    for(const auto sample : inputSamples)
    {
        input[0] = sample;
        const auto value = model.forward(input);
        if(!std::isfinite(value))
            throw std::runtime_error("RTNeural model produced NaN or Inf output");
        output.push_back(value);
    }

    return output;
}

ValidationResult compareSignals(
    const std::vector<float>& output,
    const std::vector<float>& reference,
    float tolerance)
{
    const auto sampleCount = std::min(output.size(), reference.size());
    if(sampleCount == 0)
        throw std::runtime_error("No samples available for validation");

    double squaredError = 0.0;
    float maxAbsError = 0.0f;
    float peakOutput = 0.0f;
    for(std::size_t i = 0; i < sampleCount; ++i)
    {
        const auto error = std::abs(output[i] - reference[i]);
        maxAbsError = std::max(maxAbsError, error);
        peakOutput = std::max(peakOutput, std::abs(output[i]));
        squaredError += static_cast<double>(error) * static_cast<double>(error);
    }

    ValidationResult result;
    result.status = maxAbsError <= tolerance && output.size() == reference.size() ? "pass" : "fail";
    result.sampleCount = sampleCount;
    result.maxAbsError = maxAbsError;
    result.rmse = static_cast<float>(std::sqrt(squaredError / static_cast<double>(sampleCount)));
    result.peakOutput = peakOutput;
    return result;
}

std::vector<std::string> splitCsv(const std::string& value)
{
    std::vector<std::string> items;
    std::stringstream stream(value);
    std::string item;
    while(std::getline(stream, item, ','))
    {
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
            return std::isspace(ch) != 0;
        }), item.end());
        if(!item.empty())
            items.push_back(item);
    }
    return items;
}

std::vector<int> parsePositiveIntList(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    const std::vector<int>& defaults)
{
    const auto it = args.find(key);
    if(it == args.end() || it->second.empty())
        return defaults;

    std::vector<int> values;
    for(const auto& item : splitCsv(it->second))
    {
        const auto value = std::stoi(item);
        if(value <= 0)
            throw std::runtime_error("--" + key + " values must be positive");
        values.push_back(value);
    }
    return values.empty() ? defaults : values;
}

int parseBoundedInt(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    int defaultValue,
    int minValue,
    int maxValue)
{
    const auto it = args.find(key);
    const auto value = it == args.end() ? defaultValue : std::stoi(it->second);
    return std::max(minValue, std::min(maxValue, value));
}

double parseBoundedDouble(
    const std::map<std::string, std::string>& args,
    const std::string& key,
    double defaultValue,
    double minValue,
    double maxValue)
{
    const auto it = args.find(key);
    const auto value = it == args.end() ? defaultValue : std::stod(it->second);
    return std::max(minValue, std::min(maxValue, value));
}

double median(std::vector<double> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    if(values.size() % 2 == 1)
        return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

std::uintmax_t fileSizeBytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file)
        return 0;
    return static_cast<std::uintmax_t>(std::max<std::streamoff>(0, file.tellg()));
}

nlohmann::json readJsonFile(const std::string& path)
{
    std::ifstream file(path);
    if(!file)
        throw std::runtime_error("Could not open JSON file: " + path);
    nlohmann::json value;
    file >> value;
    return value;
}

int firstJsonInt(const nlohmann::json& value, const std::string& key, int defaultValue)
{
    if(!value.contains(key))
        return defaultValue;
    const auto& item = value.at(key);
    if(item.is_number_integer())
        return item.get<int>();
    if(item.is_array() && !item.empty() && item.at(0).is_number_integer())
        return item.at(0).get<int>();
    return defaultValue;
}

nlohmann::json modelInfo(const std::string& modelPath, int sampleRate)
{
    nlohmann::json info {
        { "size_bytes", fileSizeBytes(modelPath) },
        { "receptive_field_samples", nullptr },
        { "receptive_field_ms", nullptr },
        { "conv1d_layers", 0 },
        { "latency_samples", nullptr },
        { "latency_ms", nullptr },
        { "architecture", nullptr },
        { "schema", nullptr },
    };

    try
    {
        const auto modelJson = readJsonFile(modelPath);
        const auto metadata = modelJson.value("metadata", nlohmann::json::object());
        if(metadata.contains("latency_samples") && metadata["latency_samples"].is_number())
        {
            const auto latencySamples = metadata["latency_samples"].get<double>();
            info["latency_samples"] = latencySamples;
            if(sampleRate > 0)
                info["latency_ms"] = latencySamples * 1000.0 / sampleRate;
        }
        if(metadata.contains("architecture") && metadata["architecture"].is_string())
            info["architecture"] = metadata["architecture"];
        if(metadata.contains("schema") && metadata["schema"].is_string())
            info["schema"] = metadata["schema"];

        int receptiveField = 1;
        int conv1dLayers = 0;
        for(const auto& layer : modelJson.value("layers", nlohmann::json::array()))
        {
            if(layer.value("type", "") != "conv1d")
                continue;
            const auto kernelSize = firstJsonInt(layer, "kernel_size", 1);
            const auto dilation = firstJsonInt(layer, "dilation", 1);
            receptiveField += std::max(0, kernelSize - 1) * std::max(1, dilation);
            ++conv1dLayers;
        }
        info["conv1d_layers"] = conv1dLayers;
        if(conv1dLayers > 0)
        {
            info["receptive_field_samples"] = receptiveField;
            info["receptive_field_ms"] = sampleRate > 0
                ? static_cast<double>(receptiveField) * 1000.0 / sampleRate
                : 0.0;
        }
    }
    catch(const std::exception& e)
    {
        info["metadata_error"] = e.what();
    }

    return info;
}

void resetModels(std::vector<std::unique_ptr<RTNeural::Model<float>>>& models)
{
    for(auto& model : models)
        model->reset();
}

float runBenchmarkFrames(
    std::vector<std::unique_ptr<RTNeural::Model<float>>>& models,
    std::size_t frames,
    int blockSize)
{
    float input[1] {};
    float peakOutput = 0.0f;
    std::size_t processed = 0;
    while(processed < frames)
    {
        const auto blockFrames = std::min<std::size_t>(
            static_cast<std::size_t>(blockSize),
            frames - processed);
        for(std::size_t frame = 0; frame < blockFrames; ++frame)
        {
            input[0] = 0.0f;
            for(auto& model : models)
            {
                const auto value = model->forward(input);
                if(!std::isfinite(value))
                    throw std::runtime_error("RTNeural model produced NaN or Inf output during benchmark");
                peakOutput = std::max(peakOutput, std::abs(value));
            }
        }
        processed += blockFrames;
    }
    return peakOutput;
}

nlohmann::json benchmarkCase(
    const std::string& modelPath,
    bool debug,
    int blockSize,
    int channels,
    int passes,
    int warmupBlocks,
    std::size_t frames,
    double seconds,
    double minRealtimeFactor)
{
    std::vector<std::unique_ptr<RTNeural::Model<float>>> models;
    models.reserve(static_cast<std::size_t>(channels));
    for(int channel = 0; channel < channels; ++channel)
        models.push_back(loadModel(modelPath, debug));

    const auto warmupFrames = static_cast<std::size_t>(std::max(0, warmupBlocks))
        * static_cast<std::size_t>(blockSize);
    std::vector<double> elapsedMs;
    std::vector<double> realtimeFactors;
    float peakOutput = 0.0f;

    for(int pass = 0; pass < passes; ++pass)
    {
        resetModels(models);
        if(warmupFrames > 0)
            peakOutput = std::max(peakOutput, runBenchmarkFrames(models, warmupFrames, blockSize));

        const auto start = std::chrono::steady_clock::now();
        peakOutput = std::max(peakOutput, runBenchmarkFrames(models, frames, blockSize));
        const auto end = std::chrono::steady_clock::now();

        const auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        const auto elapsedSeconds = elapsed / 1000.0;
        elapsedMs.push_back(elapsed);
        realtimeFactors.push_back(elapsedSeconds > 0.0 ? seconds / elapsedSeconds : 0.0);
    }

    const auto minElapsed = *std::min_element(elapsedMs.begin(), elapsedMs.end());
    const auto maxElapsed = *std::max_element(elapsedMs.begin(), elapsedMs.end());
    const auto minRealtime = *std::min_element(realtimeFactors.begin(), realtimeFactors.end());
    const auto maxRealtime = *std::max_element(realtimeFactors.begin(), realtimeFactors.end());
    const auto medianElapsed = median(elapsedMs);
    const auto medianRealtime = median(realtimeFactors);

    return {
        { "status", minRealtime >= minRealtimeFactor ? "pass" : "fail" },
        { "block_size", blockSize },
        { "channels", channels },
        { "passes", passes },
        { "warmup_blocks", warmupBlocks },
        { "frames_per_pass", frames },
        { "model_evaluations_per_pass", frames * static_cast<std::size_t>(channels) },
        { "elapsed_ms_min", minElapsed },
        { "elapsed_ms_median", medianElapsed },
        { "elapsed_ms_worst", maxElapsed },
        { "realtime_factor_best", maxRealtime },
        { "realtime_factor_median", medianRealtime },
        { "realtime_factor_worst", minRealtime },
        { "max_abs_output", peakOutput },
    };
}

void writeJsonReport(const std::string& path, const nlohmann::json& report)
{
    std::ofstream file(path);
    if(!file)
        throw std::runtime_error("Failed to write report: " + path);
    file << report.dump(2) << '\n';
}

int validate(const std::map<std::string, std::string>& args)
{
    const auto& reportPath = requireArg(args, "report");
    const auto& modelPath = requireArg(args, "model");
    const auto& inputPath = requireArg(args, "input");
    const auto& referencePath = requireArg(args, "reference");
    const auto tolerance = args.count("tolerance") > 0 ? std::stof(args.at("tolerance")) : 1.0e-4f;
    const auto debug = args.count("debug") > 0;

    const auto input = readWavMono(inputPath);
    const auto reference = readWavMono(referencePath);
    if(input.sampleRate != reference.sampleRate)
        throw std::runtime_error("Input and reference sample rates differ");

    auto model = loadModel(modelPath, debug);
    const auto output = runModel(*model, input.samples);
    const auto comparison = compareSignals(output, reference.samples, tolerance);

    nlohmann::json report {
        { "schema_version", 1 },
        { "status", comparison.status },
        { "validator", "rtneural-validator" },
        { "model", modelPath },
        { "input", inputPath },
        { "reference", referencePath },
        { "sample_rate", input.sampleRate },
        { "input_channels", input.channels },
        { "reference_channels", reference.channels },
        { "sample_count", comparison.sampleCount },
        { "input_sample_count", input.samples.size() },
        { "reference_sample_count", reference.samples.size() },
        { "max_abs_error", comparison.maxAbsError },
        { "rmse", comparison.rmse },
        { "max_abs_output", comparison.peakOutput },
        { "tolerance", tolerance },
        { "timestamp", nowTimestamp() },
    };

    writeJsonReport(reportPath, report);
    std::cout << report.dump(2) << '\n';
    return comparison.status == "pass" ? 0 : 1;
}

int benchmark(const std::map<std::string, std::string>& args)
{
    const auto& reportPath = requireArg(args, "report");
    const auto& modelPath = requireArg(args, "model");
    const auto sampleRate = args.count("sample-rate") > 0 ? std::stoi(args.at("sample-rate")) : 48000;
    const auto seconds = args.count("seconds") > 0 ? std::stod(args.at("seconds")) : 30.0;
    const auto frames = static_cast<std::size_t>(std::max(1.0, seconds * static_cast<double>(sampleRate)));
    const auto debug = args.count("debug") > 0;
    const auto blockSizes = parsePositiveIntList(args, "block-sizes", { 64 });
    const auto channelCounts = parsePositiveIntList(args, "channels", { 1 });
    const auto passes = parseBoundedInt(args, "passes", 1, 1, 20);
    const auto warmupBlocks = parseBoundedInt(args, "warmup-blocks", 0, 0, 1000);
    const auto minRealtimeFactor = parseBoundedDouble(args, "min-realtime-factor", 1.0, 0.01, 1000.0);

    nlohmann::json runs = nlohmann::json::array();
    double worstRealtimeFactor = std::numeric_limits<double>::infinity();
    double bestRealtimeFactor = 0.0;
    double worstElapsedMs = 0.0;
    float peakOutput = 0.0f;
    std::size_t totalModelEvaluations = 0;
    std::size_t totalFramesProcessed = 0;
    nlohmann::json worstCase = nullptr;

    for(const auto blockSize : blockSizes)
    {
        for(const auto channels : channelCounts)
        {
            auto run = benchmarkCase(
                modelPath,
                debug,
                blockSize,
                channels,
                passes,
                warmupBlocks,
                frames,
                seconds,
                minRealtimeFactor);
            const auto runWorstRealtime = run.at("realtime_factor_worst").get<double>();
            const auto runBestRealtime = run.at("realtime_factor_best").get<double>();
            const auto runWorstElapsed = run.at("elapsed_ms_worst").get<double>();
            if(runWorstRealtime < worstRealtimeFactor)
            {
                worstRealtimeFactor = runWorstRealtime;
                worstElapsedMs = runWorstElapsed;
                worstCase = {
                    { "block_size", blockSize },
                    { "channels", channels },
                    { "realtime_factor", runWorstRealtime },
                    { "elapsed_ms", runWorstElapsed },
                };
            }
            bestRealtimeFactor = std::max(bestRealtimeFactor, runBestRealtime);
            peakOutput = std::max(peakOutput, run.at("max_abs_output").get<float>());
            totalFramesProcessed += frames * static_cast<std::size_t>(passes);
            totalModelEvaluations += frames
                * static_cast<std::size_t>(channels)
                * static_cast<std::size_t>(passes);
            runs.push_back(std::move(run));
        }
    }

    if(!std::isfinite(worstRealtimeFactor))
        worstRealtimeFactor = 0.0;
    const auto status = worstRealtimeFactor >= minRealtimeFactor ? "pass" : "fail";

    nlohmann::json report {
        { "schema_version", 2 },
        { "status", status },
        { "validator", "rtneural-validator" },
        { "model", modelPath },
        { "backend", RTNEURAL_VALIDATOR_BACKEND },
        { "build_type", RTNEURAL_VALIDATOR_BUILD_TYPE },
        { "sample_rate", sampleRate },
        { "seconds", seconds },
        { "seconds_per_pass", seconds },
        { "passes", passes },
        { "warmup_blocks", warmupBlocks },
        { "block_sizes", blockSizes },
        { "channels", channelCounts },
        { "frames_per_pass", frames },
        { "frames_processed", totalFramesProcessed },
        { "model_evaluations", totalModelEvaluations },
        { "elapsed_ms", worstElapsedMs },
        { "realtime_factor", worstRealtimeFactor },
        { "max_abs_output", peakOutput },
        { "thresholds", {
            { "min_realtime_factor", minRealtimeFactor },
        } },
        { "summary", {
            { "realtime_factor_worst", worstRealtimeFactor },
            { "realtime_factor_best", bestRealtimeFactor },
            { "worst_case", worstCase },
        } },
        { "model_info", modelInfo(modelPath, sampleRate) },
        { "runs", runs },
        { "timestamp", nowTimestamp() },
    };

    writeJsonReport(reportPath, report);
    std::cout << report.dump(2) << '\n';
    return std::string(status) == "pass" ? 0 : 1;
}

void printUsage()
{
    std::cerr
        << "Usage:\n"
        << "  rtneural-validator validate --model model.rtneural.json --input input.wav --reference ref.wav --report validation-report.json [--tolerance 0.0001]\n"
        << "  rtneural-validator benchmark --model model.rtneural.json --sample-rate 48000 --seconds 30 --report benchmark-report.json [--block-sizes 16,32,64,128,256,512] [--channels 1,2] [--passes 3] [--warmup-blocks 4]\n";
}

} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printUsage();
        return 2;
    }

    try
    {
        const std::string command = argv[1];
        const auto args = parseArgs(argc, argv, 2);

        if(command == "validate")
            return validate(args);
        if(command == "benchmark")
            return benchmark(args);

        printUsage();
        return 2;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
