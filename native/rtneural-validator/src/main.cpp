#include <RTNeural/RTNeural.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

    auto model = loadModel(modelPath, debug);
    model->reset();

    float input[1] {};
    float peakOutput = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for(std::size_t i = 0; i < frames; ++i)
    {
        input[0] = 0.0f;
        const auto value = model->forward(input);
        if(!std::isfinite(value))
            throw std::runtime_error("RTNeural model produced NaN or Inf output during benchmark");
        peakOutput = std::max(peakOutput, std::abs(value));
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    const auto elapsedSeconds = elapsedMs / 1000.0;
    const auto realtimeFactor = elapsedSeconds > 0.0 ? seconds / elapsedSeconds : 0.0;
    const auto status = realtimeFactor >= 1.0 ? "pass" : "fail";

    nlohmann::json report {
        { "schema_version", 1 },
        { "status", status },
        { "validator", "rtneural-validator" },
        { "model", modelPath },
        { "backend", "rtneural-stl" },
        { "sample_rate", sampleRate },
        { "seconds", seconds },
        { "frames_processed", frames },
        { "elapsed_ms", elapsedMs },
        { "realtime_factor", realtimeFactor },
        { "max_abs_output", peakOutput },
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
        << "  rtneural-validator benchmark --model model.rtneural.json --sample-rate 48000 --seconds 30 --report benchmark-report.json\n";
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
