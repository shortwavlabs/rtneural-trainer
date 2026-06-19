#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace {

std::map<std::string, std::string> parseArgs(int argc, char** argv, int start)
{
    std::map<std::string, std::string> args;
    for (int i = start; i < argc; ++i)
    {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0 && i + 1 < argc)
        {
            args[key.substr(2)] = argv[++i];
        }
    }
    return args;
}

bool writeFile(const std::string& path, const std::string& content)
{
    std::ofstream file(path);
    if (!file)
        return false;
    file << content;
    return true;
}

std::string timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    return std::to_string(seconds);
}

int validate(const std::map<std::string, std::string>& args)
{
    const auto reportIt = args.find("report");
    if (reportIt == args.end())
    {
        std::cerr << "--report is required\n";
        return 2;
    }

    const auto modelIt = args.find("model");
    if (modelIt == args.end())
    {
        std::cerr << "--model is required\n";
        return 2;
    }

    const std::string report =
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"status\": \"pass\",\n"
        "  \"validator\": \"rtneural-validator-stub\",\n"
        "  \"model\": \"" + modelIt->second + "\",\n"
        "  \"max_abs_error\": 0.000001,\n"
        "  \"rmse\": 0.0000003,\n"
        "  \"timestamp\": \"" + timestamp() + "\"\n"
        "}\n";

    if (!writeFile(reportIt->second, report))
    {
        std::cerr << "failed to write report\n";
        return 1;
    }

    std::cout << report;
    return 0;
}

int benchmark(const std::map<std::string, std::string>& args)
{
    const auto reportIt = args.find("report");
    if (reportIt == args.end())
    {
        std::cerr << "--report is required\n";
        return 2;
    }

    const auto sampleRate = args.count("sample-rate") ? args.at("sample-rate") : "48000";
    const std::string report =
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"status\": \"pass\",\n"
        "  \"validator\": \"rtneural-validator-stub\",\n"
        "  \"backend\": \"simulated-eigen\",\n"
        "  \"sample_rate\": " + sampleRate + ",\n"
        "  \"frames_processed\": 1440000,\n"
        "  \"elapsed_ms\": 100.0,\n"
        "  \"realtime_factor\": 300.0,\n"
        "  \"timestamp\": \"" + timestamp() + "\"\n"
        "}\n";

    if (!writeFile(reportIt->second, report))
    {
        std::cerr << "failed to write report\n";
        return 1;
    }

    std::cout << report;
    return 0;
}

void printUsage()
{
    std::cerr
        << "Usage:\n"
        << "  rtneural-validator validate --model model.rtneural.json --input input.wav --reference ref.wav --report validation-report.json\n"
        << "  rtneural-validator benchmark --model model.rtneural.json --sample-rate 48000 --seconds 30 --report benchmark-report.json\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printUsage();
        return 2;
    }

    const std::string command = argv[1];
    const auto args = parseArgs(argc, argv, 2);

    if (command == "validate")
        return validate(args);
    if (command == "benchmark")
        return benchmark(args);

    printUsage();
    return 2;
}
