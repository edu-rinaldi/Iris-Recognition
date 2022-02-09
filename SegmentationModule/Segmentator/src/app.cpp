#include "Hough/HoughSegmentator.h"
#include "Isis/IsisSegmentator.h"

#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include <ezOptionParser.hpp>

namespace fs = std::filesystem;

enum struct SegmentationMethod {HOUGH, ISIS};
static std::unordered_map<std::string, SegmentationMethod> const methodTable = { {"hough", SegmentationMethod::HOUGH}, {"isis", SegmentationMethod::ISIS} };

enum struct AppMode { APP_DEBUG, APP_SEGMENTATION };
static std::unordered_map<std::string, AppMode> const appModeTable = { {"debug", AppMode::APP_DEBUG}, {"segmentation", AppMode::APP_SEGMENTATION} };


struct AppParams
{
    SegmentationMethod segmentationMethod;
    int scaleSize;
    AppMode appMode;
    std::string input = "";
    std::string output = "";

};

template<typename K, typename T>
inline T getOrDefault(const std::unordered_map<K,T>& map, const K& key, T val)
{
    return (map.find(key) != map.end()) ? map.at(key) : val;
}

int main(int argc, const char* argv[])
{

    ez::ezOptionParser opt;
    opt.overview = "Segmentation application";
    opt.syntax = "SegmentatorApp (--in|-i) \"inputImage\" [(--out|-o) \"outputDirectory\"] [--method|-mt (\"hough\"|\"isis\")] [--size|-sz n] [--mode|-m (\"debug\"|\"segmentation\")]";
    opt.example = "SegmentatorApp --in image.png\n\n";
    opt.footer = "------------------------\n";

    opt.add("hough", false, 1, ' ', "Iris segmentation method", "-mt", "--method");
    opt.add("250", false, 1, ' ', "Image scale size", "-sz", "--size");
    opt.add("debug", false, 1, ' ', "App mode: debug segmentation or save segmentation", "-m", "--mode");
    opt.add("", true, 1, ',', "Input image", "-i", "--in", "--input");
    opt.add("", false, 1, ',', "Output image", "-o", "--out");
    opt.add("", false, 1, ',', "Help", "-h", "--help");
    opt.parse(argc, argv);

    std::vector<std::string> badopt;
    if (opt.isSet("-h") || !opt.gotRequired(badopt))
    {
        std::string usage;
        opt.getUsage(usage);
        std::cout << usage << std::endl;
        return -1;
    }

    AppParams params;
    // Iris segmentation method
    std::string parse;
    opt.get("-mt")->getString(parse);
    params.segmentationMethod = getOrDefault(methodTable, parse, SegmentationMethod::HOUGH);

    // Iris scale value
    opt.get("-sz")->getString(parse);
    params.scaleSize = std::atoi(parse.c_str());

    // App mode
    opt.get("-m")->getString(parse);
    params.appMode = getOrDefault(appModeTable, parse, AppMode::APP_DEBUG);

    // Input image
    opt.get("-i")->getString(params.input);
    if (!fs::exists(fs::path(params.input)))
    {
        std::cout << "Input path to image does not exists, please select a valid one" << std::endl;
        std::cout << "Input path: " << params.input << std::endl;
        return -1;
    }

    erb::Segmentator* segmentator;
    switch (params.segmentationMethod)
    {
    case SegmentationMethod::HOUGH:
        segmentator = &hough::HoughSegmentator(params.scaleSize);
        break;
    case SegmentationMethod::ISIS:
        segmentator = &isis::IsisSegmentator(params.scaleSize);
        break;
    }

    switch (params.appMode)
    {
    case AppMode::APP_DEBUG:
    {
        auto imgPath = fs::path(params.input);
        cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_COLOR);

        auto segmentation = segmentator->Segment(img);
        if (!segmentation.iris.isValid())
        {
            std::cout << "Error while segmenting iris image" << std::endl;
            return -1;
        }

        cv::circle(segmentation.irisNormalized.eye, segmentation.iris.limbus.center, segmentation.iris.limbus.radius, cv::Scalar(255, 0, 0));
        cv::circle(segmentation.irisNormalized.eye, segmentation.iris.pupil.center, segmentation.iris.pupil.radius, cv::Scalar(255, 0, 255));
        std::cout << "Limbus at: " << segmentation.iris.limbus << std::endl;
        std::cout << "Pupil at: " << segmentation.iris.pupil << std::endl;
        cv::imshow("Iris segmented", segmentation.irisNormalized.eye);
        cv::imshow("Iris normalized", segmentation.irisNormalized.irisNormalized);
        cv::imshow("Iris normalized mask", segmentation.irisNormalized.irisNormalizedMask);
        cv::waitKey();
    }
        break;
    case AppMode::APP_SEGMENTATION:
    {
        // Output dir
        if (!opt.isSet("-o")) {
            std::string usage;
            opt.getUsage(usage);
            std::cout << usage << std::endl;
            return -1;
        }
        opt.get("-o")->getString(params.output);

        auto imgPath = fs::path(params.input, fs::path::format::generic_format);
        auto outDirPath = fs::absolute(fs::path(params.output));
        cv::Mat img = cv::imread(imgPath.string(), cv::IMREAD_COLOR);
        
        auto segmentation = segmentator->Segment(img);
        if (!segmentation.iris.isValid())
        {
            std::cout << "Error while segmenting iris image" << std::endl;
            return -1;
        }

        if (!fs::exists(outDirPath)) fs::create_directories(outDirPath);

        auto extension = imgPath.extension();
        imgPath.replace_extension("");
        auto eyePath = outDirPath / fs::path(imgPath.filename().string() + "_eye" + extension.string());
        auto eyeNormPath = outDirPath / fs::path(imgPath.filename().string() + "_eyeNorm" + extension.string());
        auto eyeNormMask = outDirPath / fs::path(imgPath.filename().string() + "_eyeNormMask" + extension.string());
        auto eyeMask = outDirPath / fs::path(imgPath.filename().string() + "_eyeMask" + extension.string());
        cv::imwrite(eyePath.string(), segmentation.irisNormalized.eye);
        cv::imwrite(eyeNormPath.string(), segmentation.irisNormalized.irisNormalized);
        cv::imwrite(eyeNormMask.string(), segmentation.irisNormalized.irisNormalizedMask);
        cv::imwrite(eyeMask.string(), segmentation.irisNormalized.eyeMask);
    }
        break;
    }

	
	return 0;
}