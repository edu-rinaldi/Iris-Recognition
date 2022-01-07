#include "lbp.h"

#include "Util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace erb{

int calcPixel(const cv::Mat& src, cv::Mat& uniformMask, int y, int x)
{
    int center = src.at<uchar>(y, x);
    auto valList = std::vector<int>();

    const cv::Point dirs[] = { {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
    for (int i = 0; i < 8; i++)
    {
        auto dir = dirs[i];
        cv::Point newPoint = { cv::Point(x,y) + dir };
        if (inside(src.size(), newPoint))
            valList.push_back((src.at<uchar>(newPoint) > center) << (7-i));
        else valList.push_back(0);
    }
    int val = 0;
    int prev = valList[0];
    int nonuniformity = 0;
    for (int i = 0; i < valList.size(); i++)
    {
        if (prev != valList[i])
        {
            prev = valList[i];
            nonuniformity++;
        }
        val |= valList[i];
    }
    if (nonuniformity <= 2) uniformMask.at<uchar>(y, x) = 255;
    
    return val;
}

void lbp(const cv::Mat& src, cv::Mat& out, cv::Mat& uniformMask)
{
    cv::Mat tmp;
    
    if (src.channels() > 1)
        cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    else tmp = src;
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    uniformMask = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            out.at<uchar>(y, x) = calcPixel(tmp, uniformMask, y, x);
}

LBPEncoder::LBPEncoder(const SegmentationData& segmentation) : mSegmentation(segmentation)
{
    lbp(mSegmentation.irisNormalized.irisNormalized, mLBPRes, mUniformMask);
}

LBPEncoding LBPEncoder::encode(int zoneNumber) const
{
    LBPEncoding encoding;
    encoding.zoneNumber = zoneNumber;
    encoding.histograms.resize(zoneNumber);
    encoding.zones.resize(zoneNumber);
    
    //cv::Mat zoneMask = mSegmentation.eyeNormMask;
    cv::Mat zoneMask;
    cv::bitwise_and(mSegmentation.irisNormalized.irisNormalizedMask, mUniformMask, zoneMask);

    int subRows = mLBPRes.rows / zoneNumber;
    cv::Mat masked;
    cv::bitwise_and(mLBPRes, mLBPRes, masked, zoneMask);

    for (int zone = 0; zone < zoneNumber; zone++)
    {
        encoding.histograms[zone] = cv::Mat::zeros(256, 1, CV_32FC1);

        //da 0 a 255
        int histSize = 256;

        float range[] = { 0, 256 }; // range esclusivo
        const float* histRange = { range };
        cv::Mat currentZone = mLBPRes(cv::Range(zone * subRows, (zone + 1) * subRows), cv::Range(0, mLBPRes.cols));

        encoding.zones[zone] = zoneMask(cv::Range(zone * subRows, (zone + 1) * subRows), cv::Range(0, zoneMask.cols));
        cv::calcHist(&currentZone, 1, 0, encoding.zones[zone], encoding.histograms[zone], 1, &histSize, &histRange);
    }
    return encoding;
}

int noise(const cv::Mat& src)
{
    int count = 0;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            if (src.at<uchar>(y, x) == 0) count++;

    return count;
}

void drawHist(const std::vector<float>& data, cv::Mat3b& dst, int binSize = 3, int height = 0)
{
    int max_value = *max_element(data.begin(), data.end());
    int rows = 0;
    int cols = 0;
    if (height == 0) {
        rows = max_value + 10;
    }
    else {
        rows = std::max(max_value + 10, height);
    }

    cols = data.size() * binSize;

    dst = cv::Mat3b(rows, cols, cv::Vec3b(0, 0, 0));

    for (int i = 0; i < data.size(); ++i)
    {
        int h = rows - data[i];
        rectangle(dst, cv::Point(i * binSize, h), cv::Point((i + 1) * binSize - 1, rows), (i % 2) ? cv::Scalar(0, 100, 255) : cv::Scalar(0, 0, 255), cv::FILLED);
    }

}

double similarity(const LBPEncoding& c1, const LBPEncoding& c2)
{
    if (c1.zoneNumber != c2.zoneNumber) return 0;
    double score = 0;
    cv::Mat norm_hist1 = cv::Mat::zeros(256, 1, CV_32FC1);
    cv::Mat norm_hist2 = cv::Mat::zeros(256, 1, CV_32FC1);
    cv::Mat3b plotHist1, plotHist2;
    for (int zone = 0; zone < c1.zoneNumber; zone++)
    {
        cv::normalize(c1.histograms[zone], norm_hist1);
        cv::normalize(c2.histograms[zone], norm_hist2);
        
        double noisei = (double)noise(c1.zones[zone]) + noise(c2.zones[zone]);
        score += cv::compareHist(norm_hist1, norm_hist2, cv::HISTCMP_BHATTACHARYYA);// *(1.0 - noisei / (double)(2.0 * c1.zones[zone].rows * c1.zones[zone].cols));
    }

    return 1 - score / c1.zoneNumber;
}

}