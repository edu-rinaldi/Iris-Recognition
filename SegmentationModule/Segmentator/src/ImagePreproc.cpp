#include "ImagePreproc.h"
#include "Util.h"
#include <opencv2/opencv.hpp>

namespace erb
{

std::vector<cv::Rect> getEyeRegionsOfInterest(const cv::Mat& src)
{
    auto eyeCascadeClassifier = cv::CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");
    auto eyes = std::vector<cv::Rect>();
    eyeCascadeClassifier.detectMultiScale(src, eyes, 1.1, 3, 0, cv::Size(), src.size());
    return eyes;
}

CropEyeInfo automaticCrop(const cv::Mat& src, cv::Mat& out)
{
    CropEyeInfo info;
    info.success = true;
    auto eyes = getEyeRegionsOfInterest(src);
    if (eyes.size() == 0) 
    {
        info.success = false;
        return info; 
    }

    // I take the eye with the largest area
    info.roi = *std::max_element(eyes.begin(), eyes.end(), [](const cv::Rect& l, const cv::Rect& r) { return l.area() < r.area(); });
    if (info.roi.width < 120) 
    {
        info.success = false;
        return info;
    }
    out = src(info.roi);
    return info;
}

CropEyeInfo manualCrop(const cv::Mat& src, cv::Mat& out)
{
    CropEyeInfo info;
    info.roi = src.cols > src.rows ? cv::Rect(src.cols * 0.5 - src.rows * 0.5, 0, src.rows, src.rows) :
        cv::Rect(0, src.rows * 0.5 - src.cols * 0.5, src.cols, src.cols);
    
    out = src(info.roi);
    info.success = true;
    return info;
}

ScaleEyeInfo scaleImage(const cv::Mat& src, cv::Mat& out, int finalSize)
{
    ScaleEyeInfo info;
    info.from = src.size();
    info.to = info.from;
    
    double sf = (double)finalSize / std::max(src.cols, src.rows);
    info.to = cv::Size((int)src.size().width * sf, (int)src.size().height * sf);
    cv::resize(src, out, info.to);
    
    //info.success = src.cols > finalSize || src.rows > finalSize;
    return info;
}

void filterReflection(const cv::Mat& src, cv::Mat& out)
{
    cv::Mat mask = cv::Mat(src.size(), CV_8UC1);
    cv::Mat greySrc;
    cv::cvtColor(src, greySrc, cv::COLOR_BGR2GRAY);
    // Pixel brighter
    double brightestPixel;
    cv::minMaxLoc(greySrc, nullptr, &brightestPixel);


    // Create mask
    cv::threshold(greySrc, mask, brightestPixel * 0.93, brightestPixel, cv::THRESH_BINARY);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7, 7)));
    cv::inpaint(src, mask, out, 5, cv::INPAINT_TELEA);
}

void automaticBrightnessContrast(const cv::Mat& src, cv::Mat& out)
{
    int channels[] = { 0 };
    int histSize[] = { 256 };

    float lranges[] = { 0, 256 };
    const float* ranges[] = { lranges };
    cv::Mat hist;
    cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    // Calculate cumulative distribution from the histogram
    std::vector<float> accumulator;
    accumulator.reserve(256);
    accumulator.push_back(hist.at<float>(0, 0));
    for (int i = 1; i < 256; ++i)
        accumulator.push_back(accumulator.back() + hist.at<float>(i, 0));

    // Locate points to clip
    float maximum = accumulator.back();
    float clipHistPercent = (maximum / 100.0f) * 0.5f;

    // Locate left cut
    int minimumGray = 0;
    while (accumulator[minimumGray] < clipHistPercent) minimumGray += 1;

    // Locate right cut
    int maximumGray = 255;
    while (accumulator[maximumGray] >= (maximum - clipHistPercent)) maximumGray -= 1;

    // Calculate alpha and beta values
    float alpha = 255.f / (maximumGray - minimumGray);
    float beta = -minimumGray * alpha;

    cv::convertScaleAbs(src, out, alpha, beta);
    //auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
}

PreprocessInfo preprocessImage(const cv::Mat& src, cv::Mat& out, int scaleSize)
{
    cv::Mat eye;
    PreprocessInfo info;
    info.crop = automaticCrop(src, eye);
    if (!info.crop.success)
    {
        LOG("Cannot find an eye in the image, assuming there's one at the center");
        info.crop = manualCrop(src, eye);
    }

    // scale image to low res for speeding up next computations
    cv::Mat eyeScaled;
    info.scale = scaleImage(eye, eyeScaled, scaleSize);
    LOG("Image scaled from [" << eye.rows << "x" << eye.cols << "] ==> [" << eyeScaled.rows << "x" << eyeScaled.cols << "]");
    out = eyeScaled;
    return info;
}

void posterization(const cv::Mat& src, cv::Mat& out, int k)
{
    out = cv::Mat::zeros(src.rows, src.cols, src.type());
    int x = 0;

    // For each row
    for (int y = 0; y < src.rows; y++)
    {
        // I initialise a new histogram and a new sliding window of size (k*2+1)^2
        auto histo = std::unordered_map<uchar, int>();
        SlidingWindow slidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);

        // topcolor is the most frequent colour
        int topColor = -1;

        // calculate histogram in the window
        for (int ty = slidingWindow.sy; ty < slidingWindow.ey; ty++)
        {
            for (int tx = slidingWindow.sx; tx < slidingWindow.ex; tx++)
            {
                int colore = src.at<uchar>(ty, tx);
                if (topColor == -1) topColor = colore;
                histo[colore] += 1;
                if (histo[topColor] < histo[colore]) topColor = colore;
            }
        }

        // Sets the colour of the pixel in the first column of the image to the most frequent one
        out.at<uchar>(y, x) = topColor;

        // For other columns
        for (int x = 1; x < src.cols; x++)
        {
            auto newSlidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);
            // Scroll for each line of the sliding window
            for (int ty = slidingWindow.sy; ty < slidingWindow.ey; ty++)
            {
                // Deletes the occurrences of the pixels in the first column of the sliding window from the histogram.
                histo[src.at<uchar>(ty, slidingWindow.sx)] -= 1;

                // scroll the window to the right
//                slidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);

                // Get the colour in the last column and update the histogram
                auto colorTmp = src.at<uchar>(ty, newSlidingWindow.ex - 1);
                histo[colorTmp] += 1;
                // Update topColor
                if (histo[topColor] < histo[colorTmp]) topColor = colorTmp;
            }
            slidingWindow = newSlidingWindow;
            // Assign top color
            out.at<uchar>(y, x) = topColor;
        }
    }
}

}