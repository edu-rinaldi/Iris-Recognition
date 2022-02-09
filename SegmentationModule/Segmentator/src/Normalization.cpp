#include "Normalization.h"

#include "Segmentation.h"
#include "Util.h"

namespace erb {

void gradientImage(const cv::Mat& src, cv::Mat& out, bool up)
{
    cv::Mat kernel = cv::Mat();
    for (int y = 0; y < 10; y++)
    {
        double v = 0;
        if (y > 3 && y < 6) v = 0;
        else if (y < 4) v = up ? 1 : -1;
        else if (y > 5) v = up ? -1 : 1;
        double tmp[] = { v, v, v, v, v };
        kernel.push_back(cv::Mat(1, 5, CV_64F, tmp));
    }
    cv::filter2D(src, out, -1, kernel, cv::Point(-1, -1));
}

void lashAttenuation(const cv::Mat& src, cv::Mat& out)
{
    cv::Mat eyelashSmoothed;
    cv::medianBlur(src, eyelashSmoothed, 11);
    gradientImage(eyelashSmoothed, out, true);
}

std::vector<cv::Point> findEyelidPoints(const cv::Mat& mat, const Circle& limbus, const Circle& pupil, bool up)
{
    auto points = std::vector<cv::Point>();

    int startCol = (int)(limbus.center[0] - limbus.radius);
    int endCol = (int)(limbus.center[0] + limbus.radius);

    points.reserve(abs(endCol - startCol) + 1);

    int upperRow = up ? (int)(limbus.center[1] - limbus.radius) : (int)(pupil.center[1] + pupil.radius);
    int lowerRow = up ? (int)(pupil.center[1] - pupil.radius) : (int)(limbus.center[1] + limbus.radius);

    for (int col = startCol; col <= endCol; col++)
    {
        double maxValue = 0; int targetRow = 0;
        for (int row = lowerRow; row > upperRow; row--)
        {
            if (inside(mat.size(), cv::Point(col, row)))
            {
                double val = mat.at<uchar>(row, col);
                if (val > maxValue && ((up && row < pupil.center[1] - pupil.radius * 1.5) || (!up && row > pupil.center[1] + pupil.radius * 1.5)))
                {
                    maxValue = val;
                    targetRow = row;
                }
            }
        }
        points.emplace_back(col, targetRow);
    }
    return points;
}

void normalizeKrupicka(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil,
    map<cv::Point, cv::Point>& mapCartPol, map<cv::Point, cv::Point>& mapPolCart)
{
    //clear map
    mapCartPol = map<cv::Point, cv::Point>();

    int h = limbus.radius * 2;
    int w = std::round(limbus.radius * 2 * M_PI);
    out = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);

    double thetaStep = (2. * M_PI) / w;
    double xp, yp, xl, yl;

    int x, y, ind = 0;

    for (double i = 3. * M_PI / 2.; i < 2. * M_PI + 3. * M_PI / 2.; i += thetaStep)
    {
        xp = pupil.center[0] + pupil.radius * std::cos(i);
        yp = pupil.center[1] + pupil.radius * std::sin(i);
        xl = limbus.center[0] + limbus.radius * std::cos(i);
        yl = limbus.center[1] + limbus.radius * std::sin(i);
        for (int j = 0; j < h; ++j)
        {
            double pas = (double)j / h;
            x = std::round((1. - pas) * xl + pas * xp);
            y = std::round((1. - pas) * yl + pas * yp);
            if (inside(src.size(), cv::Point(x, y)))
            {
                out.at<cv::Vec3b>(j, ind) = src.at<cv::Vec3b>(y, x);
                mapCartPol[cv::Point(ind, j)] = cv::Point(x, y);
                mapPolCart[cv::Point(x, y)] = cv::Point(ind, j);
            }
        }
        ind++;
    }
}

void lowerEyelidMask(const cv::Mat& normalizedRedChannel, cv::Mat& lowerEyelidMask)
{
    lowerEyelidMask = cv::Mat(normalizedRedChannel.size(), CV_8UC1, cv::Scalar(255));

    auto meanArray = std::vector<double>(), stdDevArray = std::vector<double>();

    cv::Mat mask = cv::Mat::zeros(normalizedRedChannel.size(), CV_8UC1);

    for (int y = 0; y <= normalizedRedChannel.rows / 2; ++y)
        for (int x = normalizedRedChannel.cols / 4; x <= (3 * normalizedRedChannel.cols) / 4; ++x)
            mask.at<int>(y, x) = 1;

    cv::meanStdDev(normalizedRedChannel, meanArray, stdDevArray, mask);
    double mean = meanArray[0], stdDev = stdDevArray[0];
    int threshold = (int)(mean + stdDev);

    if (stdDev > mean / 4)
    {
        for (int y = 0; y < normalizedRedChannel.rows; ++y)
            for (int x = 0; x < normalizedRedChannel.cols; ++x)
                if (normalizedRedChannel.at<uchar>(y, x) > threshold)
                    lowerEyelidMask.at<uchar>(y, x) = 0;
    }
}

void negativeMask(const cv::Mat& src, cv::Mat& out,
    const Circle& limbus, const Circle& pupil,
    const std::vector<cv::Point>& upperEyelidPoints,
    const cv::Mat& lowerEyelidMask,
    const cv::Mat& reflectionsMask,
    const map<cv::Point, cv::Point>& map)
{
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    
    for (int y = 0; y < lowerEyelidMask.rows; y++)
        for (int x = 0; x < lowerEyelidMask.cols; x++)
            if (lowerEyelidMask.at<uchar>(y, x) == 0 || reflectionsMask.at<uchar>(y, x) != 0)
                out.at<uchar>(map.at(cv::Point(x, y))) = 255;

    for (int x = 0; x < src.cols; x++)
    {
        int upperEyelidEdge = 0;
        for (const auto& p : upperEyelidPoints)
            if (p.x == x) upperEyelidEdge = p.y;

        for (int y = 0; y < src.rows; y++)
            if (limbus.inside(x, y) && (y < upperEyelidEdge)) out.at<uchar>(y, x) = 255;
    }
}

void irisMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const cv::Mat& negMask)
{
    out = cv::Mat::zeros(src.size(), CV_8UC1);

    cv::circle(out, limbus.center, (int)limbus.radius - 1, cv::Scalar(255), -1);
    cv::circle(out, pupil.center, (int)pupil.radius, cv::Scalar(0), -1);

    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            if (negMask.at<uchar>(y, x) > 0.) out.at<uchar>(y, x) = 0;
}


void normalizedMask(const cv::Mat& irisCroppedMask, cv::Mat& mask,
    const cv::Size& normalizedSize,
    const map<cv::Point, cv::Point>& mapCartPol)
{
    mask = cv::Mat::zeros(normalizedSize, CV_8UC1);
    
    for (int y = 0; y < mask.rows; y++)
        for (int x = 0; x < mask.cols; x++)
        {
            auto polar = mapCartPol.find(cv::Point(x, y));
            //if (polar == mapCartPol.end()) continue;
            //if (!inside(irisCroppedMask.size(), polar)) continue;
            if ((int)irisCroppedMask.at<uchar>(polar->second) == 255)
            {
                mask.at<uchar>(y, x) = 255;
            }
        }
    
}

NormalizedIris erb::normalizeIris(const cv::Mat& eye, const Iris& iris)
{
    NormalizedIris record;
    record.eye = eye;

    cv::Mat grayEye;
    cv::cvtColor(eye, grayEye, cv::COLOR_BGR2GRAY);

    cv::Mat gradientUp;
    lashAttenuation(grayEye, gradientUp);

    // upper eyelid points
    auto upperEyelidPoints = findEyelidPoints(gradientUp, iris.limbus, iris.pupil, true);

    // normalize iris
    map<cv::Point, cv::Point> mapCartPol, mapPolCart;
    normalizeKrupicka(eye, record.irisNormalized, iris.limbus, iris.pupil, mapCartPol, mapPolCart);

    // split channels
    std::vector<cv::Mat> normalizedBGR;
    cv::split(record.irisNormalized, normalizedBGR);

    cv::Mat lowEyelidMask, reflectionMask;

    // lower eyelid mask
    lowerEyelidMask(normalizedBGR[2], lowEyelidMask);

    // reflection mask
    cv::adaptiveThreshold(normalizedBGR[0], reflectionMask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10);

    // negative mask
    cv::Mat negMask;
    negativeMask(eye, negMask, iris.limbus, iris.pupil, upperEyelidPoints, lowEyelidMask, reflectionMask, mapCartPol);

    // iris mask
    irisMask(eye, record.eyeMask, iris.limbus, iris.pupil, negMask);

    // iris mask normalized
    normalizedMask(record.eyeMask, record.irisNormalizedMask, record.irisNormalized.size(), mapCartPol);
	return record;
}

}

