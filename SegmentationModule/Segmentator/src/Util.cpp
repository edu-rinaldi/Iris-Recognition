#include "Util.h"

#include <opencv2/imgproc.hpp>

namespace erb {

double homogeneity(const cv::Mat& src, const Circle& circle)
{
    // Calculating the circle mask
    cv::Mat circleMask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::circle(circleMask, circle.center, circle.radius, cv::Scalar(255), -1);

    // Calculating the histogram on that circle
    int channels[] = { 0 };
    int histSize[] = { 256 };

    float lranges[] = { 0, 256 };
    const float* ranges[] = { lranges };
    cv::Mat hist;
    cv::calcHist(&src, 1, channels, circleMask, hist, 1, histSize, ranges, true, false);

    // I use the formula for homogeneity
    double totalPx = 0, maxVal = 0, color = 0;
    for (int i = 0; i < hist.rows; i++)
    {
        totalPx += hist.at<float>(i, 0);
        if (hist.at<float>(i, 0) > maxVal)
        {
            maxVal = hist.at<float>(i, 0);
            color = i;
        }
    }

    return maxVal / totalPx;
}

double separability(const cv::Mat& src, const Circle& c)
{
    // Array keeping differences for each angle
    double indexes[360];

    // total differences
    double total = 0;

    // For every angle
    for (int i = 0; i < 360; i++)
    {
        // Internal abscissa, set to 4/5 of the radius
        int xInt = (int)(c.center[0] + c.radius * 0.8 * cos(degToRad(i)));
        // External abscissa, set at 6/5 of the radius
        int xExt = (int)(c.center[0] + c.radius * 1.2 * cos(degToRad(i)));
        // Internal ordinate, set at 4/5 of the radius
        int yInt = (int)(c.center[1] + c.radius * 0.8 * sin(degToRad(i)));
        // Outer ordinate, set at 6/5 of the radius
        int yExt = (int)(c.center[1] + c.radius * 1.2 * sin(degToRad(i)));

        // Corrects if external coordinates end up outside the matrix
        xExt = xExt < 0 ? 0 : xExt;
        yExt = yExt < 0 ? 0 : yExt;
        xExt = xExt >= src.cols ? src.cols - 1 : xExt;
        yExt = yExt >= src.rows ? src.rows - 1 : yExt;

        // Gets the difference between internal and external intensities
        indexes[i] = abs(src.at<uchar>(yExt, xExt) - src.at<uchar>(yInt, xInt));

        // Update sum of differences
        total += indexes[i];

    }
    // Average difference in angles
    double mean = total / 360.;

    // Calculation of standard deviation
    double sum = 0;
    for (int i = 0; i < 360; i++)
        sum += cv::pow((indexes[i] - mean), 2);

    double stddev = sqrt(sum / 360.);

    // Returns the mean value divided by the (standard deviation + 1)
    return mean / (stddev + 1);
}

double mean(const cv::Mat& src, const Circle& circle)
{
    double mean = 0;
    int countpx = 0;
    cv::Rect r = circle.getbbox();
    for (int i = r.tl().y; i < r.br().y; i++)
    {
        for (int j = r.tl().x; j < r.br().x; j++)
        {
            if (circle.inside(j, i))
            {
                mean += src.at<uchar>(i, j);
                countpx++;
            }
        }
    }
    mean /= countpx;
    return mean;
}

}