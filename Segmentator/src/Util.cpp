#include "Util.h"

#include <opencv2/imgproc.hpp>

namespace erb {

double homogeneity(const cv::Mat& src, const Circle& circle)
{
    // Calcolo la maschera del cerchio
    cv::Mat circleMask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::circle(circleMask, circle.center, circle.radius, cv::Scalar(255), -1);

    // Calcolo l'histogram su quel cerchio
    int channels[] = { 0 };
    int histSize[] = { 256 };

    float lranges[] = { 0, 256 };
    const float* ranges[] = { lranges };
    cv::Mat hist;
    cv::calcHist(&src, 1, channels, circleMask, hist, 1, histSize, ranges, true, false);

    // Uso la formula per l'omogeneità
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
    // Array che mantiene le differenze per ogni angolo
    double indexes[360];

    // totale differenze
    double total = 0;

    // Per ogni angolo
    for (int i = 0; i < 360; i++)
    {
        // Ascissa interna, settata a 4/5 del raggio
        int xInt = (int)(c.center[0] + c.radius * 0.8 * cos(degToRad(i)));
        // Ascissa esterna, settata a 6/5 del raggio
        int xExt = (int)(c.center[0] + c.radius * 1.2 * cos(degToRad(i)));
        // Ordinata interna, settata a 4/5 del raggio
        int yInt = (int)(c.center[1] + c.radius * 0.8 * sin(degToRad(i)));
        // Ordinata esterna, settata a 6/5 del raggio
        int yExt = (int)(c.center[1] + c.radius * 1.2 * sin(degToRad(i)));

        // Corregge il tiro qualora le coordinate esterne finiscano fuori dalla matrice
        xExt = xExt < 0 ? 0 : xExt;
        yExt = yExt < 0 ? 0 : yExt;
        xExt = xExt >= src.cols ? src.cols - 1 : xExt;
        yExt = yExt >= src.rows ? src.rows - 1 : yExt;

        // Ottiene la differenza tra le intensità interna ed esterna
        indexes[i] = abs(src.at<uchar>(yExt, xExt) - src.at<uchar>(yInt, xInt));

        // Aggiorna la somma delle differenze
        total += indexes[i];

    }
    // Differenza media rispetto agli angoli
    double mean = total / 360.;

    // Calcolo della deviazione standard
    double sum = 0;
    for (int i = 0; i < 360; i++)
        sum += cv::pow((indexes[i] - mean), 2);

    double stddev = sqrt(sum / 360.);

    // Restituisce il valore medio diviso per la (deviazione standard + 1)
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