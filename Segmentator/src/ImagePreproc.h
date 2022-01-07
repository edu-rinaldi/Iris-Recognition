#ifndef __IMAGEPREPROC_H_
#define __IMAGEPREPROC_H_

#include <opencv2/imgproc.hpp>
namespace erb
{

struct CropEyeInfo
{
    bool success = false;
    cv::Rect roi;
};

struct ScaleEyeInfo
{
    cv::Size from;
    cv::Size to;
};

struct PreprocessInfo
{
    CropEyeInfo crop;
    ScaleEyeInfo scale;
};

// Finestra utilizzata per il filtro di posterizzazione
struct SlidingWindow
{
    int sx, ex, sy, ey;
    static inline SlidingWindow getSlidingWindow(const cv::Mat& img, int y, int x, int k)
    {
        auto sw = SlidingWindow{};
        // colonna iniziale e finale
        sw.sx = x - k >= 0 ? x - k : 0;
        sw.ex = x + k <= img.cols ? x + k : img.cols;
        // riga iniziale e finale
        sw.sy = y - k >= 0 ? y - k : 0;
        sw.ey = y + k <= img.rows ? y + k : img.rows;
        return sw;
    }
};

std::vector<cv::Rect> getEyeRegionsOfInterest(const cv::Mat& src);
CropEyeInfo automaticCrop(const cv::Mat& src, cv::Mat& out);
CropEyeInfo manualCrop(const cv::Mat& src, cv::Mat& out);
ScaleEyeInfo scaleImage(const cv::Mat& src, cv::Mat& out, int finalSize);
void filterReflection(const cv::Mat& src, cv::Mat& out);
void automaticBrightnessContrast(const cv::Mat& src, cv::Mat& out);
PreprocessInfo preprocessImage(const cv::Mat& src, cv::Mat& out, int scaleSize);
void posterization(const cv::Mat& src, cv::Mat& out, int k);

}
#endif // !__IMAGEPREPROC_H_
