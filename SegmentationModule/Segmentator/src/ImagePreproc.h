#ifndef __IMAGEPREPROC_H_
#define __IMAGEPREPROC_H_

#include <opencv2/imgproc.hpp>
namespace erb
{

// Contains information about crop process
struct CropEyeInfo
{
    bool success = false;
    cv::Rect roi;
};

// Contains information about scale process
struct ScaleEyeInfo
{
    cv::Size from;
    cv::Size to;
};

// Contains information about iris preprocessing
struct PreprocessInfo
{
    CropEyeInfo crop;
    ScaleEyeInfo scale;
};

// Window used for the posterization filter
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

/*
* Find all possible ROIs of an eye in an image
* @param src: input image
* @return list of candidates ROIs
*/
std::vector<cv::Rect> getEyeRegionsOfInterest(const cv::Mat& src);
/*
* Crop an iris image on the eye using an Haar Cascade classifier
* @param src: input iris image
* @param out: output cropped image
* @return crop process info
*/
CropEyeInfo automaticCrop(const cv::Mat& src, cv::Mat& out);
/*
* Crop an iris image on the eye on the center of the image
* @param src: input iris image
* @param out: output cropped image
* @return crop process info
*/
CropEyeInfo manualCrop(const cv::Mat& src, cv::Mat& out);
/*
* Scale an iris image to a fixed size
* @param src: input iris image
* @param out: output scaled image
* @param finalSize: size of the output image
* @return scale process info
*/
ScaleEyeInfo scaleImage(const cv::Mat& src, cv::Mat& out, int finalSize);

/*
* Remove reflections from iris image
* @param src: input iris image
* @param out: output image
*/
void filterReflection(const cv::Mat& src, cv::Mat& out);
/*
* Automatic equalization of image brightness and contrast
* @param src: input image
* @param out: output image
*/
void automaticBrightnessContrast(const cv::Mat& src, cv::Mat& out);

/*
* Preprocess iris image, preparing it for segmentation
* @param src: input iris image
* @param out: output preprocessed image
* @return preprocessing info
*/
PreprocessInfo preprocessImage(const cv::Mat& src, cv::Mat& out, int scaleSize);

/*
* Apply a posterization filter
* @param src: input iris image
* @param out: output image
* @param k: window size parameter, wSize = k*2+1
*/
void posterization(const cv::Mat& src, cv::Mat& out, int k);

}
#endif // !__IMAGEPREPROC_H_
