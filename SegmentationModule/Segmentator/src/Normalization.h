#ifndef __NORMALIZATION_H_
#define __NORMALIZATION_H_
#include <map>
#include <opencv2/imgproc.hpp>

namespace erb{

struct ComparePoints
{
    inline bool operator () (const cv::Point& a, const cv::Point& b) const { return (a.x < b.x) || (a.x == b.x && a.y < b.y); }
};

template<typename KeyT, typename ValT>
using map = std::map<KeyT, ValT, ComparePoints>;

// Contains all information about normalized segmented iris image
struct NormalizedIris
{
    cv::Mat eye;
    cv::Mat irisNormalized;

    cv::Mat eyeMask;
    cv::Mat irisNormalizedMask;
};

struct Iris;

/**
* Normalize eye cropped image, giving as input iris circles
*
* @param eye: Eye cropped image
* @param iris: iris circles
* @return NormalizedIris struct containing all normalization informations
*/
NormalizedIris normalizeIris(const cv::Mat& eye, const Iris& iris);

};

#endif // !__NORMALIZATION_H_
