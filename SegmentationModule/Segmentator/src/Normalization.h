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

struct NormalizedIris
{
    cv::Mat eye;
    cv::Mat irisNormalized;

    cv::Mat eyeMask;
    cv::Mat irisNormalizedMask;
};

struct Iris;

NormalizedIris normalizeIris(const cv::Mat& eye, const Iris& iris);

};

#endif // !__NORMALIZATION_H_
