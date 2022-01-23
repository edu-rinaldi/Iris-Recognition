#ifndef __SEGMENTATION_H_
#define __SEGMENTATION_H_

#include "Util.h"
#include "Normalization.h"

#include <opencv2/imgproc.hpp>


// Segmentation
namespace erb 
{

struct SegmentationData
{
	Iris iris;
	NormalizedIris irisNormalized;
};

class Segmentator
{
public:
	virtual SegmentationData Segment(const cv::Mat& img) const = 0;
protected:
	inline Circle TransformCircle(const Circle& circle, const cv::Size& from, const cv::Size& to) const
	{
		Circle nc;

		nc.center[0] = static_cast<float>(circle.center[0]) / static_cast<float>(from.width) * to.width;
		nc.center[1] = static_cast<float>(circle.center[1]) / static_cast<float>(from.height) * to.height;
		nc.radius = static_cast<float>(circle.radius) / static_cast<float>(from.width) * to.width;
		return nc;
	}
};



}

#endif // !__SEGMENTATION_H_

