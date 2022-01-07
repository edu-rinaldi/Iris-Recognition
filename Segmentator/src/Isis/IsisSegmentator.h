#ifndef __ISISSEGMENTATOR_H_
#define __ISISSEGMENTATOR_H_

#include "Segmentation.h"

// Iris segmentation for identification systems v2 (ISis)
namespace isis{

using namespace erb;
class IsisSegmentator : public Segmentator
{
public:
	IsisSegmentator(int finalSize = 500);
	SegmentationData Segment(const cv::Mat& img) const override;
// Private methods
private:
	Iris IrisCircles(const cv::Mat& img) const;
	Circle PupilCircle(const cv::Mat& img, const Circle& limbus) const;
	Circle LimbusCircle(const cv::Mat& img) const;
private:
	int mFinalSize;

};
}

#endif // !__ISISSEGMENTATOR_H_
