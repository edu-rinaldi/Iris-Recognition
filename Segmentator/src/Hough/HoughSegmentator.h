#ifndef __HOUGHSEGMENTATOR_H_
#define __HOUGHSEGMENTATOR_H_
#include "Segmentation.h"


namespace hough
{
using namespace erb;
class HoughSegmentator : public Segmentator
{
public:
	HoughSegmentator(int finalSize = 500);
	SegmentationData Segment(const cv::Mat& img) const override;
// Private methods
private:
	Iris IrisCircles(const cv::Mat& img) const;
	Circle PupilCircle(const cv::Mat& img) const;
	Circle LimbusCircle(const cv::Mat& img, const Circle& pupil, int centerRange, int radiusRange) const;
private:
	int mFinalSize;
	mutable cv::RNG mRng;

};

}

#endif // !__HOUGHSEGMENTATOR_H_
