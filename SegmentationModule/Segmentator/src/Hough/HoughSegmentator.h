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
	/**
	 * Segment iris image
	 *
	 * @param img Iris image
	 * @return SegmentationData struct containing all segmentation informations: limbus and pupil circle, normalization data, etc..
	 */
	SegmentationData Segment(const cv::Mat& img) const override;
// Private methods
private:
	/**
	 * Find two circles: limbus and pupil
	 *
	 * @param img Iris image
	 * @return Iris struct object
	 */
	Iris IrisCircles(const cv::Mat& img) const;
	/**
	 * Find pupil circle
	 *
	 * @param img Iris image
	 * @return pupil circle
	 */
	Circle PupilCircle(const cv::Mat& img) const;
	/**
	 * Find limbus circle
	 *
	 * @param img Iris image
	 * @return limbus circle
	 */
	Circle LimbusCircle(const cv::Mat& img, const Circle& pupil, int centerRange, int radiusRange) const;
private:
	int mFinalSize;
	mutable cv::RNG mRng;

};

}

#endif // !__HOUGHSEGMENTATOR_H_
