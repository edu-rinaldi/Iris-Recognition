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
	Circle PupilCircle(const cv::Mat& img, const Circle& limbus) const;
	/**
	 * Find limbus circle
	 *
	 * @param img Iris image
	 * @return limbus circle
	 */
	Circle LimbusCircle(const cv::Mat& img) const;
private:
	int mFinalSize;

};
}

#endif // !__ISISSEGMENTATOR_H_
