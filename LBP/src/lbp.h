#ifndef _LBP_H_
#define _LBP_H_

#include "Segmentation.h"

#include <opencv2/imgcodecs.hpp>

namespace erb
{

struct LBPEncoding
{
	int zoneNumber = 5;
	std::vector<cv::Mat> histograms = {};		// Istogrammi in zone dell'immagine
	std::vector<cv::Mat> zones		= {};		// Maschera delle zone   
};

void lbp(const cv::Mat& src, cv::Mat& out, cv::Mat& uniformMask);

class LBPEncoder
{
public:
	LBPEncoder(const SegmentationData& segmentation);
	LBPEncoding encode(int zoneNumber) const;
	inline cv::Mat getLBPResult() const { return mLBPRes; }
private:
	SegmentationData mSegmentation;
	cv::Mat mLBPRes;
	cv::Mat mUniformMask;
};

double similarity(const LBPEncoding& c1, const LBPEncoding& c2);
}

#endif
