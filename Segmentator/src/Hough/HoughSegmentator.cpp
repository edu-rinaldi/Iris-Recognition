#include "Hough/HoughSegmentator.h"
#include "ImagePreproc.h"
#include "Normalization.h"

#include <opencv2/highgui.hpp>
#include <algorithm>
#include <execution>
#include <mutex>
namespace hough {
	
HoughSegmentator::HoughSegmentator(int finalSize) : mFinalSize(finalSize)
{
}

SegmentationData HoughSegmentator::Segment(const cv::Mat& src) const
{
	SegmentationData record;
	cv::Mat img; 
	src.copyTo(img);
	if (img.channels() > 1)
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

	auto preprocessInfo = preprocessImage(img, img, mFinalSize);

	if (!preprocessInfo.crop.success)
	{
		LOG("Crop failed");
		return {};
	}
	
	record.iris = IrisCircles(img);
	auto& iris = record.iris;

	iris.limbus = TransformCircle(iris.limbus, preprocessInfo.scale.to, preprocessInfo.scale.from);
	iris.pupil = TransformCircle(iris.pupil, preprocessInfo.scale.to, preprocessInfo.scale.from);

	img = src(preprocessInfo.crop.roi);
	
	record.irisNormalized = normalizeIris(img, iris);
	
	return record;
}

Iris HoughSegmentator::IrisCircles(const cv::Mat& img) const
{
	Iris iris;
	LOG("Looking for a pupil");
	iris.pupil = PupilCircle(img);
	if (!iris.pupil.isValid()) return {};
	LOG("Pupil found");
	// finding limbus
	int radiusRange = std::ceil(iris.pupil.radius * 1.5);
	float multiplier = 0.25f;
	do
	{
		LOG("looking for limbus");
		multiplier += 0.05f;
		int centerRange = std::ceil(iris.pupil.radius * multiplier);
		LOG("Searching limbus with multiplier " << multiplier);
		iris.limbus = LimbusCircle(img, iris.pupil, centerRange, radiusRange);
	} while (!iris.limbus.isValid() && multiplier <= 0.7);
	if (iris.limbus.isValid())
	{
		LOG("Limbus found: " << iris.limbus);
	}
	return iris;
}

Circle HoughSegmentator::PupilCircle(const cv::Mat& img) const
{
	auto getEdges = [&](const cv::Mat& img) {
		cv::Mat edges;
		cv::Canny(img, edges, 20, 100);
		cv::dilate(edges, edges, cv::Mat::ones(3, 3, CV_8UC1), cv::Point(-1, -1), 2);
		int kSize = 2 * mRng.uniform(5, 11) + 1;
		cv::GaussianBlur(edges, edges, cv::Size(kSize, kSize), 0);
		return edges;
	};

	int param1 = 200;
	int param2 = 120;
	std::vector<cv::Vec3f> pupilCircles = {};
	while (param2 > 35 && pupilCircles.size() < 100)
	{
		for (int median : {3, 5, 7})
		{
			for (int threshold : {20, 25, 30, 35, 40, 45, 50, 55, 60})
			{
				// Median blur
				cv::Mat medianImg;
				cv::medianBlur(img, medianImg, 2 * median + 1);

				// threshold
				cv::Mat thresholdImg;
				cv::threshold(medianImg, thresholdImg, threshold, 255, cv::THRESH_BINARY_INV);

				// Find contours
				std::vector<std::vector<cv::Point>> contours;
				cv::findContours(thresholdImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

				cv::drawContours(thresholdImg, contours, -1, cv::Scalar(255), -1);

				// Canny
				auto edges = getEdges(thresholdImg);

				// HoughCircles
				std::vector<cv::Vec3f> circles = {};
				cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 1, param1, param2);
				if (!circles.empty())
				{
					pupilCircles.insert(pupilCircles.end(), circles.begin(), circles.end());
				}
			}
		}
		--param2;
	}
	// Get mean circle
	auto circleVec = cv::mean(pupilCircles);
	return Circle{ static_cast<int>(circleVec[2]), cv::Vec2i(circleVec[0], circleVec[1]) };
}

float getAlphaRadius(const std::vector<cv::Vec3f>& circles)
{
	float alphaRadius = 0;
	float minDist = std::numeric_limits<float>().max();
	for (const auto& c1 : circles)
	{
		float dist = 0;
		for (const auto& c2 : circles)
			dist += std::fabs(c2[2] - c1[2]);
		if (dist < minDist)
		{
			minDist = dist;
			alphaRadius = c1[2];
		}
	}
	return alphaRadius;
}

std::vector<cv::Vec3f> filterCircles(const std::vector<cv::Vec3f>& circles)
{
	cv::Vec3d mean, std;
	cv::meanStdDev(circles, mean, std);
	std::vector<cv::Vec3f> filteredPos;
	std::vector<cv::Vec3f> filtered;
	float ratio = 1.5;
	std::mutex m;
	std::for_each(std::execution::par, circles.begin(), circles.end(), [&](auto& circle)
	{
		auto meanCenter = cv::Vec2i(mean[0], mean[1]);
		auto stdCenter = cv::Vec2i(std[0], std[1]);
		auto tmp1 = meanCenter - ratio * stdCenter;
		auto tmp2 = meanCenter + ratio * stdCenter;
		if (!(circle[0] < tmp1[0] || circle[0] > tmp2[0] || circle[1] < tmp1[1] || circle[1] > tmp2[1]))
		{
			std::lock_guard<std::mutex> guard(m);
			filteredPos.push_back(circle);
		}
	});
	if (filteredPos.size() < 3) filtered = filteredPos;
	else
	{
		float alphaRadius = getAlphaRadius(filteredPos);
		cv::Vec3d filteredMean, filteredStd; 
		cv::meanStdDev(filteredPos, filteredMean, filteredStd);
		float maxRadius = alphaRadius + filteredStd[2];
		float minRadius = alphaRadius - filteredStd[2];
		std::mutex m;
		std::for_each(std::execution::par, circles.begin(), circles.end(), [&](auto& circle)
		{
			if (circle[2] >= minRadius && circle[2] <= maxRadius) {
				std::lock_guard<std::mutex> guard(m);
				filtered.push_back(circle);
			}
		});
	}
	return filtered;
}

Circle HoughSegmentator::LimbusCircle(const cv::Mat& img, const Circle& pupil, int centerRange, int radiusRange) const
{
	auto getEdges = [&](const cv::Mat& img, int threshold)
	{
		cv::Mat edges;
		cv::Canny(img, edges, 0, threshold, 5);
		cv::dilate(edges, edges, cv::Mat::ones(cv::Size(3, 3), CV_8UC1));
		int kSize = 2 * mRng.uniform(5, 11) + 1;
		cv::GaussianBlur(edges, edges, cv::Size(kSize, kSize), 0);
		return edges;
	};

	// check if p is inside c1
	auto inside = [](const Circle& c1, const cv::Vec2i p) { return cv::norm(p - c1.center) <= c1.radius; };

	int param1 = 200;
	int param2 = 120;
	std::vector<cv::Vec3f> limbusCircles = {};
	while (param2 > 40 && limbusCircles.size() < 50)
	{
		for (int median : {8, 10, 12, 14, 16, 18, 20})
		{
			for (int threshold : {430, 480, 530})
			{
				// Median blur
				cv::Mat medianImg;
				cv::medianBlur(img, medianImg, 2 * median + 1);

				// Canny
				auto edges = getEdges(medianImg, threshold);

				// HoughCircles
				std::vector<cv::Vec3f> circles = {};
				cv::HoughCircles(edges, circles, cv::HOUGH_GRADIENT, 1, 1, param1, param2);

				if (!circles.empty())
				{
					std::mutex m;
					// Filter and push circles
					std::for_each(std::execution::par, circles.begin(), circles.end(), [&](auto& circle)
					{
						if (circle[2] > radiusRange && inside({ centerRange, pupil.center }, cv::Vec2i(circle[0], circle[1])))
						{
							std::lock_guard<std::mutex> guard(m);
							limbusCircles.push_back(circle);
						}
					});
				}
			}
		}
		--param2;
	}
	if (limbusCircles.empty()) return {};
	auto filtered = cv::mean(filterCircles(limbusCircles));
	return {static_cast<int>(filtered[2]), cv::Vec2i(filtered[0], filtered[1])};
}

}