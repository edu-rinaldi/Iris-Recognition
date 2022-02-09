#include "IsisSegmentator.h"
#include "ImagePreproc.h"

#include <algorithm>
#include <execution>

namespace isis
{
IsisSegmentator::IsisSegmentator(int finalSize) : mFinalSize(finalSize)
{
}

SegmentationData IsisSegmentator::Segment(const cv::Mat& src) const
{
	SegmentationData record;
	cv::Mat img;
	src.copyTo(img);

	auto preprocessInfo = preprocessImage(img, img, mFinalSize);
	if (!preprocessInfo.crop.success)
	{
		LOG("Crop failed");
		return {};
	}

	// filter reflections
	filterReflection(img, img);
	if (img.channels() > 1)
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	// Adjust brightness and contrast
	automaticBrightnessContrast(img, img);

	record.iris = IrisCircles(img);
	if (!record.iris.isValid())
	{
		LOG("Iris not found");
		return {};
	}
	auto& iris = record.iris;

	iris.limbus = TransformCircle(iris.limbus, preprocessInfo.scale.to, preprocessInfo.scale.from);
	iris.pupil = TransformCircle(iris.pupil, preprocessInfo.scale.to, preprocessInfo.scale.from);

	img = src(preprocessInfo.crop.roi);

	record.irisNormalized = normalizeIris(img, iris);
	
	return record;
}
Iris IsisSegmentator::IrisCircles(const cv::Mat& img) const
{
	Iris iris;
	iris.limbus = LimbusCircle(img);

	iris.pupil = PupilCircle(img, iris.limbus);
	return iris;
}

struct CircleSearchRecord
{
	Circle circle;
	double score = 0;
};

void canny(const cv::Mat& src, cv::Mat& out, double cannyThreshold)
{
	cv::Mat blurred;
	src.copyTo(blurred);

	cv::medianBlur(blurred, blurred, 3);
	cv::equalizeHist(blurred, blurred);
	cv::Canny(blurred, out, cannyThreshold, cannyThreshold * 3, 5);
}

Circle taubin(const std::vector<cv::Point>& contour)
{
	cv::Point2d sum = { 0,0 };
	for (const cv::Point& p : contour)
	{
		sum.x += p.x;
		sum.y += p.y;
	}

	cv::Point2d mean = { sum.x / contour.size(), sum.y / contour.size() };
	// Mxx = 0, Myy = 1, Mxy = 2, Mxz = 3, Myz = 4, Mzz = 5
	double m[] = { 0., 0., 0., 0., 0., 0. };

	double xi, yi, zi;
	for (const cv::Point& p : contour)
	{
		xi = p.x - mean.x;
		yi = p.y - mean.y;
		zi = cv::pow(xi, 2) + cv::pow(yi, 2);

		// Mxy
		m[2] += xi * yi;
		// Mxx
		m[0] += cv::pow(xi, 2);
		// Myy
		m[1] += cv::pow(yi, 2);
		// Mxz
		m[3] += xi * zi;
		// Myz
		m[4] += yi * zi;
		// Mzz
		m[5] += cv::pow(zi, 2);

	}

	for (int i = 0; i < 6; i++) m[i] /= contour.size();

	// coeff
	double  mz = m[0] + m[1],
		cov_xy = m[0] * m[1] - cv::pow(m[2], 2),
		var_z = m[5] - cv::pow(mz, 2),
		a3 = 4. * mz,
		a2 = -3. * cv::pow(mz, 2) - m[5],
		a1 = var_z * mz + 4. * cov_xy * mz - cv::pow(m[3], 2) - cv::pow(m[4], 2),
		a0 = m[3] * (m[3] * m[1] - m[4] * m[2]) + m[4] * (m[4] * m[0] - m[3] * m[2]) - var_z * cov_xy,
		a22 = 2. * a2,
		a33 = a3 * 3.,
		x = 0.,
		y = a0;

	for (int i = 0; i < 99; i++)
	{
		double dy = a1 + x * (a22 + a33 * x);
		double xnew = x - y / dy;
		if ((xnew == x) || (!(xnew < std::numeric_limits<double>::max()))) break;

		double ynew = a0 + xnew * (a1 + xnew * (a2 + xnew * a3));
		if (cv::abs(ynew) >= cv::abs(y)) break;

		x = xnew;
		y = ynew;
	}

	double det = cv::pow(x, 2) - x * mz + cov_xy,
		xcenter = (m[3] * (m[1] - x) - m[4] * m[2]) / det / 2.,
		ycenter = (m[4] * (m[0] - x) - m[3] * m[2]) / det / 2.;

	double radius = cv::sqrt(cv::pow(xcenter, 2) + cv::pow(ycenter, 2) + mz);
	return Circle{ static_cast<int>(radius), cv::Vec2i(xcenter + mean.x, ycenter + mean.y) };

}

void findCirclesTaubin(const cv::Mat& mat, std::vector<Circle>& outputCircles, double minRadius, double maxRadius)
{
	outputCircles = std::vector<Circle>();
	std::vector<double> cth = { 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 };
	for (auto cannyThreshold : cth)
	{
		cv::Mat cannyRes;
		canny(mat, cannyRes, cannyThreshold);

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(cannyRes, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
		std::mutex m;
		std::for_each(std::execution::par, contours.begin(), contours.end(), [&](auto& contour)
			{
				// controlla cerchio
				if (contour.size() <= 5) return;
				Circle circle = taubin(contour);

				if (circle.inside(mat) && circle.radius >= minRadius && circle.radius <= maxRadius)
				{
					std::lock_guard<std::mutex> guard(m);
					outputCircles.push_back(circle);
				}
			});
	}
}

CircleSearchRecord findLimbus(const cv::Mat& mat, const std::vector<Circle>& circles)
{
	CircleSearchRecord bestCircle;
	std::mutex m;
	std::for_each(std::execution::par, circles.begin(), circles.end(), [&](auto& circle)
	{
		// homogeneity
		double homogeneityScore = homogeneity(mat, circle);
		// separability
		double separabilityScore = separability(mat, circle);
		double score = homogeneityScore + separabilityScore;
		std::lock_guard<std::mutex> guard(m);
		if (bestCircle.circle.radius == 0 || bestCircle.score < score)
			bestCircle = { circle, score };
	});

	return bestCircle;
}

Circle IsisSegmentator::LimbusCircle(const cv::Mat& img) const
{
	cv::Mat tmpColor;
	CircleSearchRecord bestLimbus;
	int size = img.rows;
	for (int k = 1; k < 18; k++)
	{
		cv::Mat posterized;
		posterization(img, posterized, k);
		cv::cvtColor(posterized, tmpColor, cv::COLOR_GRAY2BGR);
		std::vector<Circle> circles;
		findCirclesTaubin(posterized, circles, size * 0.15, size * 0.5);
		if (circles.empty()) continue;
		// find best limbus
		auto bestKLimbus = findLimbus(img, circles);

		if (bestLimbus.circle.radius == 0 || bestLimbus.score < bestKLimbus.score)
			bestLimbus = bestKLimbus;
	}

	return bestLimbus.circle;
}

CircleSearchRecord findPupil(const cv::Mat& mat, std::vector<Circle>& circles, const Circle& limbus)
{
	CircleSearchRecord bestCircle;

	Circle defaultCircle = Circle{ static_cast<int>(limbus.radius / 4.f), cv::Vec2i(mat.cols / 2, mat.rows / 2) };
	circles.push_back(defaultCircle);
	std::mutex m;
	std::for_each(std::execution::par, circles.begin(), circles.end(), [&](auto& c)
	{
		double homogeneityScore = homogeneity(mat, c);
		double separabilityScore = separability(mat, c);
		double score = homogeneityScore + separabilityScore;
		std::lock_guard<std::mutex> guard(m);
		if (score > bestCircle.score) bestCircle = { c, score };
	});

	if (bestCircle.circle.radius == 0)
		bestCircle = { defaultCircle, homogeneity(mat, defaultCircle) + separability(mat, defaultCircle) };

	return bestCircle;
}

Circle IsisSegmentator::PupilCircle(const cv::Mat& img, const Circle& limbus) const
{
	CircleSearchRecord bestPupil;
	cv::Mat limbusCropped = img(limbus.getbbox());
	auto centerCrop = cv::Point(limbusCropped.cols / 2, limbusCropped.rows / 2);
	for (int k = 1; k < 18; k++)
	{
		cv::Mat posterized;
		posterization(limbusCropped, posterized, k);

		std::vector<Circle> circles;
		findCirclesTaubin(posterized, circles, 0.1 * limbusCropped.rows, 0.2 * limbusCropped.rows);

		for (int i = (int)circles.size() - 1; i >= 0; i--)
		{
			if (mean(posterized, circles[i]) > 40.0
				|| !circles[i].inside(centerCrop.x, centerCrop.y)
				|| distance(circles[i].center, centerCrop) > 0.05 * limbusCropped.rows
				)
				circles.erase(circles.begin() + i);
		}

		auto bestKPupil = findPupil(limbusCropped, circles, limbus);
		const auto tmp = bestKPupil.circle;

		bestKPupil = { Circle{tmp.radius, cv::Vec2i(tmp.center[0] + limbus.getbbox().x, tmp.center[0] + limbus.getbbox().y)},
					   bestKPupil.score
		};

		if (bestPupil.circle.radius == 0 || bestPupil.score < bestKPupil.score) bestPupil = bestKPupil;
	}

	return bestPupil.circle;
}
}