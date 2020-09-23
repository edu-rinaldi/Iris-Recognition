//
//  isis_segmentation.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 08/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef isis_segmentation_hpp
#define isis_segmentation_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "preprocessing.hpp"
#include "settings.h"

namespace isis
{

typedef struct
{
    cv::Mat sourceCropped;
    cv::Mat normalized;
    cv::Mat mask;
    Circle pupil;
    Circle limbus;
} segmentation_t;


void segmentation(const cv::Mat& source, segmentation_t& segmentation);
void preprocess(cv::Mat& source, cv::Mat& out, const int& reflectionFilterMode = ADAPTIVE_THRESHOLD, const int& finalSize = 300);

Circle findLimbusBF(const cv::Mat& grey);
Circle findPupilBF(const cv::Mat& grey, const Circle& bestLimbus);

std::pair<Circle, double> findLimbus(const cv::Mat& mat, const std::vector<Circle>& circles);
std::pair<Circle, double> findPupil(const cv::Mat& mat, std::vector<Circle>& circles, const Circle& limbus);

void findCircles(const cv::Mat& mat, std::vector<Circle>& outputCircles, const double& minRadius, const double& maxRadius);

void canny(const cv::Mat& src, cv::Mat& out, double cannyThreshold);
void cannyThreshold(std::vector<double>& out);

std::vector<cv::Point> findEyelidPoints(const cv::Mat& mat, const Circle& limbus, const Circle& pupil, const bool& up);

void normalizeKrupicka(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, std::unordered_map<std::string, cv::Point>& mapCartPol, std::unordered_map<std::string, cv::Point>& mapPolCart);

void lowerEyelidMask(const cv::Mat& normalizedRedChannel, cv::Mat& lowerEyelidMask);

inline void reflectionsMask(const cv::Mat& src, cv::Mat& out) { cv::adaptiveThreshold(src, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10); }

void negativeMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const std::vector<cv::Point>& upperEyelidPoints, const cv::Mat& lowerEyelidMask, const cv::Mat& reflectionsMask, const std::unordered_map<std::string, cv::Point>& map);

void irisMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const cv::Mat& negMask);

void normalizedMask(const cv::Mat& irisCroppedMask, cv::Mat& mask, const cv::Size& normalizedSize, const std::unordered_map<std::string, cv::Point>& mapCartPol);
}

#endif /* isis_segmentation_hpp */
