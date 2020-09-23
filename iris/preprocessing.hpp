//
//  preprocessing.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef preprocessing_hpp
#define preprocessing_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "circle.hpp"
#include "utility.hpp"



namespace isis
{
#define SIMPLE_THRESHOLD 0
#define ADAPTIVE_THRESHOLD 1

static const std::string HAARCASCADE_EYE_TREE_EYEGLASSES = "Resources/haarcascade_eye_tree_eyeglasses.xml";
static const std::string HAARCASCADE_EYE = "Resources/haarcascade_eye.xml";

void posterization(const cv::Mat& src, cv::Mat& out, const int& k);

void filterReflectionST(const cv::Mat& src, cv::Mat& out);
void filterReflectionAT(const cv::Mat& src, cv::Mat& out);
void filterReflection(const cv::Mat& src, cv::Mat& out, const int& mode = SIMPLE_THRESHOLD);

void cropEye(const cv::Mat& src, cv::Mat& out, const std::string& classifierPath);

void blurFilter(const cv::Mat& mat, cv::Mat& out,const int& windowSize, const int& n);

inline void eyelashSmoothing(const cv::Mat& src, cv::Mat& out) { blurFilter(src, out, 10, 10); }

void gradientImage(const cv::Mat& src, cv::Mat& out, const bool& up);

bool scaleImage(const cv::Mat& src, cv::Mat& out, const int& finalSize);
}
#endif /* preprocessing_hpp */
