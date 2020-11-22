//
//  utility.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef utility_hpp
#define utility_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

inline cv::Scalar getRandomColor(cv::RNG& rng){ return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));}
inline double distance(const cv::Point& a, const cv::Point& b) { return cv::sqrt(cv::pow(b.x - a.x, 2) + cv::pow(b.y - a.y, 2) * 1.0); }
inline bool inside(const cv::Size& size, const cv::Point& point) { return point.x >= 0 && point.x < size.width && point.y >= 0 && point.y < size.height; }

struct SlidingWindow
{
    int sx, ex, sy, ey;
    static SlidingWindow getSlidingWindow(const cv::Mat &img, const int &y,const int &x, const int &k);
};

int noise(const cv::Mat& src);

inline double map(const double& in, const double& is, const double& ie, const double& os, const double& oe)
{
    return os + ((oe - os) / (ie - is)) * (in - is);
}

std::string addStringToFileName(const std::string& fpath, const std::string& s);
std::string appendPath(const std::string& lpath, const std::string& rpath);
std::string getFileName(const std::string& fpath, const bool& withExtension = true);

void getPlainColorImg(const cv::Size& size, const cv::Scalar& scalar,cv::Mat& out);

#endif /* utility_hpp */
