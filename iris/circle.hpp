//
//  Circle.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef circle_hpp
#define circle_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utility.hpp"

#define M_PI   3.14159265358979323846264338327950288
#define degToRad(angleInDegrees) ((angleInDegrees) * M_PI / 180.0)
#define radToDeg(angleInRadians) ((angleInRadians) * 180.0 / M_PI)


namespace isis
{
class Circle
{
public:
    double radius;
    cv::Point center;
    
    Circle();
    Circle(const double& radius, const cv::Point& center);
    
    bool inside(const cv::Rect& r) const;
    bool inside(const cv::Mat& mat) const;
    bool radiusInside(const double& sradius, const double& eradius) const;
    bool inside(const int& x, const int& y) const;
    cv::Rect getbbox() const;
    double homogeneity(const cv::Mat& src) const;
    double separability(const cv::Mat& src) const;
    double mean(const cv::Mat& src) const;
    inline double diameter() const {return radius * 2;}
    
    static Circle taubin(const std::vector<cv::Point>& contour);
    static double homogeneity(const cv::Mat& src, const Circle& c);
    static double separability(const cv::Mat& src, const Circle& c);
};

std::ostream& operator<<(std::ostream& os, Circle c);

}

#endif /* circle_hpp */
