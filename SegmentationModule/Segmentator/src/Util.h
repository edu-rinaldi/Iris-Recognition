#ifndef _UTIL_H_
#define _UTIL_H_
#include<iostream>
#include <opencv2/imgcodecs.hpp>

#define DEBUG

#if defined(_DEBUG) || defined(DEBUG) 
#define LOG(x) std::cout << x << std::endl 
#else	
#define LOG(x) 
#endif

#define M_PI  3.14159265358979323846  /* pi */
#define degToRad(angleInDegrees) ((angleInDegrees) * M_PI / 180.0)
#define radToDeg(angleInRadians) ((angleInRadians) * 180.0 / M_PI)

namespace erb
{
/*
* Tells if a point is inside a 2d rectangle
* @param size: rectangle size
* @param point: point to test
* @return true if point is inside the rectangle
*/
inline bool inside(const cv::Size& size, const cv::Point& point) { return point.x >= 0 && point.x < size.width&& point.y >= 0 && point.y < size.height; }
/*
* Calculate distance between two points
* @param a: first point
* @param b: second point
* @return distance between a and b
*/
inline double distance(const cv::Point& a, const cv::Point& b) { return cv::sqrt(cv::pow(b.x - a.x, 2) + cv::pow(b.y - a.y, 2) * 1.0); }

struct Circle
{
	int radius = 0;
	cv::Vec2i center;

	inline bool isValid() const { return radius > 0; }
    
    /*
    * Tells if the circle is inside a rectangle
    * @param r: rectangle
    * @return true if circle is inside the rectangle
    */
    inline bool inside(const cv::Rect& r) const
    {
        return center[0] - radius > r.tl().x && center[1] - radius > r.tl().y &&
            center[0] + radius <= r.br().x && center[1] + radius <= r.br().y;
    }

    /*
    * Tells if the circle is inside an image
    * @param mat: image
    * @return true if circle is inside the image
    */
    inline bool inside(const cv::Mat& mat) const { return inside(cv::Rect(cv::Point(0, 0), mat.size())); }
    /*
    * Tells if a point with located at (x,y) is inside circle
    * @param x: x coordinate
    * @param y: y coordinate
    * @return true if (x,y) is inside the circle
    */
    inline bool inside(int x, int y) const { return distance(cv::Point(x, y), center) < radius; }
    /*
    * Calculate the circle bounding box
    * @return circle bounding box
    */
    inline cv::Rect getbbox() const
    {
        return cv::Rect(cv::Point(this->center[0] - this->radius, this->center[1] - this->radius), cv::Size(this->radius * 2, this->radius * 2));
    }
};

// Allow to print circle information
inline std::ostream& operator<<(std::ostream& os, const Circle& c)
{
	os << "Center: " << c.center << " Radius: " << c.radius;
	return os;
}

/*
* Calculate homogeneity score of a circle in an image
* @param src: input image
* @param circle: circle to test
* @return homogeneity score
*/
double homogeneity(const cv::Mat& src, const Circle& circle);
/*
* Calculate separability score of a circle in an image
* @param src: input image
* @param circle: circle to test
* @return separability score
*/
double separability(const cv::Mat& src, const Circle& c);

/*
* Calculate mean value in a circle in an image
* @param src: input image
* @param circle: circle to test
* @return mean value in circle
*/
double mean(const cv::Mat& src, const Circle& circle);

// Define iris as two circles
struct Iris
{
	Circle pupil;
	Circle limbus;

	inline bool isValid() const { return pupil.isValid() && limbus.isValid(); }
};


}


#endif // !_UTIL_H_

