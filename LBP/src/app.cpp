#include "lbp.h"
#include "Hough/HoughSegmentator.h"
#include "Isis/IsisSegmentator.h"

#include <iostream>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace erb;

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    //isis::IsisSegmentator segmentator(250);
    hough::HoughSegmentator segmentator(250);
    
    

    auto segmentation1 = segmentator.Segment(img);
    if (!segmentation1.iris.isValid())
    {
        std::cout << "Error while segmenting iris image" << std::endl;
        return -1;
    }

    LBPEncoder encoder1(segmentation1);
    
    auto encoding1 = encoder1.encode(30);

    img = cv::imread(argv[2], cv::IMREAD_COLOR);

    auto segmentation2 = segmentator.Segment(img);
    if (!segmentation2.iris.isValid())
    {
        std::cout << "Error while segmenting iris image" << std::endl;
        return -1;
    }

    LBPEncoder encoder2(segmentation2);
    auto encoding2 = encoder2.encode(30);

    std::cout << "Similarity: " << similarity(encoding1, encoding2) << std::endl;
    /*cv::imshow("LBP Result", encoder.getLBPResult());
    cv::waitKey();*/
    return 0;
}