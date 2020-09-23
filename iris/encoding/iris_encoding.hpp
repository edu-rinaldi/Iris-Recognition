//
//  iris_encoding.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 10/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef iris_encoding_hpp
#define iris_encoding_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utility.hpp"
#include "isis_segmentation.hpp"

#define NUM_ZONE 5

namespace isis
{

struct lbp_t
{
    cv::Mat histograms[NUM_ZONE],   // Istogrammi in zone dell'immagine
            zonesMask[NUM_ZONE];    // Maschera delle zone dell'immagine
    cv::Mat mask,
            img;
};

struct spatiogram_t
{
    cv::Mat histogram = cv::Mat::zeros(256, 1, CV_64FC1),  // Istogramma dell'immagine
            mu = cv::Mat::zeros(256, 2, CV_64FC1),         // Mean vector delle coordinate del pixel
            sigmaX = cv::Mat::zeros(256, 1, CV_64FC1),     // Matrici di covarianza rispettivamente della coordinata X e Y dei pixel
            sigmaY = cv::Mat::zeros(256, 1, CV_64FC1);
};

void encodeLBP(lbp_t& out, const segmentation_t& s);
double matchLBP(const lbp_t& c1, const lbp_t& c2);

void encodeSpatiogram(spatiogram_t& out, const segmentation_t& s);
double matchSpatiogram(const spatiogram_t& c1, const spatiogram_t& c2);
}

#endif /* iris_encoding_hpp */
