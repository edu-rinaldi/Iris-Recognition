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

// Numero di zone in cui dividere l'immagine per LBP
#define NUM_ZONE 5

namespace isis
{

struct lbp_t
{
    cv::Mat histograms[NUM_ZONE],   // Istogrammi in zone dell'immagine
            zones[NUM_ZONE];        // Maschera delle zone
    cv::Mat img;                    // Immagine con LBP applicato
};

struct spatiogram_t
{
    cv::Mat histogram = cv::Mat::zeros(256, 1, CV_64FC1),  // Istogramma dell'immagine
            mu = cv::Mat::zeros(256, 2, CV_64FC1),         // Mean vector delle coordinate del pixel
            sigmaX = cv::Mat::zeros(256, 1, CV_64FC1),     // Matrici di covarianza rispettivamente della coordinata X e Y dei pixel
            sigmaY = cv::Mat::zeros(256, 1, CV_64FC1);
};

/*
    Funzione utilizzata per LBP.
    Calcola il valore del pixel in base ai vicini (viene utilizzata la sommatoria sugli 8 pixel vicini)
 */
int getCalculatedPixel(const cv::Mat& src, const int& y, const int& x);

/*
    Funzione che applica l'operatore LBP
    @param src: immagine in input su cui applicare LBP
    @param out: immagine su cui vogliamo mettere l'output ottenuto applicando LBP.
 */
void applyLBP(const cv::Mat& src, cv::Mat& out);

/*
    Effettua la codifica LBP.
    @param out: struct lbp vuota, in questa verrà messo l'output della funzione
    @param s: struct contenente la segmentazione dell'iride da codificare
 */
void encodeLBP(lbp_t& out, const segmentation_t& s);

/*
    Date due codifiche restituisce il valore di similarità
 */
double matchLBP(const lbp_t& c1, const lbp_t& c2);

/*
    Effettua la codifica spatiogram
    @param out: struct spatiogram vuota, in questa verrà messo l'output della funzione
    @param s: struct contenente la segmentazione dell'iride da codificare
 */
void encodeSpatiogram(spatiogram_t& out, const segmentation_t& s);

/*
    Date due codifiche spatiogram restituisce un valore di dissimilarità
 */
double matchSpatiogram(const spatiogram_t& c1, const spatiogram_t& c2);
}

#endif /* iris_encoding_hpp */
