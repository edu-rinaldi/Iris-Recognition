//
//  iris_encoding.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 10/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "iris_encoding.hpp"

void isis::encodeLBP(lbp_t& out, const segmentation_t& s)
{
    cv::cvtColor(s.normalized, out.img, cv::COLOR_BGR2GRAY);
    out.mask = s.mask;
    for(int zone = 0; zone < NUM_ZONE; zone++)
    {
        out.histograms[zone] = cv::Mat::zeros(256, 1, CV_32FC1);
        
        // istogramma per la zona
        int subRows = out.img.rows / NUM_ZONE;
        
        //da 0 a 255
        int histSize = 256;
        
        // range esclusivo
        float range[] = { 0, 256 }; // range esclusivo
        const float* histRange = { range };
        out.zonesMask[zone] = out.img(cv::Range(zone * subRows, (zone+1) * subRows), cv::Range(0, out.img.cols));
        
        cv::calcHist(&out.zonesMask[zone], 1, 0, cv::Mat(), out.histograms[zone] , 1, &histSize, &histRange);
    }
}

double isis::matchLBP(const lbp_t& c1, const lbp_t& c2)
{
    double score = 0;
    for(int zone = 0; zone < NUM_ZONE; zone++)
    {
        double noisei = (noise(c1.zonesMask[zone])+noise(c2.zonesMask[zone]))/2.;
        score += cv::compareHist(c1.histograms[zone], c2.histograms[zone], cv::HISTCMP_BHATTACHARYYA) * (1 - noisei/(double)(c1.zonesMask[zone].total()));
    }
    return score / NUM_ZONE;
}

void isis::encodeSpatiogram(spatiogram_t &out, const segmentation_t &s)
{
    int bins = 256;
    cv::Mat img;
    cv::cvtColor(s.normalized, img, cv::COLOR_BGR2GRAY);
    cv::Mat binno = cv::Mat::zeros(img.size(), CV_32FC1),
            channels[img.channels()];
    
    cv::split(img, channels);

    int f = 1;
    for(int k = 0; k < img.channels(); k++)
    {
        for(int i = 0; i < channels[k].rows; i++)
            for(int j = 0; j < channels[k].cols; j++)
                binno.at<float>(i, j) += (float)(f * floor(channels[k].at<uchar>(i, j) * bins / 256));
        f *= bins;
    }
    
    cv::Mat gridX = cv::Mat::zeros(binno.size(), CV_32FC1),
            gridY = cv::Mat::zeros(binno.size(), CV_32FC1),
            kDist = cv::Mat::ones(img.size(), CV_32FC1) / img.total();
    
    for(int y = 0; y < binno.rows; y++)
    {
        for(int x = 0; x < binno.cols; x++)
        {
            gridY.at<float>(y, x) = map(y, 0, binno.rows, -1, 1);
            gridX.at<float>(y, x) = map(x, 0, binno.cols, -1, 1);
            kDist.at<float>(y, x) *= (double)s.mask.at<uchar>(y, x);
        }
    }
    
    float valSum = cv::sum(kDist)[0];
    
    for(int y = 0; y < kDist.rows; y++)
        for(int x = 0; x < kDist.cols; x++)
            kDist.at<float>(y, x) /= valSum;
    
    double minMK;
    cv::minMaxLoc(kDist, &minMK);
    
    cv::Mat wsum = cv::Mat::zeros(f, 1, CV_64FC1);
   
    // Funzione accumarray
    for(int y = 0; y < binno.rows; y++)
        for(int x = 0; x < binno.cols; x++)
        {
            out.histogram.at<double>((int)binno.at<float>(y, x)) += kDist.at<float>(y, x);
            wsum.at<double>((int)binno.at<float>(y, x)) += kDist.at<float>(y, x);
        }
    
    for(int i = 0; i < wsum.rows; i++)
        if(wsum.at<double>(i) == 0.0) wsum.at<double>(i) = 1.0;
    
    // Creazione del meanVector e delle matrici di covarianza
    for(int y = 0; y < binno.rows; y++)
        for(int x = 0; x < binno.cols; x++)
        {
            int binnoIndex = (int)binno.at<float>(y, x);
            
            out.mu.at<double>(binnoIndex, 0) += gridX.at<float>(y, x)*kDist.at<float>(y, x);
            out.mu.at<double>(binnoIndex, 1) += gridY.at<float>(y, x)*kDist.at<float>(y, x);
            
            out.sigmaX.at<double>(binnoIndex) += cv::pow(gridX.at<float>(y, x), 2.0)*kDist.at<float>(y, x);
            out.sigmaY.at<double>(binnoIndex) += cv::pow(gridY.at<float>(y, x), 2.0)*kDist.at<float>(y, x);
        }
    
    for(int i = 0; i < out.sigmaX.rows; i++)
    {
        out.sigmaX.at<double>(i) /= wsum.at<double>(i);
        out.sigmaX.at<double>(i) -= cv::pow(out.mu.at<double>(i, 0) / wsum.at<double>(i), 2.0);
        out.sigmaX.at<double>(i) += (minMK - out.sigmaX.at<double>(i)) * (out.sigmaX.at<double>(i) < minMK);
        
        out.sigmaY.at<double>(i) /= wsum.at<double>(i);
        out.sigmaY.at<double>(i) -= pow(out.mu.at<double>(i, 1) / wsum.at<double>(i), 2.0);
        out.sigmaY.at<double>(i) += (minMK - out.sigmaY.at<double>(i)) * (out.sigmaY.at<double>(i) < minMK);
    }
    
    // Normalizzazione meanVector
    for (int i = 0; i < out.mu.rows; i++)
    {
        out.mu.at<double>(i, 0) /= wsum.at<double>(i);
        out.mu.at<double>(i, 1) /= wsum.at<double>(i);
    }
    
}

double isis::matchSpatiogram(const spatiogram_t &c1, const spatiogram_t &c2)
{
    cv::Mat qx = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1),
            qy = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1),
            q  = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    
    double  C  = 2 * sqrt(M_PI * 2),
            C2 = 1 / (M_PI * 2);
    
    for (int i = 0; i < c1.sigmaX.rows; i++)
    {
        qx.at<double>(i) = c1.sigmaX.at<double>(i) + c2.sigmaX.at<double>(i);
        qy.at<double>(i) = c1.sigmaY.at<double>(i) + c2.sigmaY.at<double>(i);

        q.at<double>(i) = C * pow((qx.at<double>(i) * qy.at<double>(i)), 1 / 4.0);
    }
    
    cv::Mat sigmaiX = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1),
            sigmaiY = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    
    for (int i = 0; i < c2.sigmaX.rows; i++)
    {
        sigmaiX.at<double>(i) = 1.0 / (1.0 / (c1.sigmaX.at<double>(i) + (c1.sigmaX.at<double>(i) == 0)) + 1.0 / (c2.sigmaX.at<double>(i) + (c2.sigmaX.at<double>(i) == 0)));

        sigmaiY.at<double>(i) = 1.0 / (1.0 / (c1.sigmaY.at<double>(i) + (c1.sigmaY.at<double>(i) == 0)) + 1.0 / (c2.sigmaY.at<double>(i) + (c2.sigmaY.at<double>(i) == 0)));
    }
    
    cv::Mat Q = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    for (int i = 0; i < c1.sigmaX.rows; i++)
        Q.at<double>(i) = C * pow(sigmaiX.at<double>(i)*sigmaiY.at<double>(i), 1 / 4.0);
    
    cv::Mat x = cv::Mat::zeros(c1.mu.rows, 1, CV_64FC1);
    cv::Mat y = cv::Mat::zeros(c1.mu.rows, 1, CV_64FC1);
    
    for (int i = 0; i < c1.mu.rows; i++)
    {
        x.at<double>(i) = c1.mu.at<double>(i, 0) - c2.mu.at<double>(i, 0);
        y.at<double>(i) = c1.mu.at<double>(i, 1) - c2.mu.at<double>(i, 1);
    }
    
    for (int i = 0; i < qx.rows; i++) {
        //uso qx e qy come i sigmax del codice originale
        qx.at<double>(i) *= 2.0;
        qy.at<double>(i) *= 2.0;
    }
    
    cv::Mat iSigmaX = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    cv::Mat iSigmaY = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    for (int i = 0; i < qx.rows; i++)
    {
        iSigmaX.at<double>(i) = 1.0 / (qx.at<double>(i) + (qx.at<double>(i) == 0));
        iSigmaY.at<double>(i) = 1.0 / (qy.at<double>(i) + (qy.at<double>(i) == 0));
    }
    
    cv::Mat detSigmaX = cv::Mat::zeros(c1.sigmaX.size(), CV_64FC1);
    for (int i = 0; i < qx.rows; i++)
        detSigmaX.at<double>(i) = qx.at<double>(i) * qy.at<double>(i);
    
    cv::Mat z = cv::Mat::zeros(iSigmaX.size(), CV_64FC1);
    for (int i = 0; i < z.rows; i++)
        z.at<double>(i) = C2 / cv::sqrt(detSigmaX.at<double>(i)) * cv::exp(-0.5 * (iSigmaX.at<double>(i) * cv::pow(x.at<double>(i), 2.0) + iSigmaY.at<double>(i) * cv::pow(y.at<double>(i), 2.0)));
    
    cv::Mat dist = cv::Mat::zeros(z.size(), CV_64FC1);
    for (int i = 0; i < z.rows; i++)
        dist.at<double>(i) = q.at<double>(i)*Q.at<double>(i)*z.at<double>(i);
    
    cv::Mat s = cv::Mat::zeros(c1.histogram.size(), CV_64FC1);
    for (int i = 0; i < c1.histogram.rows; i++)
        s.at<double>(i) = cv::sqrt(c1.histogram.at<double>(i)) * cv::sqrt(c2.histogram.at<double>(i)) * dist.at<double>(i);
    
    double sumS = 0.0;
    for (int i = 0; i < s.rows; i++) {
        sumS += isnan(s.at<double>(i)) ? 0 : s.at<double>(i);
    }

    return sumS;
}
