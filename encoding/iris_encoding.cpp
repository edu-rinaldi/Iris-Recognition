//
//  iris_encoding.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 10/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "iris_encoding.hpp"

/*
    Funzione utilizzata per LBP.
    Calcola il valore del pixel in base ai vicini (viene utilizzata la sommatoria sugli 8 pixel vicini)
 */
int isis::getCalculatedPixel(const cv::Mat& src, const int& y, const int& x)
{
    int center = src.at<uchar>(y, x);
    auto valList = std::vector<int>();
    
    const cv::Point dirs[] = {{1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}};
    for(int i=0; i<8; i++)
    {
        auto dir = dirs[i];
        cv::Point newPoint = {cv::Point(x,y)+dir};
        if(inside(src.size(), newPoint))
            valList.push_back((int)src.at<uchar>(newPoint) >= center ? 1 : 0);
        else valList.push_back(0);
    }
    int val = 0;
    for(int i=0; i<valList.size(); i++) val += valList[i] * cv::pow(2, i);
    
    return val;
}

/*
    Funzione che applica l'operatore LBP
    @param src: immagine in input su cui applicare LBP
    @param out: immagine su cui vogliamo mettere l'output ottenuto applicando LBP.
 */
void isis::applyLBP(const cv::Mat& src, cv::Mat& out)
{
    cv::Mat tmp;
    cv::cvtColor(src, tmp, cv::COLOR_BGR2GRAY);
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            out.at<uchar>(y, x) = getCalculatedPixel(tmp, y, x);
}

/*
    Effettua la codifica LBP.
    @param out: struct lbp vuota, in questa verrà messo l'output della funzione
    @param s: struct contenente la segmentazione dell'iride da codificare
 */
void isis::encodeLBP(lbp_t& out, const segmentation_t& s)
{
    applyLBP(s.normalized, out.img);
    
    int subRows = out.img.rows / NUM_ZONE;
    for(int zone = 0; zone < NUM_ZONE; zone++)
    {
        out.histograms[zone] = cv::Mat::zeros(256, 1, CV_32FC1);
        
        //da 0 a 255
        int histSize = 256;
        
        float range[] = { 0, 256 }; // range esclusivo
        const float* histRange = { range };
        cv::Mat currentZone = out.img(cv::Range(zone * subRows, (zone+1) * subRows), cv::Range(0, out.img.cols));
        out.zones[zone] = s.mask(cv::Range(zone * subRows, (zone+1) * subRows), cv::Range(0, s.mask.cols));
        cv::calcHist(&currentZone, 1, 0, cv::Mat(), out.histograms[zone] , 1, &histSize, &histRange);
    }
}

/*
    Date due codifiche restituisce il valore di similarità
 */
double isis::matchLBP(const lbp_t& c1, const lbp_t& c2)
{
    double score = 0;
    cv::Mat norm_hist1 = cv::Mat::zeros(256, 1, CV_32FC1);
    cv::Mat norm_hist2 = cv::Mat::zeros(256, 1, CV_32FC1);
    for(int zone = 0; zone < NUM_ZONE; zone++)
    {
        cv::normalize(c1.histograms[zone], norm_hist1);
        cv::normalize(c2.histograms[zone], norm_hist2);
        double noisei = noise(c1.zones[zone])+noise(c2.zones[zone]);
        score += cv::compareHist(c1.histograms[zone], c2.histograms[zone], cv::HISTCMP_BHATTACHARYYA) * (1 - noisei/(double)(2*c1.zones[zone].total()));
    }
    
    return score / NUM_ZONE;
}

/*
    Effettua la codifica spatiogram
    @param out: struct spatiogram vuota, in questa verrà messo l'output della funzione
    @param s: struct contenente la segmentazione dell'iride da codificare
 */
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
            kDist = cv::Mat::ones(img.rows, img.cols, CV_32FC1) / img.total();
    
    float xf = 2.0f / (s.normalized.cols - 1);
    float yf = 2.0f / (s.normalized.rows - 1);

    
    for (int i = 0; i < binno.rows; i++)
    {
        float value = -1;
        for (int j = 0; j < binno.cols; j++)
        {
            gridX.at<float>(i, j) = value;
            value += xf;
        }
    }
        
    for (int j = 0; j < binno.cols; j++)
    {
        float value = -1;
        for (int i = 0; i < binno.rows; i++)
        {
            gridY.at<float>(i, j) = value;
            value += yf;
        }
    }

        
    for (int i = 0; i < kDist.rows; i++)
        for (int j = 0; j < kDist.cols; j++)
            kDist.at<float>(i, j) *= (double)s.mask.at<uchar>(i, j);
    
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
        out.sigmaY.at<double>(i) -= cv::pow(out.mu.at<double>(i, 1) / wsum.at<double>(i), 2.0);
        out.sigmaY.at<double>(i) += (minMK - out.sigmaY.at<double>(i)) * (out.sigmaY.at<double>(i) < minMK);
    }
    
    // Normalizzazione meanVector
    for (int i = 0; i < out.mu.rows; i++)
    {
        out.mu.at<double>(i, 0) /= wsum.at<double>(i);
        out.mu.at<double>(i, 1) /= wsum.at<double>(i);
    }
    
}

/*
    Date due codifiche spatiogram restituisce un valore di dissimilarità
 */
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
    for (int i = 0; i < s.rows; i++)
        sumS += isnan(s.at<double>(i)) ? 0 : s.at<double>(i);

    return sumS;
}
