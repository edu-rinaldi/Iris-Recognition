//
//  preprocessing.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//
#include "settings.h"
#include "preprocessing.hpp"

void isis::posterization(const cv::Mat& src, cv::Mat& out, const int& k)
{
    out = cv::Mat::zeros(src.rows, src.cols, src.type());
    int x = 0;
    
    // Per ogni riga
    for(int y=0; y < src.rows; y++)
    {
        // Inizializzo un nuovo histogram e una nuova sliding window grande (k*2+1)^2
        auto histo = std::unordered_map<uchar, int>();
        SlidingWindow slidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);
        
        // topcolor è il colore più frequente
        int topColor = -1;
        
        // calcola histogram nella finestra
        for(int ty = slidingWindow.sy; ty < slidingWindow.ey; ty++)
        {
            for(int tx = slidingWindow.sx; tx < slidingWindow.ex; tx++)
            {
                int colore = src.at<uchar>(ty, tx);
                if(topColor == -1) topColor = colore;
                histo[colore] += 1;
                if(histo[topColor] < histo[colore]) topColor = colore;
            }
        }
        
        // Setta il colore del pixel nella prima colonna dell'immagine con quello più frequente
        out.at<uchar>(y, x) = topColor;
        
        // Per le altre colonne
        for(int x = 1; x < src.cols; x++)
        {
            auto newSlidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);
            // Scorri per ogni riga della sliding window
            for(int ty = slidingWindow.sy; ty < slidingWindow.ey; ty++)
            {
                // Elimina dall'histogram le occorrenze dei pixel della prima colonna della sliding window
                histo[src.at<uchar>(ty, slidingWindow.sx)] -= 1;
                
                // fai scorrere la finestra a destra
//                slidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);
                
                // Prendi il colore nell'ultima colonna e aggiorna l'histogram
                auto color_tmp = src.at<uchar>(ty, newSlidingWindow.ex-1);
                histo[color_tmp] += 1;
                // Aggiorna il top_color
                if(histo[topColor] < histo[color_tmp]) topColor = color_tmp;
            }
            slidingWindow = newSlidingWindow;
            // Assegna il top color
            out.at<uchar>(y, x) = topColor;
        }
    }
}

void isis::filterReflectionST(const cv::Mat& src, cv::Mat& mask)
{
    // Pixel piu acceso
    double brightestPixel;
    cv::minMaxLoc(src, nullptr, &brightestPixel);
    
    
    // Crea maschera
    cv::threshold(src, mask, brightestPixel*0.93, brightestPixel, cv::THRESH_BINARY);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
}

void isis::filterReflectionAT(const cv::Mat& src, cv::Mat& mask)
{
    cv::adaptiveThreshold(src, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
}

void isis::filterReflection(const cv::Mat& src, cv::Mat& out, const int& mode)
{
    cv::Mat mask = cv::Mat(src.size(), CV_8UC1);
    switch (mode)
    {
        case SIMPLE_THRESHOLD:
        {
            cv::Mat greySrc;
            cv::cvtColor(src, greySrc, cv::COLOR_BGR2GRAY);
            filterReflectionST(greySrc, mask);
            break;
        }
        case ADAPTIVE_THRESHOLD:
        {
            std::vector<cv::Mat> channels(3);
            cv::split(src, channels);
            filterReflectionAT(channels[0], mask);
            break;
        }
        default:
        {
            std::cerr<<"Non è stato selezionato nessun metodo di filtraggio"<<std::endl;
            break;
        }
    }
    cv::inpaint(src, mask, out, 5, cv::INPAINT_TELEA);
}

void isis::cropEye(const cv::Mat& src, cv::Mat& out, const std::string& classifierPath)
{
    auto eyeCascadeClassifier = cv::CascadeClassifier(classifierPath);
    auto eyes = std::vector<cv::Rect>();
    eyeCascadeClassifier.detectMultiScale(src, eyes);
    
    if(eyes.size() == 0) return;
    
    // Prendo l'occhio con l'area più grande
    cv::Rect eye = *std::max_element(eyes.begin(), eyes.end(), [](const cv::Rect& l, const cv::Rect& r){ return l.area() < r.area(); });
    out = eye.width < 120 ? src : src(eye);
}

void isis::blurFilter(const cv::Mat& mat, cv::Mat& out, const int& windowSize, const int& n)
{
    out = cv::Mat(mat.size(), mat.type());
    for(int y=0; y<mat.rows; y++)
        for(int x=0; x<mat.cols; x++)
        {
            std::vector<double> list = std::vector<double>();
            for(int k = -windowSize; k<=windowSize; k++)
                if(inside(mat.size(), cv::Point(x+k, y)))
                    list.push_back(mat.at<uchar>(y, x+k));
            std::sort(list.begin(), list.end());
            out.at<uchar>(y, x) = list[(int)(list.size()/2)+1];
        }
}

void isis::gradientImage(const cv::Mat& src, cv::Mat& out, const bool& up)
{
    cv::Mat kernel = cv::Mat();
    for(int y=0; y < 10; y++)
    {
        double v = 0;
        if(y > 3 && y < 6) v = 0;
        else if(y < 4) v = up ? 1 : -1;
        else if(y > 5) v = up ? -1 : 1;
        double tmp[] = {v, v, v, v, v};
        kernel.push_back(cv::Mat(1,5, CV_64F, tmp));
    }
    cv::filter2D(src, out, -1, kernel, cv::Point(-1,-1));
}

bool isis::scaleImage(const cv::Mat& src, cv::Mat& out, const int& finalSize)
{
    if(src.cols > finalSize || src.rows > finalSize)
    {
        double sf =  (double)finalSize/std::max(src.cols, src.rows);
        cv::resize(src, out, cv::Size((int)src.size().width * sf, (int)src.size().height * sf));
    }
    return src.cols > finalSize || src.rows > finalSize;
}


bool isis::isSkinPixel(const cv::Mat& bgr, const cv::Mat& hsv, const cv::Mat& ycrcb, const int& y, const int& x)
{
    
    auto bgrPx = bgr.at<cv::Vec3b>(y, x);
    auto hsvPx = hsv.at<cv::Vec3b>(y, x);
    auto ycrcbPx = ycrcb.at<cv::Vec3b>(y, x);
    
    bool first = hsvPx[0] >= 0 && hsvPx[0] <= 25 && hsvPx[1] >= 58.88 && hsvPx[1] <= 174.08 && bgrPx[2] > 95 && bgrPx[1] > 40 && bgrPx[0] > 20 && bgrPx[2] > bgrPx[1] && bgrPx[2] > bgrPx[0] && cv::abs(bgrPx[2] - bgrPx[1]) > 15;
    
    bool second = bgrPx[2] > 95 && bgrPx[1] > 40 && bgrPx[0] > 20 && bgrPx[2] > bgrPx[1] && bgrPx[2] > bgrPx[0] && cv::abs(bgrPx[2] - bgrPx[1]) > 15 && ycrcbPx[1] > 135 && ycrcbPx[2] > 85 && ycrcbPx[0] > 80 && ycrcbPx[1] <= (1.5862*ycrcbPx[2])+20 && ycrcbPx[1] >= (0.3448*ycrcbPx[2])+76.2069 && ycrcbPx[1] >= (-4.5652*ycrcbPx[2])+234.5652 && ycrcbPx[1] <= (-1.15*ycrcbPx[2])+301.75 && ycrcbPx[1] <= (-2.2857*ycrcbPx[2])+432.85;
    
    return first || second;
}

int isis::getComponent(const cv::Mat& blue, const cv::Mat& white, cv::Mat& mask, const int& y, const int& x)
{
    int count = 0;
    auto q = std::queue<cv::Point>();
    cv::Mat eval = cv::Mat::zeros(blue.size(), CV_8UC1);
//    const cv::Point dirs[] = { {0,1}, {1,0}, {1,1}, {0, -1}, {-1, 0}, {-1,-1}, {1, -1}, {-1, 1}};
    const cv::Point dirs[] = { {0,1}, {1,0}, {0, -1}, {-1, 0} };
    q.push({x, y});
    
    while(!q.empty())
    {
        cv::Point current = q.front();
        q.pop();
        
        mask.at<uchar>(current) = 255;
        eval.at<uchar>(current) = 255;
        count++;
        for(const cv::Point& dir : dirs)
        {
            cv::Point newP = current+dir;
            auto alreadySeen = eval.at<uchar>(newP) == 255;
            eval.at<uchar>(newP) = 255;
            if(!inside(mask.size(), newP) or mask.at<uchar>(newP) == 255 or alreadySeen or (white.at<uchar>(newP) == 0 and blue.at<uchar>(newP) == 0))
                continue;
            q.push(newP);
        }
    }
    return count;
}

void isis::getColorDistribution(const cv::Mat& hueSrc, cv::Mat& outRGB, const std::vector<cv::Vec3b>& colors)
{
    int binSize = 180/colors.size();
    outRGB = cv::Mat(hueSrc.size(), CV_8UC3);
    for(int y=0; y<hueSrc.rows; y++)
        for(int x=0; x<hueSrc.cols; x++)
            outRGB.at<cv::Vec3b>(y, x) = colors[hueSrc.at<uchar>(y, x) / binSize];
}

bool isis::needClassifier(const cv::Mat& src)
{
    cv::Mat hsvSrc, ycrcbSrc, skinMask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<cv::Mat> hsvChannels;
    // Passa da bgr as hsv
    
    cv::cvtColor(src, hsvSrc, cv::COLOR_BGR2HSV);
    cv::cvtColor(src, ycrcbSrc, cv::COLOR_BGR2YCrCb);
    
    // Dividi in canali
    cv::split(hsvSrc, hsvChannels);
    cv::Mat histogram = cv::Mat::zeros(256, 1, CV_32FC1);
    
    
    
    // --- FILTRAGGIO CONTEGGI SBAGLIATI ---
    
    
    // calcola una maschera per i punti scuri dell'immagine
    cv::Mat blackMask;
    cv::inRange(hsvSrc, cv::Scalar(0, 0, 0, 0), cv::Scalar(180, 255, 30, 0), blackMask);
    
    // calcola una maschera per i riflessi
    cv::Mat maskReflection, srcGray;
    cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);
    isis::filterReflectionST(srcGray, maskReflection);
    
    // Qui si possono mettere altre maschere in OR con maskReflection e blackMask
    
    //blue bin mask
    cv::Mat blueMask;
    cv::inRange(hsvChannels[0], cv::Scalar(90), cv::Scalar(120), blueMask);
    
    // Viene creato il primo filtro
    cv::Mat firstFilter;
    // Creo una maschera unica
    cv::bitwise_or(blackMask, maskReflection, firstFilter);
    
    // Accendo solo i pixel sulla tonalità blu
    cv::bitwise_and(firstFilter, blueMask, firstFilter);
    // Nego la maschera
    cv::bitwise_not(firstFilter, firstFilter);
    
    
    /*
     ---- CALCOLO ISTOGRAMMA HUE VAL. ----
        0 -> Orange 0-29
        1 -> Yellow 30-59
        2 -> Green 60-89
        3 -> Blue 90-119
        4 -> Violet 120-149
        5 -> Red 150-179
    */
    std::vector<cv::Vec3b> colors = {{0,69,255}, {0,255,255}, {0,255,0}, {255,0,0}, {238,180,238}, {0,0,255}};
    
    int histSize = 6;
    float hrange[] = {0, 180};
    const float* ranges = {hrange};
    cv::calcHist(&hsvChannels[0], 1, 0, firstFilter, histogram, 1, &histSize, &ranges);
    
    // Crea un'immagine che dia l'idea della distribuzione delle varie fasce di colori
    cv::Mat colorDistribution;
    getColorDistribution(hsvChannels[0], colorDistribution, colors);
    
    for(int y=0; y<hsvChannels[0].rows; y++)
        for(int x=0; x<hsvChannels[0].cols; x++)
            // Pelle umana
            if(isSkinPixel(src, hsvSrc, ycrcbSrc, y, x)) skinMask.at<uchar>(y, x) = 255;
    
    // Percentuale di blu nell'immagine
    float bluePercentage = histogram.at<float>(3)/hsvChannels[0].total();
    // Non ha bisogno di un crop
    if(bluePercentage >= 0.03)
        return false;
    
    
    // Mi trovo la maschera del colore bianco (per la sclera)
    cv::Mat whiteMask, compMask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::inRange(hsvSrc, cv::Scalar(0, 0, 90,0), cv::Scalar(179, 90, 255,0), whiteMask);
    
//    cv::Mat tmp;
//    cv::bitwise_not(skinMask, tmp);
//    cv::bitwise_and(whiteMask, tmp, whiteMask);
    int compSize = 0;
    for(int y=0; y<whiteMask.rows; y++)
        for(int x=0; x<whiteMask.cols; x++)
            if(blueMask.at<uchar>(y, x) == 255 && whiteMask.at<uchar>(y, x) == 255 && compMask.at<uchar>(y, x) == 0)
                compSize += getComponent(blueMask, whiteMask, compMask, y, x);
    
    return compSize/src.total() < 0.04;
}
