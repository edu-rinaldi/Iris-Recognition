//
//  preprocessing.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

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
        
        // Setta il colore del pixel nella prima colonna con quello più frequente
        out.at<uchar>(y, x) = topColor;
        
        // Per le altre colonne
        for(int x = 0; x < src.cols; x++)
        {
            // Se è il top left pixel skippa (ho già il suo valore)
            if(y == 0 && x == 0) continue;
            // Scorri per ogni riga della sliding window
            for(int ty = slidingWindow.sy; ty < slidingWindow.ey; ty++)
            {
                // Elimina dall'histogram le occorrenze dei pixel della prima colonna della sliding window
                histo[src.at<uchar>(ty, slidingWindow.sx)] -= 1;
                
                // fai scorrere la finestra a destra
                slidingWindow = SlidingWindow::getSlidingWindow(src, y, x, k);
                
                // Prendi il colore nell'ultima colonna e aggiorna l'histogram
                auto color_tmp = src.at<uchar>(ty, slidingWindow.ex-1);
                histo[color_tmp] += 1;
                // Aggiorna il top_color
                if(histo[topColor] < histo[color_tmp]) topColor = color_tmp;
            }
            // Assegna il top color
            out.at<uchar>(y, x) = topColor;
        }
    }
}

void isis::filterReflectionST(const cv::Mat& src, cv::Mat& out)
{
    // Pixel piu acceso
    double brightestPixel;
    cv::minMaxLoc(src, nullptr, &brightestPixel);
    
    // Inizializzo maschera
    cv::Mat mask;
    
    // Inpaint
    cv::threshold(src, mask, brightestPixel*0.93, brightestPixel, cv::THRESH_BINARY);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
    cv::inpaint(src, mask, out, 5, cv::INPAINT_TELEA);
}

void isis::filterReflectionAT(const cv::Mat& src, cv::Mat& out)
{
    cv::Mat mask;
    cv::adaptiveThreshold(src, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
    cv::inpaint(src, mask, out, 5, cv::INPAINT_TELEA);
}

void isis::filterReflection(const cv::Mat& src, cv::Mat& out, const int& mode)
{
    switch (mode)
    {
        case SIMPLE_THRESHOLD:
        {
            cv::Mat tmp;
            cv::cvtColor(src, tmp, cv::COLOR_RGB2GRAY);
            filterReflectionST(tmp, out);
            break;
        }
        case ADAPTIVE_THRESHOLD:
        {
            std::vector<cv::Mat> channels(3);
            cv::split(src, channels);
            filterReflectionAT(channels[0], out);
            break;
        }
        default:
        {
            std::cerr<<"Non è stato selezionato nessun metodo di filtraggio"<<std::endl;
            break;
        }
    }
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
