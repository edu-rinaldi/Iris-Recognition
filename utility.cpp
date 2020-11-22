//
//  utility.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "utility.hpp"
SlidingWindow SlidingWindow::getSlidingWindow(const cv::Mat &img, const int &y, const int &x, const int &k)
{
    auto sw = SlidingWindow{};
    // colonna iniziale e finale
    sw.sx = x-k >= 0 ? x-k : 0;
    sw.ex = x+k <= img.cols ? x+k : img.cols;
    // riga iniziale e finale
    sw.sy = y-k >= 0 ? y-k : 0;
    sw.ey = y+k <= img.rows ? y+k : img.rows;
    return sw;
}


int noise(const cv::Mat& src)
{
    int count = 0;
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            if(src.at<uchar>(y, x) == 0) count++;
    
    return count;
}

std::string addStringToFileName(const std::string& fpath, const std::string& s)
{
    namespace fs = std::__fs::filesystem;
    
    fs::path path = fpath;

    std::string ext = path.extension();
    path.replace_extension("");
    std::string filename = path.filename();
    path.replace_filename(filename.append(s).append(ext));
    return path;
}


std::string getFileName(const std::string& fpath, const bool& withExstension)
{
    namespace fs = std::__fs::filesystem;
    
    auto fname = fs::path(fpath).filename();
    if(!withExstension) fname.replace_extension("");
    
    return fname;
}

std::string appendPath(const std::string& lpath, const std::string& rpath)
{
    namespace fs = std::__fs::filesystem;
    return fs::path(lpath).append(rpath);
}

void getPlainColorImg(const cv::Size& size, const cv::Scalar& scalar,cv::Mat& out)
{
    out = cv::Mat(size, CV_8UC3);
    out = scalar;
}

