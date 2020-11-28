//
//  isis_segmentation.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 08/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "isis_segmentation.hpp"

void isis::segmentation(const cv::Mat& source, segmentation_t& segmentation)
{
    
    // --- PREPROCESS -- (crop -> resize -> filtro i riflessi)
    cv::Mat sourceCropped = source;
    cv::Mat preprocessed;
    isis::preprocess(sourceCropped, preprocessed, SIMPLE_THRESHOLD);
    
    // --- SEGMENTATION ---
    
    // ricerca migliore limbo e crop su di esso per migliorare le performance
    auto limbus = findLimbusBF(preprocessed);
    auto limbusCroppesGray = preprocessed(limbus.getbbox());
    auto pupil = findPupilBF(limbusCroppesGray, limbus);
    
    
    // attenuazione ciglia
    cv::Mat eyelashSmoothed;
    eyelashSmoothing(preprocessed, eyelashSmoothed);
    cv::Mat gradientUp;
    gradientImage(eyelashSmoothed, gradientUp, true);
    
    // punti palpebra superiore
    auto upperEyelidPoints = findEyelidPoints(gradientUp, limbus, pupil, true);
    
    // normalizza immagine
    std::unordered_map<std::string, cv::Point> mapCartPol, mapPolCart;
    cv::Mat normalized;
    normalizeKrupicka(sourceCropped, normalized, limbus, pupil, mapCartPol, mapPolCart);
    
    // splitta i canali così da poter lavorare sul red-channel per la palpebra inf. e sul blue-channel per i riflessi
    std::vector<cv::Mat> normalizedBGR;
    cv::split(normalized, normalizedBGR);
    
    cv::Mat lowEyelidMask, reflectionMask;
    
    // maschera palpebra inferiore
    lowerEyelidMask(normalizedBGR[2], lowEyelidMask);
    
    // maschera riflessi
    reflectionsMask(normalizedBGR[0], reflectionMask);
    
    cv::Mat negMask, irisCroppedMask;
    // maschera negativa
    negativeMask(sourceCropped, negMask, limbus, pupil, upperEyelidPoints, lowEyelidMask, reflectionMask, mapCartPol);
    
    // maschera iride
    irisMask(sourceCropped, irisCroppedMask, limbus, pupil, negMask);
    
    cv::Mat normalizedMask;
    isis::normalizedMask(irisCroppedMask, normalizedMask, normalized.size(), mapCartPol);
    
    segmentation = {sourceCropped, normalized, normalizedMask, pupil, limbus};
}

void isis::preprocess(cv::Mat& source, cv::Mat& out, const int& reflectionFilterMode, const int& finalSize)
{
    // Cropping sull'occhio
    if(needClassifier(source))
        cropEye(source, source);
    
    // scala l'immagine per un miglioramento delle performance
    // TODO: forse va tolto, ma è molto utile su utiris
    scaleImage(source, source, finalSize);
    
    // tolgo i riflessi
    filterReflection(source, out, reflectionFilterMode);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
}

isis::Circle isis::findLimbusBF(const cv::Mat& grey)
{
    cv::Mat tmpColor;
    std::pair<Circle, double> bestLimbus = std::pair<Circle, double>(Circle(-1, cv::Point(0,0)), -10000);
    
    // Decommenta le successive due righe per usare il filtro di blur e sostituisci grey con blurred in posterization(..)
    // cv::Mat blurred;
    // cv::medianBlur(grey, blurred, 3);
    for(int k = 1; k < 18; k++)
    {
        cv::Mat posterized;
        posterization(grey, posterized, k);
        cv::cvtColor(posterized, tmpColor, cv::COLOR_GRAY2BGR);
        std::vector<Circle> circles;
        findCirclesTaubin(posterized, circles, 35, 90);
        // trova il migliore limbo
        auto bestKLimbus = findLimbus(grey, circles);

        if(bestLimbus.first.radius == -1 or bestLimbus.second < bestKLimbus.second) bestLimbus = bestKLimbus;
    }
    
    return bestLimbus.first;
}

isis::Circle isis::findLimbusBFAVG(const cv::Mat& grey)
{
    cv::Mat tmpColor, blurred;
    cv::medianBlur(grey, blurred, 3);
    
    Circle avgCircle;
    int tot = 0;
    for(int k = 1; k < 18; k++)
    {
        cv::Mat posterized;
        posterization(blurred, posterized, k);
        cv::cvtColor(posterized, tmpColor, cv::COLOR_GRAY2BGR);
        std::vector<Circle> circles;
        findCirclesTaubin(posterized, circles, 35, 90);
        // trova il migliore limbo
        auto bestKLimbus = findLimbus(grey, circles);

        if(avgCircle.radius == -1)
            avgCircle = bestKLimbus.first;
        else
        {
            avgCircle.center += bestKLimbus.first.center;
            avgCircle.radius += bestKLimbus.first.radius;
            tot++;
        }
    }
    
    avgCircle.center /= tot;
    avgCircle.radius /= tot;
    return avgCircle;
}

isis::Circle isis::findPupilBF(const cv::Mat& grey, const Circle& bestLimbus)
{
    auto bestPupil = std::pair<Circle, double>(Circle(-1, cv::Point(0,0)), -10000);
    auto centerCrop = cv::Point(grey.cols/2, grey.rows/2);
//    cv::Mat blurred;
//    cv::medianBlur(grey, blurred, 3);
    for(int k = 1; k < 18; k++)
    {
        cv::Mat posterized;
        posterization(grey, posterized, k);
        
        std::vector<Circle> circles;
        findCirclesTaubin(posterized, circles, 8, 90);
        //TODO: da ottimizzare
        //TODO: rivedere meglio le condizioni, non sembrano essere ottimali
        for(int i = (int)circles.size()-1; i>=0; i--)
        {
            if(circles[i].mean(grey) > 30.0 || !circles[i].inside(centerCrop.x, centerCrop.y) || distance(circles[i].center, centerCrop) > 15.0 || bestLimbus.radius/circles[i].radius < 3.0 || bestLimbus.radius/circles[i].radius > 5.5)
                circles.erase(circles.begin()+i);
        }
        
        auto bestKPupil = findPupil(grey, circles, bestLimbus);
        const auto tmp = bestKPupil.first;

        bestKPupil = { Circle(tmp.radius, cv::Point(tmp.center.x + bestLimbus.getbbox().x, tmp.center.y + bestLimbus.getbbox().y)), bestKPupil.second };

        if(bestPupil.first.radius == -1 || bestPupil.second < bestKPupil.second) bestPupil = bestKPupil;
    }
    
   
    return bestPupil.first;
}

isis::Circle isis::findPupilBFAVG(const cv::Mat& grey, const Circle& bestLimbus)
{
    auto centerCrop = cv::Point(grey.cols/2, grey.rows/2);
//    cv::Mat blurred;
//    cv::medianBlur(grey, blurred, 3);
    Circle avgCircle;
    int tot = 0;
    for(int k = 1; k < 18; k++)
    {
        cv::Mat posterized;
        posterization(grey, posterized, k);
        
        std::vector<Circle> circles;
        findCirclesTaubin(posterized, circles, 8, 90);
        
        for(int i = (int)circles.size()-1; i>=0; i--)
        {
            if(circles[i].mean(grey) > 30.0 || !circles[i].inside(centerCrop.x, centerCrop.y) || distance(circles[i].center, centerCrop) > 15.0 || bestLimbus.radius/circles[i].radius < 3.0 || bestLimbus.radius/circles[i].radius > 5.5)
                circles.erase(circles.begin()+i);
        }
        
        auto bestKPupil = findPupil(grey, circles, bestLimbus);
        const auto tmp = bestKPupil.first;

        bestKPupil = { Circle(tmp.radius, cv::Point(tmp.center.x + bestLimbus.getbbox().x, tmp.center.y + bestLimbus.getbbox().y)), bestKPupil.second };
        if(avgCircle.radius == -1) avgCircle = bestKPupil.first;
        else
        {
            tot++;
            avgCircle.center += bestKPupil.first.center;
            avgCircle.radius += bestKPupil.first.radius;
        }
    }
    avgCircle.center /= tot;
    avgCircle.radius /= tot;
    return avgCircle;
}

std::pair<isis::Circle, double> isis::findLimbus(const cv::Mat& mat, const std::vector<Circle>& circles)
{
    std::pair<Circle, double> bestCircle = std::pair<Circle, double>(Circle(-1, cv::Point(0,0)), -10000);
    for(const Circle& circle : circles)
    {
        // omogeneità
        double homogeneity = circle.homogeneity(mat);
    
        // separabilità
        double separability = circle.separability(mat);
        
        double score = homogeneity+separability;
        
        if(bestCircle.first.radius == -1 || bestCircle.second < score)
            bestCircle = {circle, score};
    }
    return bestCircle;
}

std::pair<isis::Circle, double> isis::findPupil(const cv::Mat& mat, std::vector<Circle>& circles, const Circle& limbus)
{
    std::pair<Circle, double> bestCircle = std::pair<Circle, double>(Circle(-1, cv::Point(0,0)), -10000);
    
    Circle defaultCircle = Circle(limbus.radius/4, cv::Point(mat.cols/2, mat.rows/2));
    circles.push_back(defaultCircle);
    for(const Circle& c : circles)
    {
        double homogeneity = c.homogeneity(mat);
        double separability = std::max(c.separability(mat), 0.0);
        double score = homogeneity + separability;
        if(score > bestCircle.second) bestCircle = {c, score};
    }
    if(bestCircle.first.radius == -1 && bestCircle.first.center.x == 0 && bestCircle.first.center.y == 0)
        bestCircle = {defaultCircle, defaultCircle.homogeneity(mat) + std::max(defaultCircle.separability(mat), 0.0)};
    
    return bestCircle;
}

void isis::findCirclesTaubin(const cv::Mat& mat, std::vector<Circle>& outputCircles, const double& minRadius, const double& maxRadius)
{
    outputCircles = std::vector<Circle>();
    std::vector<double> cth;
    isis::cannyThreshold(cth);
    for(const auto& cannyThreshold : cth)
    {
        cv::Mat cannyRes;
        canny(mat, cannyRes, cannyThreshold);
        
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(cannyRes, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
        
        for(std::vector<cv::Point>& contour : contours)
        {
            // controlla cerchio
            if(contour.size() <= 5) continue;
            Circle circle = Circle::taubin(contour);
            
            if(circle.inside(mat) && circle.radiusInside(minRadius, maxRadius)) outputCircles.push_back(circle);
        }
    }
}

void isis::findCirclesHough(const cv::Mat& mat, std::vector<Circle>& outputCircles, const double& minRadius, const double& maxRadius)
{
    outputCircles = std::vector<Circle>();
    auto tmpCircles = std::vector<cv::Vec3f>();
    std::vector<double> cth;
    isis::cannyThreshold(cth);
    for(const auto& cannyThreshold : cth)
    {
        cv::Mat cannyRes;
        canny(mat, cannyRes, cannyThreshold);
        
        cv::HoughCircles(cannyRes, tmpCircles, cv::HOUGH_GRADIENT, 1, mat.rows / 15., 50, 30, minRadius, maxRadius);
    }
    for(const cv::Vec3f& c : tmpCircles)
        outputCircles.push_back(Circle(c[2], {(int)c[0], (int)c[1]}));
}

void isis::canny(const cv::Mat& src, cv::Mat& out, double cannyThreshold)
{
    cv::Mat blurred;
    src.copyTo(blurred);
    
    // Come consigliato dalla letteratura, viene applicato un filtro
    // di blurring 3x3 alla matrice su cui effettuare il Canny
    if(BLURRING_CANNY) cv::medianBlur(blurred, blurred, 3);

    if(HIST_EQUALIZING_CANNY) cv::equalizeHist(blurred, blurred);
    
    cv::Canny(blurred, out, cannyThreshold, cannyThreshold * 3, CANNY_APERTURE);
}

void isis::cannyThreshold(std::vector<double>& out)
{
    out = std::vector<double>();
    for(int i=0; i<11; i++)
        out.push_back(0.05 + (0.05 * i));
}

std::vector<cv::Point> isis::findEyelidPoints(const cv::Mat& mat, const Circle& limbus, const Circle& pupil, const bool& up)
{
    auto points = std::vector<cv::Point>();
    
    int startCol = (int) (limbus.center.x - limbus.radius);
    int endCol = (int) (limbus.center.x + limbus.radius);
    
    int upperRow = up ? (int)(limbus.center.y - limbus.radius) : (int)(pupil.center.y + pupil.radius);
    int lowerRow = up ? (int)(pupil.center.y - pupil.radius) : (int)(limbus.center.y + limbus.radius);
    
    for(int col = startCol; col <= endCol; col++)
    {
        double maxValue = 0; int targetRow = 0;
        for(int row = lowerRow; row > upperRow; row--)
        {
            if(inside(mat.size(), cv::Point(col, row)))
            {
                double val = mat.at<uchar>(row, col);
                if(val > maxValue && ((up && row < pupil.center.y - pupil.radius * 1.5) || (!up && row > pupil.center.y + pupil.radius * 1.5)))
                {
                    maxValue = val;
                    targetRow = row;
                }
            }
        }
        points.push_back(cv::Point(col, targetRow));
    }
    return points;
}

void isis::normalizeKrupicka(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, std::unordered_map<std::string, cv::Point>& mapCartPol, std::unordered_map<std::string, cv::Point>& mapPolCart)
{
    //clear map
    mapCartPol = std::unordered_map<std::string, cv::Point>();
    
    int h = limbus.diameter();
    int w = std::round(limbus.diameter() * M_PI);
    out = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
    double thetaStep = (2. * M_PI) / w;
    double xp, yp, xl, yl;
    int x, y, ind = 0;
    for(double i = 3. * M_PI / 2.; i < 2. * M_PI + 3. * M_PI/2.; i+= thetaStep)
    {
        xp = pupil.center.x + pupil.radius * std::cos(i);
        yp = pupil.center.y + pupil.radius * std::sin(i);
        xl = limbus.center.x + limbus.radius * std::cos(i);
        yl = limbus.center.y + limbus.radius * std::sin(i);
        for(int j = 0; j < h; ++j)
        {
            double pas = (double)j / h;
            x = std::round((1. - pas) * xl + pas * xp);
            y = std::round((1. - pas) * yl + pas * yp);
            if(inside(src.size(), cv::Point(x, y)))
            {
                out.at<cv::Vec3b>(j, ind) = src.at<cv::Vec3b>(y, x);
                mapCartPol[std::to_string(ind)+","+std::to_string(j)] = cv::Point(x,y);
                mapPolCart[std::to_string(x)+","+std::to_string(y)] = cv::Point(ind, j);
            }
        }
        ind++;
    }
}

void isis::lowerEyelidMask(const cv::Mat& normalizedRedChannel, cv::Mat& lowerEyelidMask)
{
    lowerEyelidMask = cv::Mat(normalizedRedChannel.size(), CV_8UC1, cv::Scalar(255));
    
    auto meanArray = std::vector<double>(), stdDevArray = std::vector<double>();
    
    cv::Mat mask = cv::Mat::zeros(normalizedRedChannel.size(),CV_8UC1);
    
    for (int y = 0; y <= normalizedRedChannel.rows/2; ++y)
        for (int x = normalizedRedChannel.cols/4; x <= (3 * normalizedRedChannel.cols)/4; ++x)
            mask.at<int>(y, x) = 1;
    
    cv::meanStdDev(normalizedRedChannel, meanArray, stdDevArray, mask);
    double mean = meanArray[0], stdDev = stdDevArray[0];
    int threshold = (int)(mean+stdDev);
    
    if(stdDev > mean/4)
    {
        for (int y = 0; y < normalizedRedChannel.rows; ++y)
            for (int x = 0; x < normalizedRedChannel.cols; ++x)
                if(normalizedRedChannel.at<uchar>(y, x) > threshold)
                    lowerEyelidMask.at<uchar>(y, x) = 0;
    }
}

void isis::negativeMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const std::vector<cv::Point>& upperEyelidPoints, const cv::Mat& lowerEyelidMask, const cv::Mat& reflectionsMask, const std::unordered_map<std::string, cv::Point>& map)
{
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    
    for(int y = 0; y < lowerEyelidMask.rows; y++)
        for(int x = 0; x < lowerEyelidMask.cols; x++)
            if(lowerEyelidMask.at<uchar>(y, x) == 0 || reflectionsMask.at<uchar>(y, x) != 0)
                out.at<uchar>(map.at(std::to_string(x)+","+std::to_string(y))) = 255;
    
    for(int x = 0; x < src.cols; x++)
    {
        int upperEyelidEdge = 0;
        for(const auto& p : upperEyelidPoints)
            if(p.x == x) upperEyelidEdge = p.y;
        
        for(int y = 0; y < src.rows; y++)
            if(limbus.inside(x, y) && (y < upperEyelidEdge)) out.at<uchar>(y, x) = 255;
    }
}

void isis::irisMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const cv::Mat& negMask)
{
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    
    cv::circle(out, limbus.center, (int) limbus.radius - 1, cv::Scalar(255), -1);
    cv::circle(out, pupil.center, (int) pupil.radius, cv::Scalar(0), -1);
    
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            if(negMask.at<uchar>(y, x) > 0.) out.at<uchar>(y, x) = 0;
}

void isis::normalizedMask(const cv::Mat& irisCroppedMask, cv::Mat& mask, const cv::Size& normalizedSize, const std::unordered_map<std::string, cv::Point>& mapCartPol)
{
    mask = cv::Mat::zeros(normalizedSize, CV_8UC1);
    for(int y = 0; y < mask.rows; y++)
        for(int x = 0; x < mask.cols; x++)
            if((int)irisCroppedMask.at<uchar>(mapCartPol.at(std::to_string(x)+","+std::to_string(y))) == 255)
                mask.at<uchar>(y, x) = 255;
}

