//
//  preprocessing.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//
#include "settings.h"
#include "preprocessing.hpp"

/*
    Data un'immagine in input restituisce l'immagine posterizzata con una finestra grande (k*2+1) x (k*2+1).
    @param src: immagine in scala di grigi da posterizzare
    @param out: immagine posterizzata
    @param k: parametro per decidere la grandezza della finestra
 */
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

/*
    Crea la maschera che individua i riflessi tramite il Simple Thresholding.
    @param src: immagine su cui cercare i riflessi
    @param mask: maschera contenente l'output
 */
void isis::filterReflectionST(const cv::Mat& src, cv::Mat& mask)
{
    // Pixel piu acceso
    double brightestPixel;
    cv::minMaxLoc(src, nullptr, &brightestPixel);
    
    
    // Crea maschera
    cv::threshold(src, mask, brightestPixel*0.93, brightestPixel, cv::THRESH_BINARY);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
}

/*
    Crea la maschera che individua i riflessi tramite l'Adaptive Thresholding.
    @param src: immagine su cui cercare i riflessi.
    @param mask: maschera contenente l'output.
 */
void isis::filterReflectionAT(const cv::Mat& src, cv::Mat& mask)
{
    cv::adaptiveThreshold(src, mask, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10);
    cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(7,7)));
}

/*
    Individua e corregge i riflessi tramite un algoritmo di inpainting.
    @param src: immagine su cui correggere i riflessi
    @param out: immagine corretta
    @param mode: metodo di thresholding da utilizzare
 */
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

/*
    Effettua il cropping dell'immagine sull'area di interesse utilizzando un classificatore HAARCASCADE.
    @param src: immagine da croppare
    @param out: immagine in output
 */
void isis::cropEye(const cv::Mat& src, cv::Mat& out)
{
    auto eyes = isis::getROI(src, isis::HAARCASCADE_EYE_TREE_EYEGLASSES);
    
    if(eyes.size() == 0) return;
    
    // Prendo l'occhio con l'area più grande
    cv::Rect eye = *std::max_element(eyes.begin(), eyes.end(), [](const cv::Rect& l, const cv::Rect& r){ return l.area() < r.area(); });
    out = eye.width < 120 ? src : src(eye);
}

/*
    Individua una lista di rettangoli. Ogni rettangolo contiene l'area di interesse (un occhio)
    @param src: immagine in cui cercare
    @param classifierPath: percorso dove è situato il modello del classificatore.
    
    @return una lista di rettangoli corrispondenti alle varie possibili aree di interesse.
 */
std::vector<cv::Rect> isis::getROI(const cv::Mat& src, const std::string& classifierPath)
{
    auto eyeCascadeClassifier = cv::CascadeClassifier(classifierPath);
    auto eyes = std::vector<cv::Rect>();
    eyeCascadeClassifier.detectMultiScale(src, eyes);
    return eyes;
}

/*
    Applica un filtro di blurring all'immagine in input
    @param mat: immagine su cui applicare il filtro
    @param out: immagine in cui viene messo l'output
    @param windowSize: grandezza della finestra
 */
void isis::blurFilter(const cv::Mat& mat, cv::Mat& out, const int& windowSize)
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

/*
    Restituisce una nuova matrice costituente l'immagine del gradiente, in cui il valore
    di intensità nei vari punti rappresenta il gradiente con direzione verticale in un
    certo punto dell'immagine originale.
    
    @param src: matrice su cui calcolare l'immagine del gradiente.
    @param out: matrice costituita dall'immagine del gradiente della matrice in input.
    @param up: indica se ci si riferisce o meno alla palpebra superiore.
*/
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

/*
    Restituisce un'immagine scalata con grandezza massima pari a finalSize.
    L'immagine viene scalata mantenendo l'aspect ratio.
    Quindi se abbiamo un immagine grande 1280x720 (16:9), e finalSize=500,
    l'immagine in output sarà grande: 500x280.
 
    @param src: immagine da scalare
    @param out: immagine in output scalata
    @param finalSize: grandezza massima del lato più grande dell'immagine.
 */
bool isis::scaleImage(const cv::Mat& src, cv::Mat& out, const int& finalSize)
{
    if(src.cols > finalSize || src.rows > finalSize)
    {
        double sf =  (double)finalSize/std::max(src.cols, src.rows);
        cv::resize(src, out, cv::Size((int)src.size().width * sf, (int)src.size().height * sf));
    }
    return src.cols > finalSize || src.rows > finalSize;
}

/*
    Dato un certo pixel ci dice se possiamo considerarlo come pixel appartenente alla pelle o meno.
    
    @param bgr: immagine in formato bgr.
    @param hsv: immagine in formato hsv.
    @param ycrcb: immagine in formato ycrcb.
    @param y: coordinata y del pixel.
    @param x: coordinata x del pixel.
 
    @return true se è un pixel della pelle, false altrimenti
 */
bool isis::isSkinPixel(const cv::Mat& bgr, const cv::Mat& hsv, const cv::Mat& ycrcb, const int& y, const int& x)
{
    
    auto bgrPx = bgr.at<cv::Vec3b>(y, x);
    auto hsvPx = hsv.at<cv::Vec3b>(y, x);
    auto ycrcbPx = ycrcb.at<cv::Vec3b>(y, x);
    
    bool first = hsvPx[0] >= 0 && hsvPx[0] <= 25 && hsvPx[1] >= 58.88 && hsvPx[1] <= 174.08 && bgrPx[2] > 95 && bgrPx[1] > 40 && bgrPx[0] > 20 && bgrPx[2] > bgrPx[1] && bgrPx[2] > bgrPx[0] && cv::abs(bgrPx[2] - bgrPx[1]) > 15;
    
    bool second = bgrPx[2] > 95 && bgrPx[1] > 40 && bgrPx[0] > 20 && bgrPx[2] > bgrPx[1] && bgrPx[2] > bgrPx[0] && cv::abs(bgrPx[2] - bgrPx[1]) > 15 && ycrcbPx[1] > 135 && ycrcbPx[2] > 85 && ycrcbPx[0] > 80 && ycrcbPx[1] <= (1.5862*ycrcbPx[2])+20 && ycrcbPx[1] >= (0.3448*ycrcbPx[2])+76.2069 && ycrcbPx[1] >= (-4.5652*ycrcbPx[2])+234.5652 && ycrcbPx[1] <= (-1.15*ycrcbPx[2])+301.75 && ycrcbPx[1] <= (-2.2857*ycrcbPx[2])+432.85;
    
    return first || second;
}

/*
    Data la maschera dei pixel con tonalità blu, la maschera dei pixel tendenti al bianco,
    accende nella maschera mask i pixel di una componente individuata a partire dal pixel acceso nella
    maschera blue a coordinate (x,y).
    Restituisce inoltre il numero di pixel che contiene la componente appena individuata.
 
    @param blue: maschera che ha come pixel accesi quelli con tonalità blu.
    @param white: maschera che individua i pixel tendenti al bianco.
    @param mask: maschera in cui accendere i pixel relativi alle componenti.
    @param y: coordinata y del pixel da cui partire per il calcolo della componente.
    @param x: coordinata x del pixel da cui partire per il calcolo della componente.
 
    @return numero di pixel accesi nella nuova componente individuata
 */
int isis::getComponent(const cv::Mat& blue, const cv::Mat& white, cv::Mat& mask, const int& y, const int& x)
{
    int count = 0;
    
    // coda dei pixel da esaminare
    auto q = std::queue<cv::Point>();
    
    // maschera dei pixel controllati
    cv::Mat eval = cv::Mat::zeros(blue.size(), CV_8UC1);
    
    const cv::Point dirs[] = { {0,1}, {1,0}, {0, -1}, {-1, 0} };
    // Decommentare la riga sotto per imporre come adiacenti anche i pixel sulle diagonali
    //const cv::Point dirs[] = { {0,1}, {1,0}, {1,1}, {0, -1}, {-1, 0}, {-1,-1}, {1, -1}, {-1, 1}};
    
    // controlla il pixel da cui partiamo
    q.push({x, y});
    
    // BFS
    while(!q.empty())
    {
        cv::Point current = q.front();
        q.pop();
        
        // accendi il pixel in mask
        mask.at<uchar>(current) = 255;
        // segna che l'hai controllato
        eval.at<uchar>(current) = 255;
        // aumenta il numero di pixel accesi
        count++;
        
        // per ogni suo pixel vicino
        for(const cv::Point& dir : dirs)
        {
            cv::Point newP = current+dir;
            // già visto?
            auto alreadySeen = eval.at<uchar>(newP) == 255;
            
            // segnalo come controllato
            eval.at<uchar>(newP) = 255;
            // Se già è stato valutato passa al prossimo pixel
            if(!inside(mask.size(), newP) or mask.at<uchar>(newP) == 255 or alreadySeen or (white.at<uchar>(newP) == 0 and blue.at<uchar>(newP) == 0))
                continue;
            // altrimenti controllalo (mettendolo in coda)
            q.push(newP);
        }
    }
    return count;
}

/*
    Data una matrice contenente i valori di Hue, restituisce un'immagine in formato BGR.
    Questa immagine può essere utilizzata in fase di debugging per capire la distribuzione delle fasce di tonalità HSV.
    L'i-esima fascia verrà colorata in outRGB con il colore espresso dall'i-esimo colore in colors.
    
    Il numero di bin in cui quantizzare viene preso dal numero di colori presenti nella lista colors.
 
    Quindi se vogliamo quantizzare in 6 fasce possiamo definire una lista colors = {Arancione, Giallo, Verde, Blu, Viola, Rosso}.
 
    @param hueSrc: matrice contenente i valori di Hue.
    @param outRGB: immagine in formato BGR contenente la distribuzione delle tonalità
    @param colors: lista dei colori da individuare.
 */
void isis::getColorDistribution(const cv::Mat& hueSrc, cv::Mat& outRGB, const std::vector<cv::Vec3b>& colors)
{
    // numero bin
    int binSize = 180/colors.size();
    
    // inizializzo l'immagine da dare in output
    outRGB = cv::Mat(hueSrc.size(), CV_8UC3);
    
    // la coloro
    for(int y=0; y<hueSrc.rows; y++)
        for(int x=0; x<hueSrc.cols; x++)
            outRGB.at<cv::Vec3b>(y, x) = colors[hueSrc.at<uchar>(y, x) / binSize];
}

/*
    Data un'immagine ci dice se è necessario utilizzare un classificatore di occhi o meno.
    
    //TODO: Implementare ulteriori controlli
    @param src: immagine da esaminare
    @return true se è necessario utilizzare un classificatore, false altrimenti
 */
bool isis::needClassifier(const cv::Mat& src)
{
    cv::Mat hsvSrc, ycrcbSrc, skinMask = cv::Mat::zeros(src.size(), CV_8UC1);
    std::vector<cv::Mat> hsvChannels;
    
    // Passa da bgr a hsv
    cv::cvtColor(src, hsvSrc, cv::COLOR_BGR2HSV);
    // Passa da bgr a ycrcb
    cv::cvtColor(src, ycrcbSrc, cv::COLOR_BGR2YCrCb);
    
    // Dividi in canali l'immagine hsv
    cv::split(hsvSrc, hsvChannels);
    
    // inizializza un istogramma
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
