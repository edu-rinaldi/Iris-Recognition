//
//  utility.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef utility_hpp
#define utility_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

/*
    Genera un colore random
 */
inline cv::Scalar getRandomColor(cv::RNG& rng){ return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));}

/*
    Restituisce la distanza tra due punti
    @param a: punto 1
    @param b: punto 2
    
    @return distanza tra punto 1 e punto 2
 */
inline double distance(const cv::Point& a, const cv::Point& b) { return cv::sqrt(cv::pow(b.x - a.x, 2) + cv::pow(b.y - a.y, 2) * 1.0); }

/*
    Questa funzione è utile per evitare errori di index out of range quando scorriamo su un immagine
    (non nel modo classico, ovvero i doppi for)
    
    @param size: dimensioni di un rettangolo
    @param point: punto da esaminare
    @return true se point è all'interno delle dimensioni di size
 */
inline bool inside(const cv::Size& size, const cv::Point& point) { return point.x >= 0 && point.x < size.width && point.y >= 0 && point.y < size.height; }


// Finestra utilizzata per il filtro di posterizzazione
struct SlidingWindow
{
    int sx, ex, sy, ey;
    static SlidingWindow getSlidingWindow(const cv::Mat &img, const int &y,const int &x, const int &k);
};

/*
    Restituisce il numero di pixel spenti in una maschera
    @param src: maschera da esaminare
    @return numero di pixel spenti
 */
int noise(const cv::Mat& src);

/*
    Mappa un valore da un range [is,ie] ad un valore nel range [os, oe]
    @param in: valore da mappare
    @param is: valore minimo che può assumere "in"
    @param ie: valore massimo che può assumere "in"
    @param os: valore minimo che vogliamo far assure nel range di output
    @param oe: valore massimo che vogliamo far assure nel range di output
 
    @return in mappato nel range [os,oe]
 */
inline double map(const double& in, const double& is, const double& ie, const double& os, const double& oe)
{
    return os + ((oe - os) / (ie - is)) * (in - is);
}

/*
    Funzione di utilità che dato un percorso aggiunge una stringa s al file name del percorso fpath.
    Esempio:
        - fpath = /usr/foo/file.txt
        - s = "di_prova"
        - output <-- /usr/foo/file_di_prova.txt
 
    @param fpath: percorso da modificare
    @param s: stringa da aggiungere
 */
std::string addStringToFileName(const std::string& fpath, const std::string& s);

/*
    Join tra due path
 */
std::string appendPath(const std::string& lpath, const std::string& rpath);

/*
    Restituisce il nome del file, se withExtension è true allora c'è anche l'estensione altrimenti no.
 */
std::string getFileName(const std::string& fpath, const bool& withExtension = true);

/*
    Restituisce un'immagine di grandezza size, colorata tutta con il colore scalar.
    L'immagine viene scritta in out.
 */
void getPlainColorImg(const cv::Size& size, const cv::Scalar& scalar,cv::Mat& out);

#endif /* utility_hpp */
