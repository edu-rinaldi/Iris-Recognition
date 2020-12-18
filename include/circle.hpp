//
//  Circle.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef circle_hpp
#define circle_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "utility.hpp"

#define M_PI   3.14159265358979323846264338327950288
#define degToRad(angleInDegrees) ((angleInDegrees) * M_PI / 180.0)
#define radToDeg(angleInRadians) ((angleInRadians) * 180.0 / M_PI)


namespace isis
{

/*
    Questa classe serve a rappresentare le informazioni base di un cerchio.
 */
class Circle
{
public:
    double radius;
    cv::Point center;
    
    Circle();
    /*
        @param radius: raggio del cerchio
        @param center: coordinata che indica il centro del cerchio
     */
    Circle(const double& radius, const cv::Point& center);
    
    /*
        @param r: oggetto di tipo rettangolo (classe di OpenCV)
        @return true se il cerchio è all'interno del rettangolo r, false altrimenti
     */
    bool inside(const cv::Rect& r) const;
    
    /*
        @param mat: immagine
        @return true se il cerchio è all'interno dell'immagine mat, false altrimenti
     */
    bool inside(const cv::Mat& mat) const;
    
    /*
        @param x: coordinata x
        @param y: coordinata y
        @return true se il punto in coordinate (x, y) è all'interno del cerchio, false altrimenti
     */
    bool inside(const int& x, const int& y) const;
    
    /*
        Dato un range di raggi dice se il raggio del cerchio è all'interno di questo range.
        @param sradius: valore di inizio del range (inclusivo)
        @param eradius: valore di fine del range (inclusivo)
        @return true se il raggio del cerchio assume un valore nel range [sradius, eradius], false altrimenti
     */
    bool radiusInside(const double& sradius, const double& eradius) const;
    
    /*
        @return la bounding box del cerchio
     */
    cv::Rect getbbox() const;
    
    /*
        Calcola il valore di omogeneità del cerchio per l'immagine passata in input
        @param src: immagine
        @return valore di omogeneità
     */
    double homogeneity(const cv::Mat& src) const;
    
    /*
        Calcola il valore di separabilità del cerchio per l'immagine passata in input
        @param src: immagine
        @return valore di separabilità
     */
    double separability(const cv::Mat& src) const;
    
    /*
        Data un'immagine, prende tutti i pixel interni al cerchio e restituisce il valore medio
        @param src: immagine
        @return valore medio
     */
    double mean(const cv::Mat& src) const;
    
    /*
        @return diametro del cerchio
     */
    inline double diameter() const {return radius * 2;}
    
    /*
        Data un "contorno" restituisce il cerchio che meglio lo approssima
        @param contour: una lista di punti che compongono un contorno
        @return il cerchio che meglio approssima il contorno
        
     */
    static Circle taubin(const std::vector<cv::Point>& contour);
    
    /*
        Funzione statica per il calcolo del valore di omogeneità
        @param src: immagine
        @param c: cerchio
        @return valore di omogeneità
     */
    static double homogeneity(const cv::Mat& src, const Circle& c);
    /*
        Funzione statica per il calcolo del valore di separabilità
        @param src: immagine
        @param c: cerchio
        @return valore di separabilità
     */
    static double separability(const cv::Mat& src, const Circle& c);
};

/*
    Output stream per consentire l'utilizzo di std::cout
 */
std::ostream& operator<<(std::ostream& os, Circle c);

}

#endif /* circle_hpp */
