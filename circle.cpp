//
//  Circle.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "circle.hpp"
using namespace isis;

Circle::Circle() : radius(0), center(0) {}

/*
    @param radius: raggio del cerchio
    @param center: coordinata che indica il centro del cerchio
 */
Circle::Circle(const double& radius, const cv::Point& center) : radius(radius), center(center) {}

/*
    @param r: oggetto di tipo rettangolo (classe di OpenCV)
    @return true se il cerchio è all'interno del rettangolo r, false altrimenti
 */
bool Circle::inside(const cv::Rect& r) const
{
    return center.x - radius > r.tl().x && center.y - radius > r.tl().y &&
    center.x + radius <= r.br().x && center.y + radius <= r.br().y;
}

/*
    @param mat: immagine
    @return true se il cerchio è all'interno dell'immagine mat, false altrimenti
 */
bool Circle::inside(const cv::Mat& mat) const {return inside(cv::Rect(cv::Point(0,0), mat.size()));}

/*
    @param x: coordinata x
    @param y: coordinata y
    @return true se il punto in coordinate (x, y) è all'interno del cerchio, false altrimenti
 */
bool Circle::inside(const int& x, const int& y) const { return distance(cv::Point(x, y), center) < radius;}

/*
    Dato un range di raggi dice se il raggio del cerchio è all'interno di questo range.
    @param sradius: valore di inizio del range (inclusivo)
    @param eradius: valore di fine del range (inclusivo)
    @return true se il raggio del cerchio assume un valore nel range [sradius, eradius], false altrimenti
 */
bool Circle::radiusInside(const double& sradius, const double& eradius) const {return radius>= sradius && radius <= eradius;}

/*
    @return la bounding box del cerchio
 */
cv::Rect Circle::getbbox() const
{
    return cv::Rect(cv::Point(this->center.x - this->radius, this->center.y - this->radius), cv::Size(this->radius * 2, this->radius * 2));
}

/*
    Calcola il valore di omogeneità del cerchio per l'immagine passata in input
    @param src: immagine
    @return valore di omogeneità
 */
double Circle::homogeneity(const cv::Mat& src) const { return homogeneity(src, *this); }

/*
    Calcola il valore di separabilità del cerchio per l'immagine passata in input
    @param src: immagine
    @return valore di separabilità
 */
double Circle::separability(const cv::Mat& src) const { return separability(src, *this); }

/*
    Data un'immagine, prende tutti i pixel interni al cerchio e restituisce il valore medio
    @param src: immagine
    @return valore medio
 */
double Circle::mean(const cv::Mat& src) const
{
    double mean = 0;
    int countpx = 0;
    cv::Rect r = getbbox();
    for(int i=r.tl().y; i<r.br().y; i++)
    {
        for(int j=r.tl().x; j<r.br().x; j++)
        {
            if(inside(j, i))
            {
                mean += src.at<uchar>(i, j);
                countpx++;
            }
        }
    }
    mean /= countpx;
    return mean;
}

/*
    Data un "contorno" restituisce il cerchio che meglio lo approssima
    @param contour: una lista di punti che compongono un contorno
    @return il cerchio che meglio approssima il contorno
    
 */
Circle Circle::taubin(const std::vector<cv::Point>& contour)
{
    cv::Point2d sum = {0,0};
    for(const cv::Point& p : contour)
    {
        sum.x += p.x;
        sum.y += p.y;
    }

    cv::Point2d mean = {sum.x / contour.size(), sum.y / contour.size()};
    // Mxx = 0, Myy = 1, Mxy = 2, Mxz = 3, Myz = 4, Mzz = 5
    double m[] = {0., 0., 0., 0., 0., 0.};
    
    double xi, yi, zi;
    for(const cv::Point& p : contour)
    {
        xi = p.x - mean.x;
        yi = p.y - mean.y;
        zi = cv::pow(xi, 2) + cv::pow(yi, 2);

        // Mxy
        m[2] += xi*yi;
        // Mxx
        m[0] += cv::pow(xi, 2);
        // Myy
        m[1] += cv::pow(yi, 2);
        // Mxz
        m[3] += xi*zi;
        // Myz
        m[4] += yi*zi;
        // Mzz
        m[5] += cv::pow(zi, 2);

    }

    for(int i=0; i<6; i++) m[i] /= contour.size();

    // coeff
    double  mz = m[0] + m[1],
    cov_xy = m[0] * m[1] - cv::pow(m[2], 2),
    var_z = m[5] - cv::pow(mz, 2),
    a3 = 4. * mz,
    a2 = -3. * cv::pow(mz,2) - m[5],
    a1 = var_z * mz + 4. * cov_xy * mz - cv::pow(m[3], 2) - cv::pow(m[4], 2),
    a0 = m[3] * (m[3] * m[1] - m[4] * m[2]) + m[4] * (m[4] * m[0] - m[3] * m[2]) - var_z * cov_xy,
    a22 = 2. * a2,
    a33 = a3 * 3.,
    x = 0.,
    y = a0;

    // ricerca della radice del polinomio caratteristico usando il metodo di newton partendo da x=0
    // (è garantito che converge sulla giusta radice)
    for(int i=0; i<99; i++)
    {
        double dy = a1 + x * (a22 + a33 * x);
        double xnew = x - y / dy;
        if ((xnew == x) || (!(xnew < std::numeric_limits<double>::max()))) break;

        double ynew = a0 + xnew * (a1 + xnew * (a2 + xnew * a3));
        if(cv::abs(ynew) >= cv::abs(y)) break;

        x = xnew;
        y = ynew;
    }

    // calcolo dei parametri del cerchio
    double det = cv::pow(x, 2) - x * mz + cov_xy,
    xcenter = (m[3] * (m[1] - x) - m[4] * m[2]) / det / 2.,
    ycenter = (m[4] * (m[0] - x) - m[3] * m[2]) / det / 2.;

    // creo l'output
    double radius = cv::sqrt(cv::pow(xcenter, 2) + cv::pow(ycenter, 2) + mz);
    return Circle(radius, cv::Point(xcenter+mean.x, ycenter + mean.y));

}

/*
    Funzione statica per il calcolo del valore di omogeneità
    @param src: immagine
    @param c: cerchio
    @return valore di omogeneità
 */
double Circle::homogeneity(const cv::Mat& src, const Circle& circle)
{
    // Calcolo la maschera del cerchio
    cv::Mat circle_mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::circle(circle_mask, circle.center, circle.radius, cv::Scalar(255,255,255), -1);
    
    
    // Calcolo l'histogram su quel cerchio
    int channels[] = {0};
    int histSize[] = {256};
    
    float lranges[] = {0, 256};
    const float* ranges[] = {lranges};
    cv::Mat hist;
    cv::calcHist(&src, 1, channels, circle_mask, hist, 1, histSize, ranges, true, false);
    
    // Uso la formula per l'omogeneità
    double total_px = 0, max_val = 0, color = 0;
    for(int i = 0; i< hist.rows; i++)
    {
        total_px += hist.at<float>(i, 0);
        if(hist.at<float>(i, 0) > max_val)
        {
            max_val = hist.at<float>(i, 0);
            color = i;
        }
    }
    
    return max_val / total_px;
}

/*
    Funzione statica per il calcolo del valore di separabilità
    @param src: immagine
    @param c: cerchio
    @return valore di separabilità
 */
double Circle::separability(const cv::Mat &src, const Circle &c)
{
    // Array che mantiene le differenze per ogni angolo
    double indexes[360];
    
    // totale differenze
    double total = 0;
    
    // Per ogni angolo
    for(int i=0; i<360; i++)
    {
        // Ascissa interna, settata a 4/5 del raggio
        int x_int = (int) (c.center.x + c.radius * 0.8 * cos(degToRad(i)));
        // Ascissa esterna, settata a 6/5 del raggio
        int x_ext = (int) (c.center.x + c.radius * 1.2 * cos(degToRad(i)));
        // Ordinata interna, settata a 4/5 del raggio
        int y_int = (int) (c.center.y + c.radius * 0.8 * sin(degToRad(i)));
        // Ordinata esterna, settata a 6/5 del raggio
        int y_ext = (int) (c.center.y + c.radius * 1.2 * sin(degToRad(i)));
        
        // Corregge il tiro qualora le coordinate esterne finiscano fuori dalla matrice
        x_ext = x_ext < 0 ? 0 : x_ext;
        y_ext = y_ext < 0 ? 0 : y_ext;
        x_ext = x_ext >= src.cols ? src.cols - 1 : x_ext;
        y_ext = y_ext >= src.rows ? src.rows - 1 : y_ext;

        // Ottiene la differenza tra le intensità interna ed esterna
        indexes[i] = src.at<uchar>(y_ext, x_ext) - src.at<uchar>(y_int, x_int);
        
        // Aggiorna la somma delle differenze
        total += indexes[i];
        
    }
    // Differenza media rispetto agli angoli
    double mean = total / 360.;
    
    // Calcolo della deviazione standard
    double sum = 0;
    for (int i = 0; i < 360; i++)
        sum += cv::pow((indexes[i] - mean), 2);
    
    double stddev = sqrt(sum / 360.);
    
    // Restituisce il valore medio diviso per la (deviazione standard + 1)
    return mean / (stddev + 1);
}

/*
    Output stream per consentire l'utilizzo di std::cout
 */
std::ostream& operator<<(std::ostream& os, Circle c)
{
    os <<"Center: "<<c.center<<" Radius: "<<c.radius;
    return os;
}
