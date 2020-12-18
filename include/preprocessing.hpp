//
//  preprocessing.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 07/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef preprocessing_hpp
#define preprocessing_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "circle.hpp"
#include "utility.hpp"


namespace isis
{
// piccolo enum per scegliere il metodo di thresholding
#define SIMPLE_THRESHOLD 0
#define ADAPTIVE_THRESHOLD 1

// nome file classifier
static const std::string HAARCASCADE_EYE_TREE_EYEGLASSES = "haarcascade_eye_tree_eyeglasses.xml";
static const std::string HAARCASCADE_EYE = "haarcascade_eye.xml";

/*
    Data un'immagine in input restituisce l'immagine posterizzata con una finestra grande (k*2+1) x (k*2+1).
    @param src: immagine in scala di grigi da posterizzare
    @param out: immagine posterizzata
    @param k: parametro per decidere la grandezza della finestra
 */
void posterization(const cv::Mat& src, cv::Mat& out, const int& k);

/*
    Crea la maschera che individua i riflessi tramite il Simple Thresholding.
    @param src: immagine su cui cercare i riflessi
    @param mask: maschera contenente l'output
 */
void filterReflectionST(const cv::Mat& src, cv::Mat& mask);

/*
    Crea la maschera che individua i riflessi tramite l'Adaptive Thresholding.
    @param src: immagine su cui cercare i riflessi.
    @param mask: maschera contenente l'output.
 */
void filterReflectionAT(const cv::Mat& src, cv::Mat& mask);

/*
    Individua e corregge i riflessi tramite un algoritmo di inpainting.
    @param src: immagine su cui correggere i riflessi
    @param out: immagine corretta
    @param mode: metodo di thresholding da utilizzare
 */
void filterReflection(const cv::Mat& src, cv::Mat& out, const int& mode = SIMPLE_THRESHOLD);

/*
    Effettua il cropping dell'immagine sull'area di interesse utilizzando un classificatore HAARCASCADE.
    @param src: immagine da croppare
    @param out: immagine in output
 */
void cropEye(const cv::Mat& src, cv::Mat& out);

/*
    Individua una lista di rettangoli. Ogni rettangolo contiene l'area di interesse (un occhio)
    @param src: immagine in cui cercare
    @param classifierPath: percorso dove è situato il modello del classificatore.
    
    @return una lista di rettangoli corrispondenti alle varie possibili aree di interesse.
 */
std::vector<cv::Rect> getROI(const cv::Mat& src, const std::string& classifierPath);

/*
    Applica un filtro di blurring all'immagine in input
    @param mat: immagine su cui applicare il filtro
    @param out: immagine in cui viene messo l'output
    @param windowSize: grandezza della finestra
 */
void blurFilter(const cv::Mat& mat, cv::Mat& out,const int& windowSize);

/*
    Restituisce una nuova matrice su cui è stato applicato un filtro di smoothing
    applicato alle singole righe (utilizzata per attenuare le ciglia).
    
    @param src: matrice su cui applicare il filtro.
    @param out: matrice filtrata.
*/
inline void eyelashSmoothing(const cv::Mat& src, cv::Mat& out) { blurFilter(src, out, 10); }

/*
    Restituisce una nuova matrice costituente l'immagine del gradiente, in cui il valore
    di intensità nei vari punti rappresenta il gradiente con direzione verticale in un
    certo punto dell'immagine originale.
    
    @param src: matrice su cui calcolare l'immagine del gradiente.
    @param out: matrice costituita dall'immagine del gradiente della matrice in input.
    @param up: indica se ci si riferisce o meno alla palpebra superiore.
*/
void gradientImage(const cv::Mat& src, cv::Mat& out, const bool& up);

/*
    Restituisce un'immagine scalata con grandezza massima pari a finalSize.
    L'immagine viene scalata mantenendo l'aspect ratio.
    Quindi se abbiamo un immagine grande 1280x720 (16:9), e finalSize=500,
    l'immagine in output sarà grande: 500x280.
 
    @param src: immagine da scalare
    @param out: immagine in output scalata
    @param finalSize: grandezza massima del lato più grande dell'immagine.
 */
bool scaleImage(const cv::Mat& src, cv::Mat& out, const int& finalSize);

/*
    Dato un certo pixel ci dice se possiamo considerarlo come pixel appartenente alla pelle o meno.
    
    @param bgr: immagine in formato bgr.
    @param hsv: immagine in formato hsv.
    @param ycrcb: immagine in formato ycrcb.
    @param y: coordinata y del pixel.
    @param x: coordinata x del pixel.
 
    @return true se è un pixel della pelle, false altrimenti
 */
bool isSkinPixel(const cv::Mat& bgr, const cv::Mat& hsv, const cv::Mat& ycrcb, const int& y, const int& x);

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
int getComponent(const cv::Mat& blue, const cv::Mat& white, cv::Mat& mask, const int& y, const int& x);

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
void getColorDistribution(const cv::Mat& hueSrc, cv::Mat& outRGB, const std::vector<cv::Vec3b>& colors);

/*
    Data un'immagine ci dice se è necessario utilizzare un classificatore di occhi o meno.
    
    //TODO: Implementare ulteriori controlli
    @param src: immagine da esaminare
    @return true se è necessario utilizzare un classificatore, false altrimenti
 */
bool needClassifier(const cv::Mat& src);
}
#endif /* preprocessing_hpp */
