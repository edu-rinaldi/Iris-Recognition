//
//  isis_segmentation.hpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 08/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#ifndef isis_segmentation_hpp
#define isis_segmentation_hpp

#include "settings.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "preprocessing.hpp"

namespace isis
{

typedef struct
{
    cv::Mat sourceCropped;  // Immagine croppata dal classificatore
    cv::Mat normalized;     // Immagine normalizzata
    cv::Mat mask;           // Maschera dell'immagine normalizzata
    Circle pupil;           // Cerchio della pupilla
    Circle limbus;          // Cerchio del limbo
} segmentation_t;

/*
    Data un'immagine effettua la segmentazione dell'iride
    @param source: immagine da segmentare
    @param segmentation: struct vuota, conterrà la segmentazione ottenuta come output
 */
void segmentation(const cv::Mat& source, segmentation_t& segmentation);

/*
    Effettua il preprocessing dell'immagine.
    In particolare va ad effettuare:
        - il crop tramite l'uso di un classificatore (se necessario)
        - correzione dei riflessi
        - resize dell'immagine (attualmente non utilizzato, infatti la riga è commentata)
    @param source: immagine da preprocessare
    @param out: immagine che conterrà l'immagine preprocessata
    @param reflectionFilterMode: metodo di rilevamento dei riflessi (ADAPTIVE_THRESHOLD, SIMPLE_THRESHOLD)
    @param finalSize: grandezza dell'immagine finale
 */
void preprocess(cv::Mat& source, cv::Mat& out, const int& reflectionFilterMode = ADAPTIVE_THRESHOLD, const int& finalSize = 300);

/*
    Cerca con un approccio bruteforce il miglior limbo per l'immagine passata in input.
    @param grey: immagine in bianco e nero in cui cercare
    @return cerchio che identifica il limbo
 */
Circle findLimbusBF(const cv::Mat& grey);

/*
    Cerca con un approccio bruteforce il miglior limbo per l'immagine passata in input.
    A differenza di findLimbusBF, prende ogni cerchio individuato e effettua una media delle posizioni e della grandezza.
    I valori inidividuati costituiranno il miglior limbo.
    @param grey: immagine in bianco e nero in cui cercare
    @return cerchio che identifica il limbo
 */
Circle findLimbusBFAVG(const cv::Mat& grey);

/*
    Cerca con un approccio bruteforce la miglior pupilla per l'immagine passata in input.
    La pupilla è individuata all'interno del cerchio che viene passato come parametro.
    @param grey: immagine in bianco e nero in cui cercare
    @param bestLimbus: miglior limbo in cui cercare la pupilla
    @return cerchio che identifica la pupilla
 */
Circle findPupilBF(const cv::Mat& grey, const Circle& bestLimbus);

/*
    Cerca con un approccio bruteforce la miglior pupilla per l'immagine passata in input.
    La pupilla è individuata all'interno del cerchio che viene passato come parametro.
    A differenza di findPupilBF, prende ogni cerchio individuato ed effettua una media delle posizioni e della grandezza.
    I valori individuati costituiranno la migliore pupilla.
    @param grey: immagine in bianco e nero in cui cercare
    @param bestLimbus: miglior limbo in cui cercare la pupilla
    @return cerchio che identifica la pupilla
 */
Circle findPupilBFAVG(const cv::Mat& grey, const Circle& bestLimbus);

/*
    Data una lista di cerchi restituisce il miglior cerchio candidato ad essere limbo in base ai valori di omogeneità e separabilità.
    @param mat: immagine su cui calcolare i valori di omogeneità e separabilità per i vari cerchi
    @param circles: lista di cerchi tra cui cercare
    @return una coppia di valori <C,v>. C è il miglior cerchio trovato e v è il suo score
 */
std::pair<Circle, double> findLimbus(const cv::Mat& mat, const std::vector<Circle>& circles);

/*
    Data una lista di cerchi restituisce il miglior cerchio candidato ad essere la pupilla in base ai valori di omogeneità e separabilità.
    @param mat: immagine su cui calcolare i valori di omogeneità e separabilità per i vari cerchi
    @param circles: lista di cerchi tra cui cercare
    @param limbus: limbo in cui cercare
    @return una coppia di valori <C,v>. C è il miglior cerchio trovato e v è il suo score
*/
std::pair<Circle, double> findPupil(const cv::Mat& mat, std::vector<Circle>& circles, const Circle& limbus);

/*
    Data un'immagine trova tutti i possibili cerchi con raggio compreso nel range di valori [minRadius, maxRadius].
    I cerchi vengono individuati con l'algoritmo di Taubin.
    @param mat: immagine in cui cercare i cerchi
    @param outputCircles: lista dei cerchi individuati (output)
    @param minRadius: lunghezza minima del raggio
    @param maxRadius: lunghezza massima del raggio
 */
void findCirclesTaubin(const cv::Mat& mat, std::vector<Circle>& outputCircles, const double& minRadius, const double& maxRadius);

/*
     Data un'immagine trova tutti i possibili cerchi con raggio compreso nel range di valori [minRadius, maxRadius].
     I cerchi vengono individuati con l'algoritmo hough.
     @param mat: immagine in cui cercare i cerchi
     @param outputCircles: lista dei cerchi individuati (output)
     @param minRadius: lunghezza minima del raggio
     @param maxRadius: lunghezza massima del raggio
 */
void findCirclesHough(const cv::Mat& mat, std::vector<Circle>& outputCircles, const double& minRadius, const double& maxRadius);

/*
    Data un'immagine applica il filtro di edge detection Canny parametrizzato da cannyThreshold.
    @param src: immagine su cui applicare canny
    @param out: immagine in cui salvare l'immagine risultante
    @param cannyThreshold: threshold per l'individuamento dei contorni
 */
void canny(const cv::Mat& src, cv::Mat& out, double cannyThreshold);

/*
    Restituisce una serie di possibili threshold da utilizzare durante un algoritmo di individuamento cerchi.
    @param out: lista che conterrà l'output
 */
void cannyThreshold(std::vector<double>& out);

/*
    Restituisce una lista di punti relativi al confine della palpebra superiore,
    limitatamente alla parte che sovrasta l'iride nella sua ampiezza orizzontale.
    @param mat    matrice su cui identificare i punti della palpebra.
    @param limbus cerchio rappresentante il limbo dell'occhio.
    @param pupil  cerchio rappresentante la pupilla dell'occhio.
    @param up  indica se ci si riferisce o meno alla palpebra superiore.
    @return lista di punti relativi al confine della palpebra superiore.
*/
std::vector<cv::Point> findEyelidPoints(const cv::Mat& mat, const Circle& limbus, const Circle& pupil, const bool& up);

/*
    Restituisce una nuova matrice su coordinate polari che rappresenta l'iride normalizzata, calcolata
    a partire dalla matrice che la contiene e dai cerchi rappresentanti la pupilla e il limbo ad essi relativi.
    La normalizzazione è effettuata seguendo l'implementazione presente nell'algoritmo di Krupicka.
    
    Vengono inoltre riempiti due dizionari per il mapping in entrambe le "direzioni"
 
    @param src: immagine contenente l'iride da normalizzare.
    @param out: immagine in output (ovvero l'immagine contenente l'iride normalizzata)
    @param limbus: cerchio del limbo
    @param pupil: cerchio della pupilla
    @param mapCartPol: dizionario che mappa da coordinate cartesiane a polari (output)
    @param mapPolCart: dizionario che mappa da coordinate polari a cartesiane (output)
 */
void normalizeKrupicka(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, std::unordered_map<std::string, cv::Point>& mapCartPol, std::unordered_map<std::string, cv::Point>& mapPolCart);

/*
     METODO UTILIZZATO PER INDIVIDUARE LA PALPEBRA INFERIORE, PRESO DALL'ALGORITMO DI KRUPICKA
     
     IL METODO PER INDIVIDUARE LA PALPEBRA SUPERIORE (CHE INDIVIDUA I PUNTI SULL'IMMAGINE DEL GRADIENTE OTTENUTA
     MEDIANTE CONVOLUZIONE) DIFFERISCE DA QUESTO, UTILIZZATO INVECE PER INDIVIDUARE LA PALEBRA INFERIORE,
     IN QUANTO IL PRIMO FUNZIONA PER LA PALPEBRA SUPERIORE, MA NON PER QUELLA INFERIORE.
     BISOGNA SISTEMARLO PER FARLO FUNZIONARE ANCHE SULLA PALPEBRA INFERIORE, IN MODO TALE DA UTILIZZARE
     QUELLO E RIMUOVERE QUESTA CLASSE.
     
     Prende in input l'immagine su spettro rosso, normalizzata secondo il Rubber Sheet Model utilizzato
     nell'algoritmo di Krupicka (vedi il metodo "normalizeKrupicka" in "IrisFunc"), e restituisce
     la maschera binaria (in biano e nero) in cui i pixel appartenenti alla palpebra inferiore sono
     evidenziati in bianco su sfondo nero.
     
     @param normalizedRedChannel: Immagine normalizzata con il Rubber Sheet Model, secondo l'implementazione di Krupicka.
     @param lowerEyelidMask: Maschera binaria normalizzata contenente i soli pixel relativi alla palpebra inferiore
 */
void lowerEyelidMask(const cv::Mat& normalizedRedChannel, cv::Mat& lowerEyelidMask);

/*
    Data un'immagine restituisce una maschera contenente i riflessi che vengono individuati attraverso l'adaptive tresholding.
    @param src: immagine su cui individuare i riflessi
    @param out: maschera dei riflessi
 */
inline void reflectionsMask(const cv::Mat& src, cv::Mat& out) { cv::adaptiveThreshold(src, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, -10); }

/*
    Restituisce una nuova matrice che rappresenta la maschera binaria contenente tutti i punti da ignorare all'interno dell'area che approssima l'iride.
     
    @param src: matrice su cui calcolare la maschera.
    @param out: maschera relativa ai punti da ignorare nell'iride.
    @param limbus: cerchio rappresentante il limbo dell'occhio.
    @param pupil: cerchio rappresentante la pupilla
    @param upperEyelidPoints: lista di punti relativi alla curva rappresentante il bordo della palpebra superiore
    @param lowerEyelidMask: maschera binaria "normalizzata", contenente i soli pixel della palpebra inferiore
    @param reflectionsMask: maschera binaria contenente i soli pixel relativi ad aree di riflesso (quelli in bianco)
    @param map: mappa che contiene coppie di coordinate cartesiane e polari corrispondenti. Serve a trasformare nuovamente l'immagine normalizzata in coordinate cartesiane.
*/
void negativeMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const std::vector<cv::Point>& upperEyelidPoints, const cv::Mat& lowerEyelidMask, const cv::Mat& reflectionsMask, const std::unordered_map<std::string, cv::Point>& map);

/*
    Prende in input la mat originaria, le circonferenze dell'iride (limbo e pupilla) e la maschera negativa,
    e restituisce la maschera binaria con i soli pixel dell'iride (in bianco),
    escludendo le occlusioni della maschera negativa.
 
    @param src: Mat originaria, di cui servono le dimensioni
    @param out: maschera binaria dell'iride
    @param limbus: cerchio del limbo
    @param pupil: cerchio della pupilla
    @param negMask: maschera negativa, contenente le occlusioni dei riflessi e delle palpebre
*/
void irisMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const cv::Mat& negMask);

/*
    Prende in input la maschera dell'iride croppata e restituisce la sua maschera binaria espressa in coordinate polari.
    
    @param irisCroppedMask: maschera dell'iride in coordinate cartesiane
    @param mask: maschera da dare in output
    @param normalizedSize: dimensioni immagine normalizzata
    @param mapCartPol: dizionario che mappa da coordinate cartesiane a polari
 */
void normalizedMask(const cv::Mat& irisCroppedMask, cv::Mat& mask, const cv::Size& normalizedSize, const std::unordered_map<std::string, cv::Point>& mapCartPol);
}

#endif /* isis_segmentation_hpp */
