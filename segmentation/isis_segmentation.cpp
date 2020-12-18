//
//  isis_segmentation.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 08/09/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include "isis_segmentation.hpp"

/*
    Data un'immagine effettua la segmentazione dell'iride
    @param source: immagine da segmentare
    @param segmentation: struct vuota, conterrà la segmentazione ottenuta come output
 */
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
void isis::preprocess(cv::Mat& source, cv::Mat& out, const int& reflectionFilterMode, const int& finalSize)
{
    // Cropping sull'occhio
    if(needClassifier(source))
        cropEye(source, source);
    
    // scala l'immagine per un miglioramento delle performance
    // TODO: forse va tolto, ma è molto utile su utiris
    //scaleImage(source, source, finalSize);
    
    // tolgo i riflessi
    filterReflection(source, out, reflectionFilterMode);
    cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
}

/*
    Cerca con un approccio bruteforce il miglior limbo per l'immagine passata in input.
    @param grey: immagine in bianco e nero in cui cercare
    @return cerchio che identifica il limbo
 */
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

/*
    Cerca con un approccio bruteforce il miglior limbo per l'immagine passata in input.
    A differenza di findLimbusBF, prende ogni cerchio individuato e effettua una media delle posizioni e della grandezza.
    I valori inidividuati costituiranno il miglior limbo.
    @param grey: immagine in bianco e nero in cui cercare
    @return cerchio che identifica il limbo
 */
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

/*
    Cerca con un approccio bruteforce la miglior pupilla per l'immagine passata in input.
    La pupilla è individuata all'interno del cerchio che viene passato come parametro.
    @param grey: immagine in bianco e nero in cui cercare
    @param bestLimbus: miglior limbo in cui cercare la pupilla
    @return cerchio che identifica la pupilla
 */
isis::Circle isis::findPupilBF(const cv::Mat& grey, const Circle& bestLimbus)
{
    auto bestPupil = std::pair<Circle, double>(Circle(-1, cv::Point(0,0)), -10000);
    auto centerCrop = cv::Point(grey.cols/2, grey.rows/2);
    // Decommentare le due righe sotto per applicare la soluzione parziale del blurring, e in posterization cambiare 'grey' con 'blurred'
    //cv::Mat blurred;
    //cv::medianBlur(grey, blurred, 3);
    for(int k = 1; k < 18; k++)
    {
        cv::Mat posterized;
        posterization(grey, posterized, k);
        
        std::vector<Circle> circles;
        findCirclesTaubin(posterized, circles, 8, 90);
        //TODO: da ottimizzare
        //TODO: rivedere meglio le condizioni, non sembrano essere ottime
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

/*
    Cerca con un approccio bruteforce la miglior pupilla per l'immagine passata in input.
    La pupilla è individuata all'interno del cerchio che viene passato come parametro.
    A differenza di findPupilBF, prende ogni cerchio individuato ed effettua una media delle posizioni e della grandezza.
    I valori individuati costituiranno la migliore pupilla.
    @param grey: immagine in bianco e nero in cui cercare
    @param bestLimbus: miglior limbo in cui cercare la pupilla
    @return cerchio che identifica la pupilla
 */
isis::Circle isis::findPupilBFAVG(const cv::Mat& grey, const Circle& bestLimbus)
{
    auto centerCrop = cv::Point(grey.cols/2, grey.rows/2);
    // Decommentare le due righe sotto per applicare la soluzione parziale del blurring, e in posterization cambiare 'grey' con 'blurred'
    //cv::Mat blurred;
    //cv::medianBlur(grey, blurred, 3);
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

/*
    Data una lista di cerchi restituisce il miglior cerchio candidato ad essere limbo in base ai valori di omogeneità e separabilità.
    @param mat: immagine su cui calcolare i valori di omogeneità e separabilità per i vari cerchi
    @param circles: lista di cerchi tra cui cercare
    @return una coppia di valori <C,v>. C è il miglior cerchio trovato e v è il suo score
 */
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

/*
    Data una lista di cerchi restituisce il miglior cerchio candidato ad essere la pupilla in base ai valori di omogeneità e separabilità.
    @param mat: immagine su cui calcolare i valori di omogeneità e separabilità per i vari cerchi
    @param circles: lista di cerchi tra cui cercare
    @param limbus: limbo in cui cercare
    @return una coppia di valori <C,v>. C è il miglior cerchio trovato e v è il suo score
*/
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

/*
    Data un'immagine trova tutti i possibili cerchi con raggio compreso nel range di valori [minRadius, maxRadius].
    I cerchi vengono individuati con l'algoritmo di Taubin.
    @param mat: immagine in cui cercare i cerchi
    @param outputCircles: lista dei cerchi individuati (output)
    @param minRadius: lunghezza minima del raggio
    @param maxRadius: lunghezza massima del raggio
 */
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

/*
     Data un'immagine trova tutti i possibili cerchi con raggio compreso nel range di valori [minRadius, maxRadius].
     I cerchi vengono individuati con l'algoritmo hough.
     @param mat: immagine in cui cercare i cerchi
     @param outputCircles: lista dei cerchi individuati (output)
     @param minRadius: lunghezza minima del raggio
     @param maxRadius: lunghezza massima del raggio
 */
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

/*
    Data un'immagine applica il filtro di edge detection Canny parametrizzato da cannyThreshold.
    @param src: immagine su cui applicare canny
    @param out: immagine in cui salvare l'immagine risultante
    @param cannyThreshold: threshold per l'individuamento dei contorni
 */
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

/*
    Restituisce una serie di possibili threshold da utilizzare durante un algoritmo di individuamento cerchi.
    @param out: lista che conterrà l'output
 */
void isis::cannyThreshold(std::vector<double>& out)
{
    out = std::vector<double>();
    for(int i=0; i<11; i++)
        out.push_back(0.05 + (0.05 * i));
}

/*
    Restituisce una lista di punti relativi al confine della palpebra superiore,
    limitatamente alla parte che sovrasta l'iride nella sua ampiezza orizzontale.
    @param mat    matrice su cui identificare i punti della palpebra.
    @param limbus cerchio rappresentante il limbo dell'occhio.
    @param pupil  cerchio rappresentante la pupilla dell'occhio.
    @param up  indica se ci si riferisce o meno alla palpebra superiore.
    @return lista di punti relativi al confine della palpebra superiore.
*/
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
void isis::irisMask(const cv::Mat& src, cv::Mat& out, const Circle& limbus, const Circle& pupil, const cv::Mat& negMask)
{
    out = cv::Mat::zeros(src.size(), CV_8UC1);
    
    cv::circle(out, limbus.center, (int) limbus.radius - 1, cv::Scalar(255), -1);
    cv::circle(out, pupil.center, (int) pupil.radius, cv::Scalar(0), -1);
    
    for(int y=0; y<src.rows; y++)
        for(int x=0; x<src.cols; x++)
            if(negMask.at<uchar>(y, x) > 0.) out.at<uchar>(y, x) = 0;
}

/*
    Prende in input la maschera dell'iride croppata e restituisce la sua maschera binaria espressa in coordinate polari.
    
    @param irisCroppedMask: maschera dell'iride in coordinate cartesiane
    @param mask: maschera da dare in output
    @param normalizedSize: dimensioni immagine normalizzata
    @param mapCartPol: dizionario che mappa da coordinate cartesiane a polari
 */
void isis::normalizedMask(const cv::Mat& irisCroppedMask, cv::Mat& mask, const cv::Size& normalizedSize, const std::unordered_map<std::string, cv::Point>& mapCartPol)
{
    mask = cv::Mat::zeros(normalizedSize, CV_8UC1);
    for(int y = 0; y < mask.rows; y++)
        for(int x = 0; x < mask.cols; x++)
            if((int)irisCroppedMask.at<uchar>(mapCartPol.at(std::to_string(x)+","+std::to_string(y))) == 255)
                mask.at<uchar>(y, x) = 255;
}

