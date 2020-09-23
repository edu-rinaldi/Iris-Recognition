//
//  main.cpp
//  Tirocinio
//
//  Created by Eduardo Rinaldi on 02/08/2020.
//  Copyright © 2020 Eduardo Rinaldi. All rights reserved.
//

#include <iostream>
#include <unordered_map>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>

#include "ezOptionParser.hpp"
#include "isis_segmentation.hpp"
#include "iris_encoding.hpp"

#define SEGMENTATION_OUTPUT_FOLDER "./segmentation/"

namespace fs = std::__fs::filesystem;

int main(int argc, const char * argv[])
{
    ez::ezOptionParser opt;
    opt.overview = "Progetto di tirocinio triennale per la segmentazione e la codifica dell'iride.";
    opt.syntax = "iris (-r|-R) inputImage";
    opt.example = "iris -r path/to/image.png\n\nisis -R path/to/image.png\n\n";
    opt.footer = "Iris recognition using is_is v.2 2020\n";
    
    opt.add("", 0, 1, 0, "Permette la registrazione dell'iride all'interno del programma", "-r", "--register");
    opt.add("", 0, 1, 0, "Restituisce la lista delle iridi più simili a quella che viene passata come parametro", "-R", "--recognize");
    opt.add("", 0, 1, 0, "Debugging della segmentazione, fornisce una ui grafica", "-ds", "--debug-seg");
    
    opt.parse(argc, argv);
    
    if(opt.isSet("-r"))
    {
        
        // carica fname
        std::string fpath;
        opt.get("-r")->getString(fpath);
        
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        if(source.empty())
        {
            std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl;
            return -1;
        }
        
        // segmenta
        isis::segmentation_t segmentation;
        isis::segmentation(source, segmentation);
        
        //  salva
        
        // Nome di base del file in input
        auto filename = getFileName(fpath);
        
        // Costruisco i nuovi filename
        auto normalizedFilename = addStringToFileName(filename, "_normalized");
        auto maskFilename = addStringToFileName(filename, "_mask");
        
        //TODO: Path immagine normalizzata e della relativa maschera normalizzata, inizializzati con la directory di output
        std::string fpathOut = appendPath(SEGMENTATION_OUTPUT_FOLDER, getFileName(fpath, false));
        
        // creo la directory nuova
        fs::create_directory(fpathOut);
        
        // Scrivo le due immagini
        auto saved = cv::imwrite(appendPath(fpathOut, normalizedFilename), segmentation.normalized);
        saved &= cv::imwrite(appendPath(fpathOut, maskFilename), segmentation.mask);
        
        if(!saved)
        {
            std::cerr<<"ERRORE: path di output non valido"<<std::endl;
            exit(-1);
        }
    }
    else if(opt.isSet("-R"))
    {
        // Codice recognize
        std::cout<<"Recognize"<<std::endl;
        
        // carica fname
        std::string fpath;
        opt.get("-R")->getString(fpath);
        
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        if(source.empty())
        {
            std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl;
            return -1;
        }
        
        // segmenta
        isis::segmentation_t sourceSeg;
        isis::segmentation(source, sourceSeg);
        
        // lbp encode di source
        isis::lbp_t lbpSource;
        isis::encodeLBP(lbpSource, sourceSeg);
        
        // spatiogram encode di source
        isis::spatiogram_t spatiogramSource;
        isis::encodeSpatiogram(spatiogramSource, sourceSeg);
        
        auto candidates = std::vector<std::pair<std::string, double>>();
        // Scorro le varie cartelle in segmentation
        for(auto& p : fs::directory_iterator(SEGMENTATION_OUTPUT_FOLDER))
        {
            // Per evitare eventuali file tipo .DS_STORE
            if(!p.is_directory()) continue;
            
            isis::segmentation_t otherSeg;
            // Prendo i file all'interno
            for(auto& tmpPath : fs::directory_iterator(p))
            {
                if(!tmpPath.is_regular_file()) continue;
                if(tmpPath.path().string().find("_mask") != -1) otherSeg.mask = cv::imread(tmpPath.path(), cv::IMREAD_GRAYSCALE);
                if(tmpPath.path().string().find("_normalized") != -1) otherSeg.normalized = cv::imread(tmpPath.path(), cv::IMREAD_COLOR);
            }
            
            // lbp encode di otherSeg
            isis::lbp_t lbpOther;
            isis::encodeLBP(lbpOther, otherSeg);
            
            // spatiogram di otherSeg
            isis::spatiogram_t spatiogramOther;
            isis::encodeSpatiogram(spatiogramOther, otherSeg);
            
            double score = (1 - isis::matchLBP(lbpSource, lbpOther) + isis::matchSpatiogram(spatiogramSource, spatiogramOther)) / 2;
            candidates.push_back({p.path().filename(), score});
        }
        
        
        std::sort(candidates.begin(), candidates.end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2){ return p1.second > p2.second; });
        int count = 0;
        for(auto& pair : candidates)
        {
            if(++count > 5) break;
            std::cout<<pair.first<<": "<<pair.second<<std::endl;
        }
        
    }
    else if(opt.isSet("-ds"))
    {
        std::cout<<"DEBUGGING SEGMENTAZIONE"<<std::endl<<std::endl;
        
        // carica fname
        std::string fpath;
        opt.get("-ds")->getString(fpath);
        
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        if(source.empty())
        {
            std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl;
            return -1;
        }
        std::cout<<"L'immagine presa in considerazione è "<<fpath<<std::endl<<std::endl;
        // segmenta
        isis::segmentation_t sourceSeg;
        isis::segmentation(source, sourceSeg);
        
        cv::circle(sourceSeg.sourceCropped, sourceSeg.limbus.center, sourceSeg.limbus.radius, cv::Scalar(255,0,0));
        cv::circle(sourceSeg.sourceCropped, sourceSeg.pupil.center, sourceSeg.pupil.radius, cv::Scalar(0,255,0));
        cv::putText(sourceSeg.sourceCropped, "Raggio limbo: "+std::to_string(sourceSeg.limbus.radius), sourceSeg.limbus.center, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,0,0));
        cv::putText(sourceSeg.sourceCropped, "Raggio pupilla: "+std::to_string(sourceSeg.pupil.radius), sourceSeg.pupil.center + cv::Point(20,20), cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0,255,0));
        
        std::cout<<"Info limbo"<<std::endl<<"Centro limbo: "<<sourceSeg.limbus.center<<" Raggio limbo: "<<sourceSeg.limbus.radius<<std::endl<<std::endl;
        std::cout<<"Info pupilla"<<std::endl<<"Centro pupilla: "<<sourceSeg.pupil.center<<" Raggio pupilla: "<<sourceSeg.pupil.radius<<std::endl;
        cv::imshow("Segmentation: "+fs::path(fpath).filename().string(), sourceSeg.sourceCropped);
        
        cv::waitKey(0);
    }
    else
    {
        // Se nessuna opzione viene scelta --> print usage
        std::string usage;
        opt.getUsage(usage);
        std::cerr<<"E' stato passato un argomento non valido: "<<std::endl<<usage;
        std::exit(-1);
    }
    
    return 0;
}
