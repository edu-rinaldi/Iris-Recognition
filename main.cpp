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
#include <filesystem>
#include <ctime>

#include <ezOptionParser.hpp>
#include <isis_segmentation.hpp>
#include <iris_encoding.hpp>

#define SEGMENTATION_OUTPUT_FOLDER "./segmentation/"

//namespace fs = std::__fs::filesystem;
namespace fs = std::filesystem;

int main(int argc, const char * argv[])
{
    ez::ezOptionParser opt;
    opt.overview = "Progetto di tirocinio triennale per la segmentazione e la codifica dell'iride.";
    opt.syntax = "iris (-e|-r|-ds) inputImage";
    opt.example = "iris -e path/to/image.png\n\nisis -r path/to/image.png\n\n";
    opt.footer = "Iris recognition using is_is v.2 2020\n";
    
    opt.add("", 0, 1, 0, "Permette la registrazione dell'iride all'interno del programma", "-e", "--enrolling");
    opt.add("", 0, 1, 0, "Restituisce la lista delle iridi più simili a quella che viene passata come parametro", "-r", "--recognize");
    opt.add("", 0, 1, 0, "Debugging della segmentazione, fornisce una ui grafica", "-ds", "--debug-seg");
    opt.add("", 0, 1, 0, "Funzionalità in testing", "--test", "-t");
    opt.add("", 0, 0, 0, "Mostra i possibili utilizzi", "-h", "--help");
    
    opt.parse(argc, argv);
    if(opt.isSet("-h"))
    {
        std::string usage;
        opt.getUsage(usage);
        std::cout<<usage<<std::endl;
        exit(1);
    }
    else if(opt.isSet("-e"))
    {
        
        // carica fname
        std::string fpath;
        opt.get("-e")->getString(fpath);
        
        // Percorso non inserito
        if(fpath.empty())
        {
            std::cerr<<"Non è stato inserito alcun percorso"<<std::endl;
            return -1;
        }
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        
        // Percorso non valido
        if(source.empty())
        {
            std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl;
            return -1;
        }
        
        // segmenta
        isis::segmentation_t segmentation;
        isis::segmentation(source, segmentation);
        
        // salva
        if(!fs::exists(fs::path(SEGMENTATION_OUTPUT_FOLDER))) fs::create_directory(SEGMENTATION_OUTPUT_FOLDER);
        // Nome di base del file in input
        auto filename = getFileName(fpath);
        
        // Costruisco i nuovi filename
        auto normalizedFilename = addStringToFileName(filename, "_normalized");
        auto maskFilename = addStringToFileName(filename, "_mask");
        
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
    else if(opt.isSet("-r"))
    {
        // Codice recognize
        std::cout<<"Recognize"<<std::endl;
        
        // carica fname
        std::string fpath;
        opt.get("-r")->getString(fpath);
        
        // Percorso non inserito
        if(fpath.empty())
        {
            std::cerr<<"Non è stato inserito alcun percorso"<<std::endl;
            return -1;
        }
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        
        // Percorso non valido
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
            
            
            double score = (isis::matchLBP(lbpSource, lbpOther) + (1 - isis::matchSpatiogram(spatiogramSource, spatiogramOther))) / 2.;
            candidates.push_back({p.path().filename(), score});
        }
        
        // Ordina il set individuato
        std::sort(candidates.begin(), candidates.end(), [](const std::pair<std::string, double>& p1, const std::pair<std::string, double>& p2){ return p1.second < p2.second; });
        
        // Print dei primi 5
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
        
        // Percorso non inserito
        if(fpath.empty())
        {
            std::cerr<<"Non è stato inserito alcun percorso"<<std::endl;
            return -1;
        }
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        
        // Percorso non valido
        if(source.empty())
        {
            std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl;
            return -1;
        }
        std::cout<<"L'immagine presa in considerazione è "<<fpath<<std::endl<<std::endl;
        // segmenta
        isis::segmentation_t sourceSeg;
        isis::segmentation(source, sourceSeg);

        
        // Disegna limbo e pupilla
        cv::circle(sourceSeg.sourceCropped, sourceSeg.limbus.center, sourceSeg.limbus.radius, cv::Scalar(255,0,0));
        cv::circle(sourceSeg.sourceCropped, sourceSeg.pupil.center, sourceSeg.pupil.radius, cv::Scalar(0,255,0));
        
        
        std::cout<<"Info limbo"<<std::endl<<"Centro limbo: "<<sourceSeg.limbus.center<<" Raggio limbo: "<<sourceSeg.limbus.radius<<std::endl<<std::endl;
        std::cout<<"Info pupilla"<<std::endl<<"Centro pupilla: "<<sourceSeg.pupil.center<<" Raggio pupilla: "<<sourceSeg.pupil.radius<<std::endl;
        cv::imshow("Segmentation: "+fs::path(fpath).filename().string(), sourceSeg.sourceCropped);
        cv::imshow("normalized", sourceSeg.normalized);
        cv::imshow("mask", sourceSeg.mask);
        cv::waitKey();
    }
    else if(opt.isSet("-t"))
    {
        
        // carica fname
        std::string fpath;
        opt.get("-t")->getString(fpath);
        
        // Percorso non inserito
        if(fpath.empty()) { std::cerr<<"Non è stato inserito alcun percorso"<<std::endl; return -1; }
        
        // Leggi l'immagine
        auto source = cv::imread(fpath, cv::IMREAD_COLOR);
        
        // Percorso non valido
        if(source.empty()) { std::cerr<<"L'immagine selezionata: "<<fpath<<" non è stata trovata o non è valida"<<std::endl; return -1; }
        std::cout<<"L'immagine presa in considerazione è "<<fpath<<std::endl<<std::endl;
        
        cv::Mat out;
        isis::filterReflection(source, out, SIMPLE_THRESHOLD);
        
        
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
