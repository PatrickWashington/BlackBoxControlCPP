//
//  NeuralNetwork.hpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include "Eigen/Dense"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

class NeuralNetwork
{
public:
    int inputsize;
    int outputsize;
    int hiddensize;
    int numhidden;
    int activation;
    Eigen::MatrixXd inweights;
    Eigen::MatrixXd inbiases;
    Eigen::MatrixXd hiddenweights;
    Eigen::MatrixXd hiddenbiases;
    Eigen::MatrixXd outweights;
    Eigen::MatrixXd outbiases;

    Eigen::MatrixXd inweightsinv;
    Eigen::MatrixXd hiddenweightsinv;
    Eigen::MatrixXd outweightsinv;
    
    Eigen::MatrixXd inshift;
    Eigen::MatrixXd inscale;
    Eigen::MatrixXd outshift;
    Eigen::MatrixXd outscale;
    
    NeuralNetwork();
    NeuralNetwork(std::string filename);
    NeuralNetwork(int in, int out, int hidden, int layer, int act);
    
    Eigen::MatrixXd evaluate(Eigen::MatrixXd inputs);
    Eigen::MatrixXd evaluate(Eigen::MatrixXd state, Eigen::MatrixXd control);
    
    void train(Eigen::MatrixXd inputs, Eigen::MatrixXd actual, int EPOCH, int BATCH, double lr, double decay);
//    Eigen::MatrixXd shuffle(Eigen::MatrixXd mat);
    double validate(Eigen::MatrixXd inputs, Eigen::MatrixXd actual);
    
    double mseLoss(Eigen::MatrixXd predicted, Eigen::MatrixXd actual);
    
    Eigen::MatrixXd activationfun(Eigen::MatrixXd x);
    Eigen::MatrixXd activationdiff(Eigen::MatrixXd x);
    Eigen::MatrixXd activationinv(Eigen::MatrixXd y);
    
    Eigen::MatrixXd ReLU(Eigen::MatrixXd x);
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd x);
    Eigen::MatrixXd LeakyReLU(Eigen::MatrixXd x);
    Eigen::MatrixXd Tanh(Eigen::MatrixXd x);
    
    Eigen::MatrixXd diffReLU(Eigen::MatrixXd x);
    Eigen::MatrixXd diffSigmoid(Eigen::MatrixXd x);
    Eigen::MatrixXd diffLeakyReLU(Eigen::MatrixXd x);
    Eigen::MatrixXd diffTanh(Eigen::MatrixXd x);

    Eigen::MatrixXd invSigmoid(Eigen::MatrixXd y);
    Eigen::MatrixXd invTanh(Eigen::MatrixXd y);
    Eigen::MatrixXd invLeakyReLU(Eigen::MatrixXd y);
    
    Eigen::MatrixXd col2diag(Eigen::MatrixXd mat);
    Eigen::MatrixXd diag2col(Eigen::MatrixXd mat);
    bool checkmatrix(Eigen::MatrixXd mat);
    Eigen::MatrixXd capchanges(Eigen::MatrixXd mat, double lim);
};



#endif /* NeuralNetwork_hpp */
