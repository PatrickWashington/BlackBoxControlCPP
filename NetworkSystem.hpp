//
//  NetworkSystem.hpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/16/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#ifndef NetworkSystem_hpp
#define NetworkSystem_hpp


#include "System.hpp"
#include "NeuralNetwork.hpp"
#include "Eigen/Dense"
#include<iostream>

class NetworkSystem : public System
{
public:
    NeuralNetwork net;
    System * basesystem;
    
    NetworkSystem();
    NetworkSystem(NeuralNetwork network, System *base);
    
    Eigen::MatrixXd dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd linearize(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd execute(Eigen::MatrixXd state, Eigen::MatrixXd control) override;

    Eigen::MatrixXd testcontrol(Eigen::MatrixXd start, Eigen::MatrixXd control);
};



#endif /* NetworkSystem_hpp */
