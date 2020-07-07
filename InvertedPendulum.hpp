//
//  InvertedPendulum.hpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#ifndef InvertedPendulum_hpp
#define InvertedPendulum_hpp

#include "System.hpp"
#include "Eigen/Dense"
#include<iostream>

class InvertedPendulum : public System
{
public:
    double m;
    double l;
    double g;
    
    InvertedPendulum();
    InvertedPendulum(double mass, double len, double grav, double sim, double step);
    
    Eigen::MatrixXd dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd linearize(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd execute(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
};

#endif /* InvertedPendulum_hpp */
