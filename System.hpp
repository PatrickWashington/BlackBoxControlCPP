//
//  System.hpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#ifndef System_hpp
#define System_hpp


#include "Eigen/Dense"

class System
{
public:
    int numstate;
    int numinput;
    Eigen::MatrixXd ranges;
    double simdt;
    double stepdt;
    
    System();
    
    virtual Eigen::MatrixXd dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control) = 0;
//    Eigen::MatrixXd dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control, double trajdt, double simdt);
    
    virtual Eigen::MatrixXd linearize(Eigen::MatrixXd state, Eigen::MatrixXd control) = 0;
//    Eigen::MatrixXd linearize(Eigen::MatrixXd state, Eigen::MatrixXd control,double dt);
    
    virtual Eigen::MatrixXd execute(Eigen::MatrixXd state, Eigen::MatrixXd control) = 0;
    
    Eigen::MatrixXd simulate(Eigen::MatrixXd start, Eigen::MatrixXd u, int numstep);
};

#endif /* System_hpp */
