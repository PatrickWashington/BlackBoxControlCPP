//
//  ILQR.hpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#ifndef ILQR_hpp
#define ILQR_hpp

#include "Eigen/Dense"
#include "System.hpp"
#include "InvertedPendulum.hpp"
#include <chrono>

class ILQR
{
public:
    System * sys;
    bool isnetsys;
    
    int maxiter;
    double controlchange;
    double enderror;
    
    double lr;
    double itertime;
    
    int executesteps;
    int horizonsteps;
    int totalsteps;
    
    Eigen::MatrixXd Qf;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    
    Eigen::MatrixXd K;
    Eigen::MatrixXd Ku;
    Eigen::MatrixXd Kv;
    Eigen::MatrixXd v;
    
    ILQR();
    ILQR(System *system, int mi, double du, double ee, double rate, int execstep, int horizstep, int totstep, double q, double qf, double r, bool net);
    
    Eigen::MatrixXd solve(Eigen::MatrixXd start, Eigen::MatrixXd goal, Eigen::MatrixXd u);
    Eigen::MatrixXd iterate(Eigen::MatrixXd goal, Eigen::MatrixXd x, Eigen::MatrixXd u);
    bool checkdone(int itercount, double du, double error);
    
    Eigen::MatrixXd execute(Eigen::MatrixXd x, Eigen::MatrixXd u, Eigen::MatrixXd state);
};


#endif /* ILQR_hpp */
