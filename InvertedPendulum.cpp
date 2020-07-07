//
//  InvertedPendulum.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#include "InvertedPendulum.hpp"

InvertedPendulum::InvertedPendulum()
{
    numstate = 2;
    numinput = 1;
    m = 1;
    l = 1;
    g = 1;
    ranges.resize(numstate+numinput,2);
    ranges.row(0) << -5,5;
    ranges.row(1) << -20,20;
    ranges.row(2) << -10,10;
    simdt = 0.001;
    stepdt = 0.01;
}

InvertedPendulum::InvertedPendulum(double mass, double len, double grav, double sim, double step)
{
    numstate = 2;
    numinput = 1;
    m = mass;
    l = len;
    g = grav;
    ranges.resize(numstate+numinput,2);
    ranges.row(0) << -5,5;
    ranges.row(1) << -20,20;
    ranges.row(2) << -10,10;
    simdt = sim;
    stepdt = step;
}

Eigen::MatrixXd InvertedPendulum::dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    Eigen::MatrixXd th, thd;
    Eigen::MatrixXd F = control.row(0);
    Eigen::MatrixXd nextstate = Eigen::MatrixXd::Zero(state.rows(),state.cols()) + state;
    Eigen::MatrixXd rates = Eigen::MatrixXd::Zero(state.rows(),state.cols());
    
//    std::cout << "got here\n";
    
    double t = 0;
    while (t < stepdt-simdt/100)
    {
        th = nextstate.row(0);
        thd = nextstate.row(1);
        rates.row(0) = thd;
        rates.row(1) = th.array().sin().matrix()*g/l + F/(m*l*l);
        nextstate += rates*simdt;
        t += simdt;
    }
    
    return nextstate;
}

Eigen::MatrixXd InvertedPendulum::linearize(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(state.rows(),state.rows()+control.rows());

    // A, continuous
    M(0,0) = 0;
    M(0,1) = 1;
    M(1,0) = std::cos(state(0,0))*g/l;
    M(1,1) = 0;
    
    // B, continuous
    M(0,2) = 0;
    M(1,2) = 1/(m*l*l);
    
    // Discretize
    M *= stepdt;
    M.block(0,0,numstate,numstate) += Eigen::MatrixXd::Identity(numstate, numstate);
    
    return M;
}

Eigen::MatrixXd InvertedPendulum::execute(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    return dynamics(state, control);
}
