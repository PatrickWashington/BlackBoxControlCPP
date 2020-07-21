//
//  System.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#include "System.hpp"
#include <iostream>

System::System() {};

//Eigen::MatrixXd System::dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control)
//{
//    std::cout << "Override this!\n";
//    return state*0;
//}

//Eigen::MatrixXd System::dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control, double trajdt, double simdt)
//{
//    double t = 0;
//    Eigen::MatrixXd rates;
//    while (t < trajdt)
//    {
//        rates = dynamics(state, control);
//        state += rates * std::min(simdt,trajdt-t);
//        t += simdt;
//    }
//    return state;
//}

//Eigen::MatrixXd System::linearize(Eigen::MatrixXd state, Eigen::MatrixXd control)
//{
//    Eigen::MatrixXd M;
//    M.resize(numstate, numstate+numinput+1);
//    return M;
//}

//Eigen::MatrixXd System::linearize(Eigen::MatrixXd state, Eigen::MatrixXd control, double dt)
//{
//    Eigen::MatrixXd M = linearize(state, control);
////    std::cout << M << "\n";
//    M *= dt;
////    std::cout << M << "\n";
//    M.block(0,0,numstate,numstate) += Eigen::MatrixXd::Identity(numstate, numstate);
////    std::cout << M << "\n";
//    return M;
//}

Eigen::MatrixXd System::simulate(Eigen::MatrixXd start, Eigen::MatrixXd u, int numstep)
{
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(numstate, numstep+1);
    x.col(0) = start;
    
    for(int ii = 0; ii < numstep; ii++)
    {
        x.col(ii+1) = dynamics(x.col(ii), u.col(ii));
    }
    
    return x;
}

Eigen::MatrixXd System::findequilibria(Eigen::MatrixXd startpoints, int maxiter, double thresh, double step)
{
    startpoints = startpoints.block(0,0,numstate,startpoints.cols());
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(numstate,startpoints.cols());
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(numinput,1);
    Eigen::MatrixXd x,f,A,dx;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(numstate,numstate);

    for(int ii = 0; ii < startpoints.cols(); ii++)
    {
        x = startpoints.block(0,ii,numstate,1);
        for(int jj = 0; jj < maxiter; jj++)
        {
            f = dynamics(x,u);
            A = linearize(x,u).block(0,0,numstate,numstate);
            dx = (A-I).inverse() * (f-x);
            dx *= step;
            for(int kk = 0; kk < numstate/2; kk++)
            {
                dx.block(kk,0,1,1) = dx.block(kk,0,1,1).array().min((ranges(kk,1)-ranges(kk,0))/5.0).max((ranges(kk,0)-ranges(kk,1))/5.0).matrix();
            }
            x -= dx;
            if(dx.norm() < thresh){ break; }
            // std::cout << x.transpose() << "\n";
        }
        result.col(ii) = x;
    }

    return result;
}
