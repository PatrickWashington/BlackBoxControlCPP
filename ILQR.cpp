//
//  ILQR.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#include "ILQR.hpp"

ILQR::ILQR() {}

ILQR::ILQR(System *system, int mi, double du, double ee, double rate, int execstep, int horizstep, int totstep, double q, double qf, double r, bool net)
{
    sys = system;
    maxiter = mi;
    controlchange = du;
    enderror = ee;
    lr = rate;
    executesteps = execstep;
    horizonsteps = horizstep;
    totalsteps = totstep;
    itertime = (sys->stepdt) * executesteps;
    
    Qf = qf * Eigen::MatrixXd::Identity(sys->numstate,sys->numstate);
    Q = q * Eigen::MatrixXd::Identity(sys->numstate,sys->numstate);
    // Q(1,1) *= 0.1;
    R = r * Eigen::MatrixXd::Identity(sys->numinput,sys->numinput);
    
    K = Eigen::MatrixXd::Zero(sys->numinput,sys->numstate*(horizonsteps));
    Ku = Eigen::MatrixXd::Zero(sys->numinput,sys->numinput*(horizonsteps));
    Kv = Eigen::MatrixXd::Zero(sys->numinput,sys->numstate*(horizonsteps));
    v = Eigen::MatrixXd::Zero(sys->numstate,horizonsteps+1);
    
    isnetsys = net;
}

Eigen::MatrixXd ILQR::solve(Eigen::MatrixXd start, Eigen::MatrixXd goal, Eigen::MatrixXd u)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(sys->numstate+sys->numinput,horizonsteps+1);
    
    Eigen::MatrixXd x = sys->simulate(start,u,horizonsteps);
    
    int itercount = 0;
    double maxdu, error;
    Eigen::MatrixXd changes, dx, du;
    Eigen::MatrixXd state = start;

    do {
        changes = iterate(goal, x, u);
        dx = changes.block(0,0,sys->numstate,changes.cols());
        du = changes.block(sys->numstate,0,sys->numinput,changes.cols()-1);
        
        auto test_time = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(test_time-start_time);
        if (diff.count()/1000000.0 >= itertime*0.9) {std::cout << itercount << " timeout\n"; break;} //should find a way to fix this such that it does not use the last step. Issue now is how I'm storing K,Ku,Kv,v as member variables

        x += dx;
        u += du;
        
        itercount++;
        maxdu = du.cwiseAbs().maxCoeff();
        error = (x.col(x.cols()-1)-goal).norm();
    } while (!checkdone(itercount, maxdu, error));

    result.block(0,0,sys->numstate,horizonsteps+1) = x;
    result.block(sys->numstate,0,sys->numinput,horizonsteps) = u;
    return result;
}

Eigen::MatrixXd ILQR::iterate(Eigen::MatrixXd goal, Eigen::MatrixXd x, Eigen::MatrixXd u)
{
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(sys->numstate,sys->numstate*(horizonsteps+1));
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(sys->numstate,sys->numstate*(horizonsteps));
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(sys->numstate,sys->numinput*(horizonsteps));
    Eigen::MatrixXd du = Eigen::MatrixXd::Zero(sys->numinput,horizonsteps);
    Eigen::MatrixXd dx = Eigen::MatrixXd::Zero(sys->numstate,horizonsteps+1);
    Eigen::MatrixXd changes = Eigen::MatrixXd::Zero(sys->numstate+sys->numinput,horizonsteps+1);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(sys->numstate,sys->numstate+sys->numinput);
    
    S.block(0,S.cols()-sys->numstate,sys->numstate,sys->numstate) = Qf;
    v.block(0,v.cols()-1,sys->numstate,1) = Qf * (x.col(x.cols()-1) - goal);
    
    for(int kk = 0; kk < horizonsteps; kk++)
    {
        M = sys->linearize(x.col(kk), u.col(kk));
        A.block(0,kk*sys->numstate,sys->numstate,sys->numstate) = M.block(0,0,sys->numstate,sys->numstate);
        B.block(0,kk*sys->numinput,sys->numstate,sys->numinput) = M.block(0,sys->numstate,sys->numstate,sys->numinput);
    }
    
    int kk;
    Eigen::MatrixXd temp, Ak, Bk, Sk1, vk1;
    for(int ii = 0; ii < horizonsteps; ii++)
    {
        kk = horizonsteps - ii - 1;
        Ak = A.block(0,kk*sys->numstate,sys->numstate,sys->numstate);
        Bk = B.block(0,kk*sys->numinput,sys->numstate,sys->numinput);
        Sk1 = S.block(0,(kk+1)*sys->numstate,sys->numstate,sys->numstate);
        vk1 = v.block(0,kk+1,sys->numstate,1);
        temp = ( Bk.transpose() * Sk1 * Bk + R + 0.0*Eigen::MatrixXd::Identity(R.rows(),R.cols())).inverse();
        K.block(0,kk*sys->numstate,sys->numinput,sys->numstate) = temp * Bk.transpose() * Sk1 * Ak;
        Kv.block(0,kk*sys->numstate,sys->numinput,sys->numstate) = temp * Bk.transpose();
        Ku.block(0,kk*sys->numinput,sys->numinput,sys->numinput) = temp * R;
        temp = Ak - Bk * K.block(0,kk*sys->numstate,sys->numinput,sys->numstate);
        S.block(0,kk*sys->numstate,sys->numstate,sys->numstate) = Ak.transpose() * Sk1 * temp + Q;
        v.block(0,kk,sys->numstate,1) = temp.transpose()*vk1 - K.block(0,kk*sys->numstate,sys->numinput,sys->numstate).transpose()*R*u.col(kk) + Q*x.col(kk);
    }
    
    Eigen::MatrixXd control;
    dx *= 0;
    du *= 0;
    for(int kk = 0; kk < horizonsteps; kk++)
    {
        du.col(kk) = (-K.block(0,kk*sys->numstate,sys->numinput,sys->numstate)*dx.col(kk) - Kv.block(0,kk*sys->numstate,sys->numinput,sys->numstate)*v.col(kk+1) - Ku.block(0,kk*sys->numinput,sys->numinput,sys->numinput)*u.col(kk))*lr;
        control = u.col(kk) + du.col(kk);
        control = control.array().max(sys->ranges.block(sys->numstate,0,sys->numinput,1).array()).matrix();
        control = control.array().min(sys->ranges.block(sys->numstate,1,sys->numinput,1).array()).matrix();
        du.col(kk) = control - u.col(kk);
        dx.col(kk+1) = sys->dynamics(x.col(kk)+dx.col(kk), control) - x.col(kk+1);
    }
    
    changes.block(0,0,sys->numstate,horizonsteps+1) = dx;
    changes.block(sys->numstate,0,sys->numinput,horizonsteps) = du;
    
    return changes;
}

bool ILQR::checkdone(int itercount, double du, double error)
{
    if (itercount >= maxiter) {std::cout << itercount << " iter\n"; return true;}
    if (du < controlchange) {std::cout << du << " du\n"; return true;}
    if (error < enderror) {std::cout << error << " error\n"; return true;}
    return false;
}

Eigen::MatrixXd ILQR::execute(Eigen::MatrixXd x, Eigen::MatrixXd u, Eigen::MatrixXd state)
{
    Eigen::MatrixXd traj = Eigen::MatrixXd::Zero(x.rows(),x.cols());
    traj.col(0) = state;
    Eigen::MatrixXd du = Eigen::MatrixXd::Zero(sys->numinput,1);
    Eigen::MatrixXd control = Eigen::MatrixXd::Zero(u.rows(),u.cols());
    long numstep = u.cols();
    Eigen::MatrixXd toreturn = Eigen::MatrixXd::Zero(sys->numstate+sys->numinput, x.cols());
    
    for (int kk = 0; kk < numstep; kk++)
    {
        du = (-K.block(0,kk*sys->numstate,sys->numinput,sys->numstate)*(traj.col(kk)-x.col(kk)) - Kv.block(0,kk*sys->numstate,sys->numinput,sys->numstate)*v.col(kk+1) - Ku.block(0,kk*sys->numinput,sys->numinput,sys->numinput)*u.col(kk)); // took out *lr. Is that right?
        control.col(kk) = u.col(kk) + du;
        control.col(kk) = control.col(kk).array().max(sys->ranges.block(sys->numstate,0,sys->numinput,1).array()).matrix();
        control.col(kk) = control.col(kk).array().min(sys->ranges.block(sys->numstate,1,sys->numinput,1).array()).matrix();
        traj.col(kk+1) = sys->execute(traj.col(kk), control.col(kk));
    }
    
    toreturn.block(0,0,traj.rows(),traj.cols()) = traj;
    toreturn.block(traj.rows(),0,control.rows(),control.cols()) = control;
    return toreturn;
}
