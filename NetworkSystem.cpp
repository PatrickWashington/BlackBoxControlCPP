//
//  NetworkSystem.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/16/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#include "NetworkSystem.hpp"

NetworkSystem::NetworkSystem() {}

NetworkSystem::NetworkSystem(NeuralNetwork network, System *base)
{
    net = network;
    basesystem = base;
    simdt = base->stepdt;  //intentionally have both as stepdt
    stepdt = base->stepdt; //network should have stepdt as its trained dt
    numstate = base->numstate;
    numinput = base->numinput;
    ranges = base->ranges;
}

Eigen::MatrixXd NetworkSystem::dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    return net.evaluate(state, control);
}

Eigen::MatrixXd NetworkSystem::linearize(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(net.hiddensize,1);
    Eigen::MatrixXd mult = Eigen::MatrixXd::Zero(net.hiddensize,net.hiddensize);
    Eigen::MatrixXd Mat = Eigen::MatrixXd::Identity(net.inputsize, net.inputsize);
    Eigen::MatrixXd bias = Eigen::MatrixXd::Zero(net.hiddensize, 1);
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Zero(numstate+numinput,1);
    inputs.block(0,0,numstate,1) = state;
    inputs.block(numstate,0,numinput,1) = control;
    
    Mat = net.inweights * Mat;
    bias = net.inbiases + bias;
    
    for(int ii = 0; ii < net.numhidden; ii++)
    {
        temp = net.activationdiff(Mat*inputs + bias);
        // temp = (net.activationfun(Mat*inputs + bias).array() / (Mat*inputs + bias).array()).matrix();

        for(int jj = 0; jj < temp.rows(); jj++) { mult(jj,jj) = temp(jj,0); }
        Mat = mult * Mat;
        bias = mult * bias;
        
        if((mult.array() != mult.array()).any())
        {
            std::cout << temp.transpose() << "\n\n";
            std::cout << mult << "\n\n";
            std::cout << Mat*inputs + bias << "\n\n";
            std::cout << Mat << "\n\n";
            std::cout << bias << "\n\n";
            std::cout << inputs << "\n\n";
            std::cout << ii << "\n\n";
        }
        
        Mat = net.hiddenweights.block(0,ii*net.hiddensize,net.hiddensize,net.hiddensize) * Mat;
        bias = net.hiddenweights.block(0,ii*net.hiddensize,net.hiddensize,net.hiddensize) * bias + net.hiddenbiases.block(0,ii,net.hiddensize,1);
    }
    temp = net.activationdiff(Mat*inputs + bias);
    // temp = (net.activationfun(Mat*inputs + bias).array() / (Mat*inputs + bias).array()).matrix();

    for(int jj = 0; jj < temp.rows(); jj++) { mult(jj,jj) = temp(jj,0); }
    Mat = mult * Mat;
    bias = mult * bias; // unnecessary line
    
    Mat = net.outweights * Mat;
    bias = net.outweights * bias + net.outbiases; // unnecessary line
    
    return Mat;
}

Eigen::MatrixXd NetworkSystem::execute(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    // return dynamics(state,control);
    return basesystem->dynamics(state, control);
}

Eigen::MatrixXd NetworkSystem::testcontrol(Eigen::MatrixXd start, Eigen::MatrixXd control)
{
    long numstep = control.cols();
    Eigen::MatrixXd state = Eigen::MatrixXd::Zero(start.rows(),numstep+1);
    state.col(0) = start;
    for(int ii = 0; ii < numstep; ii++)
    {
        state.col(ii+1) = dynamics(state.col(ii),control.col(ii));
    }
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(start.rows()+control.rows(),numstep+1);
    result.block(0,0,state.rows(),state.cols()) = state;
    result.block(state.rows(),0,control.rows(),control.cols()) = control;
    return result;
}
