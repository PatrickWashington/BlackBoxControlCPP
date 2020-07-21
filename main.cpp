//
//  main.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

// #include <Python/Python.h>
// #include <boost/boost/python.hpp>
#include <iostream>
#include <fstream>
#include <random>
#include <filesystem>
#include <chrono>
#include "Eigen/Dense"
#include "System.hpp"
#include "InvertedPendulum.hpp"
#include "SlidingBlock.hpp"
#include "NeuralNetwork.hpp"
#include "NetworkSystem.hpp"
#include "ILQR.hpp"
//#include "../../madplotlib/Madplotlib.h"
//#include "../../matplotlib-cpp-master/matplotlibcpp.h"

void matrixToCSV(Eigen::MatrixXd mat, std::string filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << std::to_string(mat.rows()) << "," << std::to_string(mat.cols()) << "\n";
        for (int ii = 0; ii < mat.rows(); ii++)
        {
            for (int jj = 0; jj < mat.cols()-1; jj++)
            {
                file << std::to_string(mat(ii,jj)) << ",";
            }
            file << std::to_string(mat(ii,mat.cols()-1)) << std::endl;
        }
        file.close();
    }
    else
    {
        std::cout << "did not open file\n";
    }
    
}

Eigen::MatrixXd csvToMatrix(std::string filename)
{
    std::ifstream file;
    std::string line, substr;
    int numrow,numcol;
    Eigen::MatrixXd mat;
    
    file.open(filename,std::ios::in);
    if(file.is_open())
    {
        std::getline(file,line);
        std::stringstream ss(line);
        while(ss.good())
        {
            getline(ss, substr, ',');
            numrow = stoi(substr);
            getline(ss, substr, ',');
            numcol = stoi(substr);
        }
        
        mat = Eigen::MatrixXd::Zero(numrow,numcol);

        for(int row = 0; row < numrow; row++)
        {
            std::getline(file,line);
            std::stringstream ss(line);
            for(int col = 0; col < numcol; col++)
            {
                getline(ss, substr, ',');
                mat(row,col) = stof(substr);
            }
        }
    }

    return mat;
}

Eigen::MatrixXd generateInputData(System *sys, int numsample)
{
    Eigen::MatrixXd inputs = (Eigen::MatrixXd::Random(sys->numstate+sys->numinput, numsample).array()+1).matrix()/2;
    double width, minval, maxval;
    
    for(int ii = 0; ii < sys->ranges.rows(); ii++)
    {
        minval = sys->ranges(ii,0);
        maxval = sys->ranges(ii,1);
        width = maxval - minval;
        inputs.row(ii) = ((inputs.row(ii)*width).array() + minval).matrix();
    }

    return inputs;
}

Eigen::MatrixXd generateOutputData(System *sys, Eigen::MatrixXd inputs)
{
    long numsample = inputs.cols();
    Eigen::MatrixXd state = inputs.block(0, 0, sys->numstate, numsample);
    Eigen::MatrixXd control = inputs.block(sys->numstate, 0, sys->numinput, numsample);
    Eigen::MatrixXd outputs = sys->dynamics(state, control);
    return outputs;
}

Eigen::MatrixXd runILQR(System *sys, Eigen::MatrixXd start, Eigen::MatrixXd goal, bool isnet)
{
    int maxiter = 1000;
    double controlchange = 0;
    double enderror = 0;
    double lr = 0.1;
    int executesteps = 10;
    int horizonsteps = 100;
    int totalsteps = 1000;
    double q = 0.001;
    double qf = 1;
    double r = 0.001;
    
    Eigen::MatrixXd xu = Eigen::MatrixXd::Zero(sys->numinput+sys->numstate, totalsteps+horizonsteps+1);
    xu.block(0,0,sys->numstate,1) = start;
    Eigen::MatrixXd plan = Eigen::MatrixXd::Zero(sys->numinput+sys->numstate, horizonsteps+1);
    plan.block(0,0,sys->numstate,1) = start;
    Eigen::MatrixXd exec = Eigen::MatrixXd::Zero(sys->numinput+sys->numstate, executesteps+1);
    exec.block(0,executesteps,sys->numstate,1) = start;
    
    ILQR solver = ILQR(sys, maxiter, controlchange, enderror, lr, executesteps, horizonsteps, totalsteps, q, qf, r, isnet);
    
    int startcol = 0;
    
    for(int step = 0; step < totalsteps; step += executesteps)
    {
        plan = solver.solve(exec.block(0,executesteps,sys->numstate,1), goal, Eigen::MatrixXd::Zero(sys->numinput,horizonsteps));
        // plan = solver.solve(plan.block(0, 0, sys->numstate, 1), goal, Eigen::MatrixXd::Zero(sys->numinput,horizonsteps));
        // plan = solver.solve(plan.block(0, 0, sys->numstate, 1), goal, plan.block(sys->numstate, 0, sys->numinput, horizonsteps));
        
        exec = solver.execute(plan.block(0,0,sys->numstate,executesteps+1), plan.block(sys->numstate,0,sys->numinput,executesteps), exec.block(0,executesteps,sys->numstate,1));
        
        xu.block(0,startcol,exec.rows(),exec.cols()) = exec;
        startcol += executesteps;
        plan.block(0,0,plan.rows(),plan.cols()-executesteps) = plan.block(0,executesteps,plan.rows(),plan.cols()-executesteps);
        plan.block(0,plan.cols()-executesteps,plan.rows(),executesteps) = Eigen::MatrixXd::Zero(plan.rows(), executesteps);
    }
    
    return xu.block(0,0,xu.rows(),totalsteps+1);
}

Eigen::MatrixXd findeq(System *sys, std::string filename)
{
    Eigen::MatrixXd inputs = generateInputData(sys,1000);
    Eigen::MatrixXd result = sys->findequilibria(inputs,1000,0.0000001,1);
    Eigen::MatrixXd diff = Eigen::MatrixXd::Zero(1,result.cols());
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(sys->numinput,1);
    for(int ii = 0; ii < result.cols(); ii++)
    {
        diff(0,ii) = (sys->dynamics(result.col(ii),u) - result.col(ii)).norm();
    }
    std::cout << "Mean error of found points is: " << diff.array().mean() << "\n";
    matrixToCSV(result,filename);
    return result;
}

// Eigen::MatrixXd findpath(System *sys, Eigen::MatrixXd start, Eigen::MatrixXd goal)
// {

// }

int main(int argc, char *argv[])
{
    srand(uint(time(0)));
    // srand(0);

    // std::cout << "entered " << argc << " arguments\n";
    // for(int i = 0; i < argc; i++) {std::cout << argv[i] << "\n";}    
    // std::cout << "Hello, World!\n";

    if(argc == 1) {return 0;}

    if(std::string(argv[1]) == "invpen") 
    {
        std::cout << "Inverted Pendulum\n";
        InvertedPendulum invpen = InvertedPendulum();

        if(std::string(argv[2]) == "gendata")
        {
            int numsample = std::stoi(argv[3]);
            std::cout << "generating " << numsample << " i/o pairs for inverted pendulum\n";
            Eigen::MatrixXd x = generateInputData(&invpen,numsample);
            Eigen::MatrixXd y = generateOutputData(&invpen,x);
            Eigen::MatrixXd data = Eigen::MatrixXd::Zero(x.rows()+y.rows(),x.cols());
            data.block(0,0,x.rows(),x.cols()) = x;
            data.block(x.rows(),0,y.rows(),y.cols()) = y;
            matrixToCSV(data, std::string(argv[4])+"_trainingdata.csv");
            std::cout << "Saved training data to " << std::string(argv[4])+"_trainingdata.csv\n";
        } 

        Eigen::MatrixXd start = Eigen::MatrixXd::Zero(2,1);
        start(0,0) = acos(0)*2.0;
        Eigen::MatrixXd goal = Eigen::MatrixXd::Zero(2,1);

        if(std::string(argv[2]) == "net")
        {
            std::cout << "Network\n";
            NeuralNetwork net = NeuralNetwork(std::string(argv[3]));
            NetworkSystem netsys = NetworkSystem(net,&invpen);

            if(std::string(argv[4]) == "ilqr")
            {
                std::cout << "iLQR\n";
                Eigen::MatrixXd result = runILQR(&netsys, start, goal, true);
                matrixToCSV(result,"invpen/net_invpen_ilqr_result.csv");
                std::cout << "Saved result to invpen/net_invpen_ilqr_result.csv\n";
            }
            else if(std::string(argv[4]) == "test")
            {
                std::cout << "Test Control\n";
                Eigen::MatrixXd xu = csvToMatrix(std::string(argv[5]));
                Eigen::MatrixXd u = xu.block(invpen.numstate,0,invpen.numinput,xu.cols()-1);
                // std::cout << netsys.dynamics(start,u.col(0)) << "\n";
                Eigen::MatrixXd result = netsys.testcontrol(start,u);
                matrixToCSV(result,"invpen/net_invpen_test_result.csv");
                std::cout << "Saved test result to invpen/net_invpen_test_result.csv\n";
            }
            else if(std::string(argv[4]) == "eq")
            {
                std::cout << "Finding equilibria\n";
                Eigen::MatrixXd result = findeq(&netsys,"invpen/invpen_net_eq.csv");
                // std::cout << result << "\n";
            }
        }

        else if(std::string(argv[2]) == "ilqr")
        {
            std::cout << "iLQR\n";
            Eigen::MatrixXd result = runILQR(&invpen, start, goal, false);
            matrixToCSV(result,"invpen/invpen_ilqr_result.csv");
            std::cout << "Saved result to invpen/invpen_ilqr_result.csv\n";
        }

        else if(std::string(argv[2]) == "eq")
        {
            std::cout << "Finding equilibria\n";
            Eigen::MatrixXd result = findeq(&invpen,"invpen/invpen_eq.csv");
            // std::cout << result << "\n";
        }
    }

    if(std::string(argv[1]) == "block")
    {
        std::cout << "Sliding Block\n";
        SlidingBlock sldblk = SlidingBlock();

        if(std::string(argv[2]) == "gendata")
        {
            int numsample = std::stoi(argv[3]);
            std::cout << "generating " << numsample << " i/o pairs for sliding block\n";
            Eigen::MatrixXd x = generateInputData(&sldblk,numsample);
            Eigen::MatrixXd y = generateOutputData(&sldblk,x);
            Eigen::MatrixXd data = Eigen::MatrixXd::Zero(x.rows()+y.rows(),x.cols());
            data.block(0,0,x.rows(),x.cols()) = x;
            data.block(x.rows(),0,y.rows(),y.cols()) = y;
            matrixToCSV(data, std::string(argv[4])+"_trainingdata.csv");
            std::cout << "Saved training data to " << std::string(argv[4])+"_trainingdata.csv\n";
        }

        Eigen::MatrixXd start = Eigen::MatrixXd::Zero(2,1);
        start(0,0) = 5;
        Eigen::MatrixXd goal = Eigen::MatrixXd::Zero(2,1);

        if(std::string(argv[2]) == "ilqr")
        {
            std::cout << "iLQR\n";
            Eigen::MatrixXd result = runILQR(&sldblk, start, goal, false);
            matrixToCSV(result,"block/block_ilqr_result.csv");
            std::cout << "Saved result to block/block_ilqr_result.csv\n";
        }

    }
            
    return 0;
}
