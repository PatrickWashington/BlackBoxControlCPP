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
#include "NeuralNetwork.hpp"
#include "NetworkSystem.hpp"
#include "ILQR.hpp"
//#include "../../madplotlib/Madplotlib.h"
//#include "../../matplotlib-cpp-master/matplotlibcpp.h"

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
    int executesteps = 50;
    int horizonsteps = 500;
    int totalsteps = 500;
    double q = 0.01;
    double qf = 1;
    double r = 0.00001;
    
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
            matrixToCSV(x, std::string(argv[4])+"_indata.csv");
            std::cout << "Saved input data to " << std::string(argv[4])+"_indata.csv\n";
            matrixToCSV(y, std::string(argv[4])+"_outdata.csv");
            std::cout << "Saved output data to " << std::string(argv[4])+"_outdata.csv\n";
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
        }

        if(std::string(argv[2]) == "ilqr")
        {
            std::cout << "iLQR\n";
            Eigen::MatrixXd result = runILQR(&invpen, start, goal, false);
            matrixToCSV(result,"invpen/invpen_ilqr_result.csv");
            std::cout << "Saved result to invpen/invpen_ilqr_result.csv\n";
        }
    }
    
//    InvertedPendulum invpen = InvertedPendulum(1, 1, 1, 0.001, 0.01);
//    double pi = 2*acos(0.0);
//    Eigen::Matrix<double,2,1> start;
//    start << pi,0;
//    Eigen::MatrixXd goal = Eigen::MatrixXd::Zero(2, 1);
//    goal *= 0;
////    ILQR solver = ILQR(&invpen, 10, 0.01, 0.01, 0.1, 0.001, 0.01, 0.1, 1, 10, 0, 1, 0.0001);
//
//    NeuralNetwork net = NeuralNetwork("/Users/patrickwashington/Desktop/pendulumnet_csv.csv");
//    NetworkSystem netsys = NetworkSystem(net, &invpen);
//
////    Eigen::MatrixXd result = runILQR(&netsys, start, goal, true);
//    Eigen::MatrixXd result = runILQR(&invpen, start, goal, false);
//    matrixToCSV(result, "/Users/patrickwashington/Desktop/testfile.csv");
    
    // std::string one = "testing";
    // std::string two = "this";
    // std::cout << one+two << "\n";

    // InvertedPendulum invpen = InvertedPendulum(1,1,1,0.001,0.01);
    // NetworkSystem netsys;
    // NeuralNetwork net;

    // int systemchoice;
    // int basesystemchoice;
    // int runchoice;
    // std::string netfile;
    // std::string filestart;

    // std::cout << "Choose system type to use:\n\t(1) Inverted Pendulum\n\t(2) Network\n";
    // std::cin >> systemchoice;

    // if(systemchoice == 1)
    // {
    //     // idk
    // }

    // else if(systemchoice == 2)
    // {
    //     std::cout << "What base system should be used?\n\t(1) Inverted Pendulum\n";
    //     std::cin >> basesystemchoice;

    //     std::cout << "Enter file with network information: ";
    //     std::cin >> netfile;
    //     net = NeuralNetwork(netfile);

    //     if(basesystemchoice == 1)
    //     {
    //         netsys = NetworkSystem(net,&invpen);
    //     }
    // }


    // std::cout << "What do you want to do?\n\t(1) Run iLQR\n\t(2) Generate I/O data for training\n";
    // std::cin >> runchoice;

    // if(runchoice == 1)
    // {
    //     // run ilqr here
    // }
    // else if(runchoice == 2)
    // {
    //     Eigen::MatrixXd x;
    //     Eigen::MatrixXd y;

    //     if(systemchoice == 1)
    //     {
    //         x = generateInputData(&invpen, 100000);
    //         y = generateOutputData(&invpen, x);
    //     }

    //     std::cout << x.rows() << " " << x.cols() << "\n";
    //     std::cout << y.rows() << " " << y.cols() << "\n";

    //     std::cout << "Done generating training data\n";
    //     std::cout << "Enter file prefix:";
    //     std::cin >> filestart;
    //     matrixToCSV(x,filestart+"_indata.csv");
    //     std::cout << "Input data saved to " << filestart+"_indata.csv\n";
    //     matrixToCSV(y,filestart+"_outdata.csv");
    //     std::cout << "Output data saved to " << filestart+"_outdata.csv\n";
    // }

    // InvertedPendulum invpen = InvertedPendulum(1,1,1,0.001,0.01);
    // Eigen::MatrixXd x = generateInputData(&invpen, 10000);
    // Eigen::MatrixXd y = generateOutputData(&invpen, x);
    
    // NeuralNetwork net = NeuralNetwork(3, 2, 20, 4, 0);
//    net.train(x, y, 200, 100, 0.01, 1);
    
    // boost::cout << "test\n";

    // char pyname[] = "python3";
    // FILE * pyfile;
    // char pyfilename[] = "test.py";
    // pyfile = fopen(pyfilename, "r");
        
    // Py_SetProgramName(pyname);
    // Py_Initialize();
    // PyRun_SimpleFile(pyfile,pyfilename);
    // Py_Finalize();
        
    return 0;
}
