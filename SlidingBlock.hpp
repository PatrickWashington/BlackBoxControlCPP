

#ifndef SlidingBlock_hpp
#define SlidingBlock_hpp

#include "System.hpp"
#include "Eigen/Dense"
#include<iostream>

class SlidingBlock : public System
{
public:
    double m;
    double f;
    
    SlidingBlock();
    SlidingBlock(double mass, double friction, double sim, double step);
    
    Eigen::MatrixXd dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd linearize(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
    
    Eigen::MatrixXd execute(Eigen::MatrixXd state, Eigen::MatrixXd control) override;
};

#endif /* InvertedPendulum_hpp */