#include "SlidingBlock.hpp"

SlidingBlock::SlidingBlock()
{
    numstate = 2;
    numinput = 1;
    m = 1;
    f = 0;
    ranges.resize(numstate+numinput,2);
    ranges.row(0) << -10,10;
    ranges.row(1) << -20,20;
    ranges.row(2) << -10,10;
    simdt = 0.001;
    stepdt = 0.01;
}

SlidingBlock::SlidingBlock(double mass, double friction, double sim, double step)
{
    numstate = 2;
    numinput = 1;
    m = mass;
    f = friction;
    ranges.resize(numstate+numinput,2);
    ranges.row(0) << -10,10;
    ranges.row(1) << -20,20;
    ranges.row(2) << -10,10;
    simdt = sim;
    stepdt = step;
}

Eigen::MatrixXd SlidingBlock::dynamics(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    Eigen::MatrixXd x, xd;
    Eigen::MatrixXd F = control.row(0);
    Eigen::MatrixXd nextstate = Eigen::MatrixXd::Zero(state.rows(),state.cols()) + state;
    Eigen::MatrixXd rates = Eigen::MatrixXd::Zero(state.rows(),state.cols());
    
//    std::cout << "got here\n";
    
    double t = 0;
    while (t < stepdt-simdt/100)
    {
        x = nextstate.row(0);
        xd = nextstate.row(1);
        rates.row(0) = xd;
        rates.row(1) = F/m - f/m*xd.array().sign().matrix();
        nextstate += rates*simdt;
        t += simdt;
    }
    
    return nextstate;
}

Eigen::MatrixXd SlidingBlock::linearize(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(state.rows(),state.rows()+control.rows());

    // A, continuous
    M(0,0) = 0;
    M(0,1) = 1;
    M(1,0) = 0;
    if(state(1,0) > 0) { M(1,1) = -f/m; }
    else if(state(1,0) < 0) { M(1,1) = f/m; }
    else { M(1,1) = 0; }
    
    // B, continuous
    M(0,2) = 0;
    M(1,2) = 1/m;
    
    // Discretize
    M *= stepdt;
    M.block(0,0,numstate,numstate) += Eigen::MatrixXd::Identity(numstate, numstate);
    
    return M;
}

Eigen::MatrixXd SlidingBlock::execute(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    return dynamics(state, control);
}
