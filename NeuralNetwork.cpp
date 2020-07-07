//
//  NeuralNetwork.cpp
//  BlackBoxControl
//
//  Created by Patrick Washington on 6/1/20.
//  Copyright © 2020 Patrick Washington. All rights reserved.
//

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork() {};

NeuralNetwork::NeuralNetwork(std::string filename)
{
    std::ifstream file;
    std::string line, substr;
    int item = 0;
    int start = 0;
    
    file.open(filename,std::ios::in);
    if(file.is_open())
    {
        while(true)
        {
            if(item==0)
            {
                std::getline(file,line);
                std::stringstream ss(line);
                while(ss.good())
                {
                    getline(ss, substr, ',');
                    inputsize = stoi(substr);
                    getline(ss, substr, ',');
                    hiddensize = stoi(substr);
                    getline(ss, substr, ',');
                    outputsize = stoi(substr);
                    getline(ss, substr, ',');
                    numhidden = stoi(substr);
                    getline(ss, substr, ',');
                    if(substr[0] == 's') { std::cout << "set sigmoid\n"; activation = 0; }
                    else if(substr[0] == 't') {std::cout << "set tanh\n"; activation = 3; }
                    
                    inweights.resize(hiddensize, inputsize);
                    inbiases.resize(hiddensize,1);
                    hiddenweights.resize(hiddensize,hiddensize*numhidden);
                    hiddenbiases.resize(hiddensize,numhidden);
                    outweights.resize(outputsize, hiddensize);
                    outbiases.resize(outputsize,1);
                }
                item++;
            }
            else if (item == 1)
            {
                for(int row = 0; row < hiddensize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    for(int col = 0; col < inputsize; col++)
                    {
                        getline(ss, substr, ',');
                        inweights(row,col) = stof(substr);
                    }
                }
                item++;
            }
            else if (item == 2)
            {
                for(int row = 0; row < hiddensize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    getline(ss, substr, ',');
                    inbiases(row,0) = stof(substr);
                }
                item++;
            }
            else if (item == 3+2*numhidden)
            {
                for(int row = 0; row < outputsize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    for(int col = 0; col < hiddensize; col++)
                    {
                        getline(ss, substr, ',');
                        outweights(row,col) = stof(substr);
                    }
                }
                item++;
            }
            else if (item == 3+2*numhidden+1)
            {
                for(int row = 0; row < outputsize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    getline(ss, substr, ',');
                    outbiases(row,0) = stof(substr);
                }
                break;
            }
            else if (item%2 == 1)
            {
                start = ((item-1)/2-1) * hiddensize;
                for(int row = 0; row < hiddensize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    for(int col = start; col < start+hiddensize; col++)
                    {
                        getline(ss, substr, ',');
                        hiddenweights(row,col) = stof(substr);
                    }
                }
                item++;
            }
            else
            {
                start = (item-2)/2 - 1;
                for(int row = 0; row < hiddensize; row++)
                {
                    std::getline(file,line);
                    std::stringstream ss(line);
                    getline(ss, substr, ',');
                    hiddenbiases(row,start) = stof(substr);
                }
                item++;
            }
        }
        item++;
    }
    file.close();
    
    inshift = Eigen::MatrixXd::Zero(inputsize,1); //SHOULD PROBABLY FEED THIS IN AS A FIELD IN THE CSV
    inscale = Eigen::MatrixXd::Ones(inputsize, 1);
    outshift = Eigen::MatrixXd::Zero(outputsize,1);
    outscale = Eigen::MatrixXd::Ones(outputsize, 1);
}

NeuralNetwork::NeuralNetwork(int in, int out, int hidden, int layer, int act)
{
    inputsize = in;
    outputsize = out;
    hiddensize = hidden;
    numhidden = layer;
    activation = act;
    inweights = Eigen::MatrixXd::Random(hiddensize,inputsize);
    inbiases = Eigen::MatrixXd::Zero(hiddensize,1);
    hiddenweights = Eigen::MatrixXd::Random(hiddensize,numhidden*hiddensize);
    hiddenbiases = Eigen::MatrixXd::Zero(hiddensize,numhidden);
    outweights = Eigen::MatrixXd::Random(outputsize,hiddensize);
    outbiases = Eigen::MatrixXd::Zero(outputsize,1);
    
    inshift = Eigen::MatrixXd::Zero(inputsize,1);
    inscale = Eigen::MatrixXd::Ones(inputsize, 1);
    outshift = Eigen::MatrixXd::Zero(outputsize,1);
    outscale = Eigen::MatrixXd::Ones(outputsize, 1);
    
}

Eigen::MatrixXd NeuralNetwork::evaluate(Eigen::MatrixXd inputs)
{
    long numinputs = inputs.cols();
    Eigen::MatrixXd output;
        
    
    inputs = ((inputs + inshift.replicate(1,numinputs)).array() * inscale.replicate(1,numinputs).array()).matrix();
    
    output = activationfun(inweights * inputs + inbiases.replicate(1,numinputs));
    
    if(checkmatrix(output)) {std::cout << "issue at 1\n";}
    if(checkmatrix(inweights * inputs + inbiases.replicate(1,numinputs))) {std::cout << "issue at 1.5\n";}
    if(checkmatrix(inweights)) {std::cout << "issue in inweights\n";}
    if(checkmatrix(inputs)) {std::cout << "issue in inputs\n";}
    if(checkmatrix(inbiases)) {std::cout << "issue in inbiases\n";}
    
    for(int ii = 0; ii < numhidden; ii++)
    {
        output = activationfun(hiddenweights.block(0,hiddensize*ii,hiddensize,hiddensize)*output + hiddenbiases.block(0,ii,hiddensize,1).replicate(1,numinputs));
    }
    if(checkmatrix(output)) {std::cout << "issue at 2\n";}
    
    output = outweights*output + outbiases.replicate(1,numinputs);
    if(checkmatrix(output)) {std::cout << "issue at 3\n";}
    
    output = (output.array() * outscale.replicate(1,numinputs).array()).matrix() + outshift.replicate(1,numinputs);
    
    return output;
}

Eigen::MatrixXd NeuralNetwork::evaluate(Eigen::MatrixXd state, Eigen::MatrixXd control)
{
    assert(state.cols()==control.cols());
    Eigen::MatrixXd inputs = Eigen::MatrixXd::Zero(state.rows()+control.rows(), state.cols());
    inputs.block(0,0,state.rows(),state.cols()) = state;
    inputs.block(state.rows(),0,control.rows(),state.cols()) = control;
    return evaluate(inputs);
}


void NeuralNetwork::train(Eigen::MatrixXd inputs, Eigen::MatrixXd actual, int EPOCH, int BATCH, double lr, double decay)
{
//    if(BATCH > 1)
//    {
//        std::cout << "Only working with batch size = 1 right now. Exiting...\n";
//        return;
//    }
    
//    for(int ii = 0; ii < inputsize; ii++)
//    {
//        inshift(ii,0) = -inputs.row(ii).array().mean();
//        inscale(ii,0) = 1.0/(((inputs.row(ii).array() + inshift(ii,0)).matrix().maxCoeff() + (inputs.row(ii).array() + inshift(ii,0)).matrix().minCoeff())/2.0);
//    }
    
//    for(int ii = 0; ii < outputsize; ii++)
//    {
//        outshift(ii,0) = -outputs.row(ii).array().mean();
//        outscale(ii,0) = 1.0/(((outputs.row(ii).array() + outshift(ii,0)).matrix().maxCoeff() + (outputs.row(ii).array() + outshift(ii,0)).matrix().minCoeff())/2.0);
//    }
    
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<double> dist(0,1);
    
    if(checkmatrix(inputs)) {std::cout << "issue in inputs\n";}
    if(checkmatrix(actual)) {std::cout << "issue in outputs\n";}
    if(checkmatrix(evaluate(inputs))) {std::cout << "issue in prediction\n";}
    
    long numtrain = 8*inputs.cols()/10;
    long numval = inputs.cols()/10;
    long numtest = inputs.cols()/10;
    
    Eigen::MatrixXd xtrain = inputs.block(0,0,inputsize,numtrain);
    Eigen::MatrixXd ytrain = actual.block(0,0,outputsize,numtrain);
    Eigen::MatrixXd xval = inputs.block(0,numtrain,inputsize,numval);
    Eigen::MatrixXd yval = actual.block(0,numtrain,outputsize,numval);
    Eigen::MatrixXd xtest = inputs.block(0,numtrain+numval,inputsize,numtest);
    Eigen::MatrixXd ytest = actual.block(0,numtrain+numval,outputsize,numtest);
    
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(inputs.rows(),1);
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(actual.rows(),1);
    Eigen::MatrixXd dJdy = Eigen::MatrixXd::Zero(outputsize,1);
    Eigen::MatrixXd dJdydiag = Eigen::MatrixXd::Zero(outputsize,outputsize);
    
    Eigen::MatrixXd dydz, dzk1dzk, dzdw;
    
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(hiddensize,numhidden+1);
    
    Eigen::MatrixXd outweightdiff = Eigen::MatrixXd::Zero(outweights.rows(),outweights.cols());
    Eigen::MatrixXd outbiasdiff = Eigen::MatrixXd::Zero(outbiases.rows(),1);
    Eigen::MatrixXd hiddenweightdiff = Eigen::MatrixXd::Zero(hiddenweights.rows(),hiddenweights.cols());
    Eigen::MatrixXd hiddenbiasdiff = Eigen::MatrixXd::Zero(hiddenbiases.rows(),hiddenbiases.cols());
    Eigen::MatrixXd inweightdiff = Eigen::MatrixXd::Zero(inweights.rows(),inweights.cols());
    Eigen::MatrixXd inbiasdiff = Eigen::MatrixXd::Zero(inbiases.rows(),inbiases.cols());
    
    long numbatch = numtrain / BATCH;

    std::cout << "Training loss starting at " << validate(xtrain, ytrain) << "\n";
    std::cout << "Validation loss starting at " << validate(xval,yval) << "\n";
    std::cout << "Test loss starting at " << validate(xtest,ytest) << "\n";
    
    for(int epoch = 0; epoch < EPOCH; epoch++)
    {
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> perm(numtrain);
        perm.setIdentity();
        std::shuffle(perm.indices().data(),perm.indices().data()+perm.indices().size(),std::default_random_engine(0));
//        std::cout << perm << "\n";
//        std::cout << perm.rows() << " " << perm.cols() << "\n";
        
        xtrain = xtrain * perm;
        ytrain = ytrain * perm;
        
        for(int batch = 0; batch < numbatch; batch++)
        {
            outweightdiff *= 0;
            outbiasdiff *= 0;
            hiddenweightdiff *= 0;
            hiddenbiasdiff *= 0;
            inweightdiff *= 0;
            inbiasdiff *= 0;
            
            for(int sample = 0; sample < BATCH; sample++)
            {
                x = xtrain.col(batch*BATCH+sample);
    //            std::cout << x.transpose() << "\n\n";
                e = evaluate(x) - ytrain.col(batch*BATCH+sample);
    //            std::cout << e.transpose() << "\n\n";
                
    //            if((e.array().abs() > 1e+10).any())
    //            {
    //                std::cout << ytrain.col(batch).transpose() << "\n\n";
    //                std::cout << hiddenweights << "\n\n";
    //                std::cout << hiddenbiases << "\n\n";
    //            }
                
                if(checkmatrix(x)) {std::cout << "issue in x\n";}
                if(checkmatrix(ytrain.col(batch))) {std::cout << "issue in y\n";}
                if(checkmatrix(e)) {std::cout << "issue in e\n";}
                
                dJdy = e.transpose() * 2.0/(outputsize*BATCH);
                
    //            std::cout << dJdy << "\n";
    //            std::cout << "\n";
                
                z *= 0;
                z.col(0) = inweights * x + inbiases;
                for(int ii = 0; ii < numhidden; ii++)
                {
                    z.col(ii+1) = hiddenweights.block(0,hiddensize*ii,hiddensize,hiddensize)*activationfun(z.col(ii)) + hiddenbiases.col(ii);
                }
                
                if(checkmatrix(dJdy)) {std::cout << "issue in dJdy\n";}
    //            if(checkmatrix(z)) {std::cout << "issue in z\n";}
                
                dydz = Eigen::MatrixXd::Identity(outputsize,outputsize);
                
    //            if(checkmatrix(dydz)) {std::cout << "issue in dydz\n";}
                
                for(int r = 0; r < outputsize; r++)
                {
                    for(int c = 0; c < hiddensize; c++)
                    {
                        dzdw = Eigen::MatrixXd::Zero(outputsize,1);
                        dzdw.block(r,0,1,1) = activationfun(z.block(c,numhidden,1,1));
                        outweightdiff.block(r,c,1,1) += dJdy * dydz * dzdw;
                    }
                }
                
                outbiasdiff += (dJdy * dydz).transpose();
    //            std::cout << outweightdiff.array().abs().mean() << "\n";
    //            std::cout << outbiasdiff.array().abs().mean() << "\n";
    //            std::cout << "\n";
                
                dzk1dzk = outweights * col2diag(activationdiff(z.col(numhidden)));
                dydz *= dzk1dzk;
                
                for(int kk = numhidden-1; kk >= 0; kk--)
                {
                    for(int r = 0; r < hiddensize; r++)
                    {
                        for(int c = 0; c < hiddensize; c++)
                        {
                            dzdw = Eigen::MatrixXd::Zero(hiddensize,1);
                            dzdw.block(r,0,1,1) = activationfun(z.block(c,kk,1,1));
                            hiddenweightdiff.block(r,hiddensize*kk+c,1,1) += dJdy * dydz * dzdw;
                        }
                    }
                    
                    hiddenbiasdiff.col(kk) += (dJdy * dydz).transpose();
                    
                    dzk1dzk = hiddenweights.block(0,hiddensize*kk,hiddensize,hiddensize) * col2diag(activationdiff(z.col(kk)));
                    dydz *= dzk1dzk;
                }
                
    //            std::cout << hiddenweightdiff.array().abs().mean() << "\n";
    //            std::cout << hiddenbiasdiff.array().abs().mean() << "\n";
    //            std::cout << "\n";
                
                
                for(int r = 0; r < hiddensize; r++)
                {
                    for(int c = 0; c < inputsize; c++)
                    {
                        dzdw = Eigen::MatrixXd::Zero(hiddensize,1);
                        dzdw.block(r,0,1,1) = x.block(c,0,1,1);
                        inweightdiff.block(r,c,1,1) += dJdy * dydz * dzdw;
                    }
                }
                
                inbiasdiff += (dJdy * dydz).transpose();
    //            std::cout << inweightdiff.array().abs().mean() << "\n";
    //            std::cout << inbiasdiff.array().abs().mean() << "\n";
    //            std::cout << "\n";
            }
            
            if(checkmatrix(outweightdiff)){std::cout<<"issue in outweightdiff\n";}
            if(checkmatrix(outbiasdiff)){std::cout<<"issue in outbiasdiff\n";}
            if(checkmatrix(hiddenweightdiff)){std::cout<<"issue in hiddenweightdiff\n";}
            if(checkmatrix(hiddenbiasdiff)){std::cout<<"issue in hiddenbiasdiff\n";}
            if(checkmatrix(inweightdiff)){std::cout<<"issue in inweightdiff\n";}
            if(checkmatrix(inbiasdiff)){std::cout<<"issue in inbiasdiff\n";}
            
            outweightdiff = capchanges(outweightdiff, 1.0/lr);
            outbiasdiff = capchanges(outbiasdiff, 1.0/lr);
            hiddenweightdiff = capchanges(hiddenweightdiff, 1.0/lr);
            hiddenbiasdiff = capchanges(hiddenbiasdiff, 1.0/lr);
            inweightdiff = capchanges(inweightdiff, 1.0/lr);
            inbiasdiff = capchanges(inbiasdiff, 1.0/lr);
            
            
            outweights -= lr * outweightdiff;
            outbiases -= lr * outbiasdiff;
            hiddenweights -= lr * hiddenweightdiff;
            hiddenbiases -= lr * hiddenbiasdiff;
            inweights -= lr * inweightdiff;
            inbiases -= lr * inbiasdiff;
            
            if(checkmatrix(outweights)){std::cout<<"issue in outweights\n";}
            if(checkmatrix(outbiases)){std::cout<<"issue in outbiases\n";}
            if(checkmatrix(hiddenweights)){std::cout<<"issue in hiddenweights\n";}
            if(checkmatrix(hiddenbiases)){std::cout<<"issue in hiddenbiases\n";}
            if(checkmatrix(inweights)){std::cout<<"issue in inweights\n";}
            if(checkmatrix(inbiases)){std::cout<<"issue in inbiases\n";}
//            std::cout << "\n";
            
//            if (epoch > 0)
//            {
//                std::cout << outbiasdiff * lr << "\n\n";
//                std::cout << outbiases << "\n\n\n\n";
//            }
//            std::cout << validate(xval,yval) << "\n";
        }
        
//        std::cout << inweightdiff << "\n\n";
        
        std::cout << "Through epoch " << epoch+1 << " of " << EPOCH << " with training loss " << validate(xtrain, ytrain) << " and validation loss " << validate(xval,yval) << " and test loss " << validate(xtest,ytest) << "\n";
        
        lr *= decay;
    }
    
    std::cout << "Done Training\n";
    std::cout << "Test loss is " << validate(xtest, ytest) << "\n";
}

double NeuralNetwork::validate(Eigen::MatrixXd inputs, Eigen::MatrixXd actual)
{
    return mseLoss(evaluate(inputs), actual);
}

double NeuralNetwork::mseLoss(Eigen::MatrixXd predicted, Eigen::MatrixXd actual)
{
    return (predicted - actual).array().square().mean();
}

Eigen::MatrixXd NeuralNetwork::activationfun(Eigen::MatrixXd x)
{
    if(activation == 0) { return Sigmoid(x); }
    else if(activation == 1){ return ReLU(x); }
    else if(activation == 2){ return LeakyReLU(x); }
    else if(activation == 3){ return Tanh(x); }
    else { return x; }
}

Eigen::MatrixXd NeuralNetwork::activationdiff(Eigen::MatrixXd x)
{
    if(activation == 0) {return diffSigmoid(x); }
    else if(activation == 1){ return diffReLU(x); }
    else if(activation == 2) { return diffLeakyReLU(x); }
    else if(activation == 3) { return diffTanh(x); }
    else { return Eigen::MatrixXd::Ones(x.rows(),x.cols()); }
}

Eigen::MatrixXd NeuralNetwork::ReLU(Eigen::MatrixXd x)
{
    Eigen::MatrixXd out = (x.array().max(0)).matrix();
    if(checkmatrix(x)) {std::cout << x << "\n\n";}
    if(checkmatrix(out)) {
            std::cout << x << "\n\n";
            std::cout << out << "\n\n";
            std::cout << inweights << "\n\n";
            std::cout << inbiases << "\n\n";
    }
    return out;
}

Eigen::MatrixXd NeuralNetwork::LeakyReLU(Eigen::MatrixXd x)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(x.rows(),x.cols());
    for(int r = 0; r < x.rows(); r++)
    {
        for(int c = 0; c < x.cols(); c++)
        {
            if(x(r,c)>0) { out(r,c) = x(r,c); }
            else { out(r,c) = 0.01 * x(r,c); }
        }
    }
    return out;
}

Eigen::MatrixXd NeuralNetwork::Sigmoid(Eigen::MatrixXd x)
{
   Eigen::MatrixXd out = (x.array().exp() / (1 + x.array().exp())).matrix();
    // Eigen::MatrixXd out = (1 / (1 + (-x).array().exp())).matrix();
//    if(checkmatrix(x)) {std::cout << x << "\n\n";}
//    if(checkmatrix(out)) {
//        std::cout << x << "\n\n";
//        std::cout << out << "\n\n";
//        std::cout << inweights << "\n\n";
//        std::cout << inbiases << "\n\n";
//    }
    return out;
}

Eigen::MatrixXd NeuralNetwork::Tanh(Eigen::MatrixXd x)
{
    Eigen::MatrixXd out = ((x.array().exp() - (-x).array().exp()) / (x.array().exp() + (-x).array().exp())).matrix();

    return out;
}

Eigen::MatrixXd NeuralNetwork::diffReLU(Eigen::MatrixXd x)
{
    return x.array().max(0).min(1).ceil().matrix();
}

Eigen::MatrixXd NeuralNetwork::diffLeakyReLU(Eigen::MatrixXd x)
{
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(x.rows(),x.cols());
    for(int r = 0; r < x.rows(); r++)
    {
        for(int c = 0; c < x.cols(); c++)
        {
            if(x(r,c)>0) { out(r,c) = 1; }
            else { out(r,c) = 0.01; }
        }
    }
    return out;
}

Eigen::MatrixXd NeuralNetwork::diffSigmoid(Eigen::MatrixXd x)
{
    Eigen::MatrixXd sig = Sigmoid(x);
    return (sig.array() * (1 - sig.array())).matrix();
//    return ((x.array().exp() / (1 + x.array().exp())) * (1 - (x.array().exp() / (1 + x.array().exp())))).matrix();
}

Eigen::MatrixXd NeuralNetwork::diffTanh(Eigen::MatrixXd x)
{
    Eigen::MatrixXd t = Tanh(x);
    return (1 - t.array()*t.array()).matrix();
}

Eigen::MatrixXd NeuralNetwork::col2diag(Eigen::MatrixXd mat)
{
    assert(mat.cols() == 1);
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(mat.rows(),mat.rows());
    for(int ii = 0; ii < mat.rows(); ii++) { d(ii,ii) = mat(ii,0); }
    return d;
}

Eigen::MatrixXd NeuralNetwork::diag2col(Eigen::MatrixXd mat)
{
    assert(mat.rows() == mat.cols());
    Eigen::MatrixXd c = Eigen::MatrixXd::Zero(mat.rows(),1);
    for(int ii = 0; ii < mat.rows(); ii++) { c(ii,0) = mat(ii,ii); }
    return c;
}

bool NeuralNetwork::checkmatrix(Eigen::MatrixXd mat)
{
    return !(mat.array()==mat.array()).all();
//    return (mat.array().abs() < 0.0001).all();
//    return (mat.array().abs() > 1e+50).any();
}

Eigen::MatrixXd NeuralNetwork::capchanges(Eigen::MatrixXd mat, double lim)
{
    mat = mat.array().min(lim).max(-lim).matrix();
    return mat;
}
