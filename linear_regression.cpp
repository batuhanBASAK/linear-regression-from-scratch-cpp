#include "linear_regression.hpp"
#include <cstdlib>


namespace lrc {

    double LinearRegressionClassifier::predict(std::vector<double> x) const
    {

        if (x.size() != dimension)
        {
            std::cerr << "Invalid dimension!!!" <<std::endl;
            exit(EXIT_FAILURE);
        }

        double h = 0.0f;
        for (size_t i = 0; i < dimension; i++)
            h += x[i] * w[i];
        h += bias;

        return h;
    }


    std::vector<double> LinearRegressionClassifier::predictAll(std::vector< std::vector <double> > X) const 
    {
        std::vector<double> outputs;

        for (size_t i = 0; i < X.size(); i++)
            outputs.push_back(predict(X[i]));
        return outputs;
    }


    void LinearRegressionClassifier::train(std::vector<std::vector <double> > X, std::vector<double> y, size_t epochs, double lr) 
    {

        dimension = X[0].size();

        initializeW();


        for (size_t i = 0; i < epochs; i++)
        {
            // Backward Propagation
            for (size_t j = 0; j < X.size(); j++)
                train_sample(X[j], y[j], lr);

            // Compute loss 
            std::vector<double> y_pred = predictAll(X);
            double loss = computeLoss(y_pred, y);
            std::cout << "Epoch " << i+1 << ", loss: " << loss << std::endl;
        }
    }


    void LinearRegressionClassifier::train_sample(std::vector<double> X, double y, double lr)
    {

        if (X.size() != dimension)
        {
            std::cerr << "Invalid dimension!!!" <<std::endl;
            exit(EXIT_FAILURE);
        }


        double diff = predict(X) - y;

        for (size_t i = 0; i < dimension; i++)
            w[i] = w[i] - lr * diff * X[i];

        bias = bias - lr * diff;

    }

    double LinearRegressionClassifier::computeLoss(std::vector<double> y_pred, std::vector<double> y_true) const 
    {
        double loss = 0.0f;

        for (int i = 0; i < dimension; i++) {
            loss += pow( y_pred[i] - y_true[i], 2);
        }
        return loss / 2;
    }



    void LinearRegressionClassifier::initializeW()
    {
        // Create a random device and seed the generator
        std::random_device rd;  // Seed generator with a random value
        std::mt19937 gen(rd()); // Standard Mersenne Twister generator
        
        // Define the range [min, max) for the random double numbers
        double min = 0.75;
        double max = 1.25;
        
        // Create a distribution in the specified range
        std::uniform_real_distribution<> dis(min, max);
        
        for (size_t i = 0; i < dimension; i++)
            w.push_back(dis(gen));

        bias = dis(gen);
    }

    void LinearRegressionClassifier::printW() const
    {
        for (size_t i = 0; i < dimension; i++)
            std::cout << w[i] << " ";
        std::cout << bias << std::endl;
    }

}
