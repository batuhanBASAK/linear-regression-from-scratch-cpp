#ifndef LINEAR_REGRESSION_H_
#define LINEAR_REGRESSION_H_


#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>


namespace lrc {


    class LinearRegressionClassifier {
    public:
        double predict(std::vector<double> x) const;
        std::vector<double> predictAll(std::vector< std::vector <double> > X) const;
        void train(std::vector<std::vector <double> > X, std::vector<double> y, size_t epochs, double lr);
        double computeLoss(std::vector<double> y_pred, std::vector<double> y_true) const;

        void printW() const;
    private:
        double bias;
        std::vector<double> w; // weights vector
        size_t dimension; // dimension of vector w

        void initializeW();
        void train_sample(std::vector<double> X, double y, double lr);
    };

}

#endif
