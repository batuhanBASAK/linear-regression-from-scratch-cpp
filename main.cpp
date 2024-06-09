#include "linear_regression.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>


int main()
{
    srand(time(NULL));

    lrc::LinearRegressionClassifier clf;

    std::vector< std::vector<double> > X;
    std::vector<double> y;
    std::vector <double> x;


    // Initialize Samples
    size_t sample_size = 100;
    size_t dimension = 3;
    for (size_t i = 0; i < sample_size; i++)
    {
        x.clear();
        double sum = 0.0f;
        for (size_t j = 0; j < dimension; j++)
        {
            double randNum = (double) (rand() % 100);
            x.push_back(randNum/100.0f);
            sum += randNum;
        }
        y.push_back(sum);
        X.push_back(x);
    }


    // Display samples
    std::cout << "-----------------------" << std::endl;
    std::cout << "Samples" << std::endl;
    for (size_t i = 0; i < sample_size; i++)
    {
        for (size_t j = 0; j < dimension; j++)
            std::cout << X[i][j] << " ";
        std::cout << y[i] << std::endl;
    }

    std::cout << "-----------------------" << std::endl;


    clf.train(X, y, 100, 0.01);



    std::cout << "-----------------------" << std::endl;
    std::cout << "Weights of linear regression model" << std::endl;
    clf.printW();

    return 0;
}
