#include "Eigen/Dense"
#include "library.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
#if linux
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

extern "C"{
    DLLEXPORT double* create_linear_model(const int numberOfParams) {
        srand(time(nullptr));
        auto model = (double*) malloc(sizeof(double) * (numberOfParams + 1));
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = fRand(-1.0, 1.0);
        }
        return model;
    }

    DLLEXPORT void train_linear_class_model(double* model, double* dataset, double* expected_output){

    }

    DLLEXPORT void train_linear_model(double* model, double* dataset, double* expected_output, const int numberOfParams,  const int datasetSize) {
        int size = datasetSize/numberOfParams;
        MatrixXd X(size, numberOfParams + 1);
        for(int i = 0; i < size; i++){
            X(i, 0) = 1;
            for(int j = 1; j < numberOfParams + 1; j++){
                X(i,j) = dataset[i * numberOfParams + j - 1];
            }
        }

        std::cout << "X : " << X << endl;

        MatrixXd Y(size, 1);
        for(int i = 0; i < size; i++){
            Y(i, 0) = expected_output[i];
        }

        std::cout << "Y : " << Y << endl;
        auto v = (((X.transpose() * X).inverse() * X.transpose()) * Y);

        std::cout << "W : " << v << endl;
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = v(i, 0);
        }
    }

    DLLEXPORT double predict_linear_model(const double* model, const double* params, const int numberOfParams) {
        double result = 0;
        result += model[0];
        for(int y = 1; y < numberOfParams + 1; y++){
            result += params[y - 1] * model[y];
        }
        return result;
    }

    DLLEXPORT double predict_linear_class_model(const double* model, const double* params, const int numberOfParams) {
        return signbit(predict_linear_model(model, params, numberOfParams));
    }
}

int main()
{
    auto dataset = new double[6] {-10.0, 0.0, 1.0, 0.0, 5.0, 1.0};
    auto expectedOutputs = new double[3] {-30.0, 42.0, 1.0};

    auto m = create_linear_model(2);
    train_linear_model(m, dataset, expectedOutputs, 2, 6);

    cout << predict_linear_model(m, dataset, 2) << endl;
    cout << predict_linear_model(m, dataset + 2, 2) << endl;
    cout << predict_linear_model(m, dataset + 4, 2) << endl;

    std::cout << "feoizjfozeijfoizej" << endl;
    return 0;
}