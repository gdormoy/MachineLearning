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

    DLLEXPORT void* train_linear_class_model(double* model, double* dataset, double* expected_output){

    }

    DLLEXPORT void train_linear_model(double* model, double* dataset, double* expected_output, const int numberOfParams,  const int datasetSize) {
        int size = datasetSize/numberOfParams;
        MatrixXd X(size, numberOfParams + 1);
        for(int i = 0; i < size; i++){
            X(0, i) = 1;
            for(int j = 1; j < numberOfParams + 1; j++){
                X(i,j) = dataset[i * numberOfParams + j];
            }
        }

        VectorXd Y(size);
        for(int i = 0; i < size; i++){
            Y[i] = expected_output[i];
        }
        VectorXd v = (((X.transpose() * X).inverse() * X.transpose()) * Y).array();
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = v[i];
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