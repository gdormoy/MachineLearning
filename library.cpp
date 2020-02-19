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

typedef struct MLP{
    double*** model; // [{{1, 2, 3}, {2, 1, 3}}, {{1, 3}, {2, 1}, {1 ,2}}, {{3}, {3}}] layersSize - 1
    int* layers; // [2, 3, 2, 1]
    int layersSize; // 4
}MLP;

extern "C"{

    DLLEXPORT double predict_linear_model(const double* model, const double* params, const int numberOfParams) {
        double result = 0;
        result += model[0];
        for(int y = 1; y < numberOfParams + 1; y++){
            result += params[y - 1] * model[y];
        }
        return result;
    }

    DLLEXPORT double predict_linear_class_model(const double* model, const double* params, const int numberOfParams) {
        return predict_linear_model(model, params, numberOfParams) > 0 ? 1 : -1;
    }

    DLLEXPORT double* create_linear_model(const int numberOfParams) {
        srand(time(nullptr));
        auto model = (double*) malloc(sizeof(double) * (numberOfParams + 1));
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = fRand(-1.0, 1.0);
        }
        return model;
    }

    DLLEXPORT void train_linear_class_model(double* model, double* dataset, double* expected_output, const int numberOfParams, const int datasetSize, double step, int epoch){
        int size = datasetSize / numberOfParams;
        MatrixXd X(size, numberOfParams + 1);
        for(int i = 0; i < size; i++){
            X(i, 0) = 1;
            for(int j = 1; j < numberOfParams + 1; j++){
                X(i,j) = dataset[i * numberOfParams + j - 1];
            }
        }

        MatrixXd Y(size, 1);
        for(int i = 0; i < size; i++){
            Y(i, 0) = expected_output[i];
        }

        MatrixXd w(numberOfParams + 1, 1);
        for(int i = 0; i < numberOfParams + 1; i++){
            w(i, 0) = model[i];
        }

        for(int i = 0; i < epoch; i++){
            int exemple_number = rand() % size;
            MatrixXd exemple(numberOfParams + 1, 1);
            auto exemple_array = new double[numberOfParams+1];

            for(int j = 0; j < numberOfParams + 1; j++){
                exemple(j, 0) = X(exemple_number, j);
                exemple_array[j] = X(exemple_number, j);
            }
            w = w + step * (Y(exemple_number, 0) - predict_linear_class_model(model, exemple_array, 2)) * exemple;
        }

        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = w(i, 0);
        }
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

        MatrixXd Y(size, 1);
        for(int i = 0; i < size; i++){
            Y(i, 0) = expected_output[i];
        }

        auto v = (((X.transpose() * X).inverse() * X.transpose()) * Y);

        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = v(i, 0);
        }
    }

    DLLEXPORT MLP * create_mlp_model(const int* layers, int layersSize){
        srand(time(nullptr));

        auto mlp = (MLP*) malloc(sizeof(MLP));
        mlp->layers = (int*) malloc(sizeof(int) * (layersSize - 1));
        mlp->layersSize = layersSize;
        for(int i = 0; i < layersSize; i++){
            mlp->layers[i] = layers[i];
        }
        cout << "bonjour a tous" << endl;
        mlp->model = (double***) malloc(sizeof(double**) * (layersSize - 1));
        for(int i = 0; i < layersSize - 1; i++){
            mlp->model[i] = (double**) malloc(sizeof(double*) * layers[i]);
            for(int j = 0; j < layers[i]; j++){
                mlp->model[i][j] = (double*) malloc(sizeof(double) * ((layers[i] + 1) * layers[i + 1]));
                for(int k = 0; k < (layers[i] * layers[i + 1]) ; k++){
                    mlp->model[i][j][k] = fRand(-1, 1);
                }
            }
        }
        return mlp;
    }


    DLLEXPORT int predict_mlp_class_model(MLP* mlp, const double* inputs){
        auto result_layers = new double*[mlp->layersSize - 1];
        double result = 0;

        for(int i = 0; i < mlp->layersSize - 1; i++){
            result_layers[i] = new double[mlp->layers[i]];
            for(int j = 0; j < mlp->layers[i]; j++){
                result_layers[i][j] = 0;
            }
        }

        for(int i = 0; i < mlp->layers[0]; i++){
            result_layers[0][i] = inputs[i];
        }

        for(int i = 1; i < mlp->layersSize - 2; i++){
            for(int j = 0; j < mlp->layers[i + 1]; j++){
                for(int k = 0; k < mlp->layers[i + 2]; k++){
                    result_layers[i][j] += mlp->model[i - 1][k][j] * result_layers[i - 1][j];
                }
            }
        }
        for(int i = 0; i < mlp->layers[mlp->layersSize - 2]; i++){
            cout << i << mlp->layers[mlp->layersSize - 2] << endl;
            result += result_layers[mlp->layersSize - 2][i];
        }
        return result > 0 ? 1 : -1;

    }
}

int main()
{
    auto dataset = new double[6] {-10.0, 0.0, 1.0, 0.0, 5.0, 1.0};
    auto expectedOutputs = new double[3] {1.0, 0.0, 1.0};

    /*cout << "bonjour";
    auto m = create_linear_model(2);
    train_linear_class_model(m, dataset, expectedOutputs, 2, 6, 0.0001 , 10000000);

    cout << "bonjour";
    cout << predict_linear_class_model(m, dataset, 2) << endl;
    cout << predict_linear_class_model(m, dataset + 2, 2) << endl;
    cout << predict_linear_class_model(m, dataset + 4, 2) << endl;

    for(int i = 0; i < 3; i++){
        cout << m[i] << endl;
    }*/

    MLP* mlp = create_mlp_model(new int[4] {2, 3, 2, 1}, 4);
    cout << mlp->model[1][1][0] << endl;
    cout << predict_mlp_class_model(mlp, new double[2] {2, 3}) << endl;
    std::cout << "feoizjfozeijfoizej" << endl;
    return 0;
}