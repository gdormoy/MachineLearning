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
    double f = (double)rand() / (double) RAND_MAX;
    return fMin + f * (fMax - fMin);
}

typedef struct MLP{
    double*** model; // [{{1, 2, 3}, {2, 1, 3}}, {{1, 3}, {2, 1}, {1 ,2}}, {{3}, {3}}] layersSize - 1
    double** delta;
    double** result;
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
        //srand(time(nullptr));
        auto model = (double*) malloc(sizeof(double) * (numberOfParams + 1));
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = fRand(-1.0, 1.0);
        }
        return model;
    }

    DLLEXPORT void train_linear_class_model(double* model, double* dataset, double* expected_output, const int numberOfParams, const int datasetSize, double step, int epoch){

    int size = datasetSize / numberOfParams;
    for(int i = 0; i < epoch; i++){
        int exemple_number = rand() % size;

        auto exemple_input = dataset + exemple_number * numberOfParams;
        auto predicted = predict_linear_class_model(model, exemple_input, numberOfParams);
        auto machin =  step * (expected_output[exemple_number] - predicted);
        model[0] += machin;
        for(int j = 0; j < numberOfParams; j++){
            model[j+1] += machin * exemple_input[j];
        }
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
        mlp->layers = new int[layersSize];
        mlp->layersSize = layersSize;
        mlp->delta = (double**) malloc(sizeof(double*)*layersSize);
        mlp->result = (double**) malloc(sizeof(double*)*layersSize);

        for(int i = 0; i < layersSize; i++){
            mlp->layers[i] = layers[i];
            mlp->result[i] = new double[layers[i]];
            mlp->delta[i] = new double[layers[i]];
            for(int j = 0; j < layers[i]; j++){
                mlp->delta[i][j] = 0;
                mlp->result[i][j] = 0;
            }
        }

        mlp->model = (double***) malloc(sizeof(double**) * (layersSize - 1));
        for(int i = 0; i < layersSize - 1; i++){
            mlp->model[i] = (double**) malloc(sizeof(double*) * layers[i]);
            for(int j = 0; j < layers[i]; j++){
                mlp->model[i][j] = (double*) malloc(sizeof(double) * (layers[i + 1]));
                for(int k = 0; k < layers[i + 1] ; k++){
                    mlp->model[i][j][k] = fRand(-1, 1);
                }
            }
        }
        return mlp;
    }

    DLLEXPORT int predict_mlp_class_model(MLP* mlp, double* inputs){
        for(int i = 0; i < mlp->layersSize - 1; i++){
            for(int j = 0; j < mlp->layers[i]; j++){
                mlp->result[i][j] = 0;
            }
        }

        for(int i = 0; i < mlp->layers[0]; i++){
            mlp->result[0][i] = inputs[i];
        }


        for(int l = 1; l < mlp->layersSize - 1; l++){
            for(int j = 0; j < mlp->layers[l] ; j++){
                mlp->result[l][j] = l == 1 ? mlp->model[l][0][j] : tanh(mlp->result[l-1][j] * mlp->model[l][0][j]);
                for(int i = 0; i < mlp->layers[l + 1]; i++){
                    mlp->result[l][j] = l == 1 ? mlp->result[l-1][j] * mlp->model[l][i + 1][j] : tanh(mlp->result[l-1][j] * mlp->model[l][i + 1][j]);
                }
            }
        }
        return mlp->result[mlp->layersSize - 1][0] > 0 ? 1 : -1;

    }

    DLLEXPORT void train_mlp_class_model(MLP* mlp, double* dataset, double* expeced_outputs, int datasetSize, int step, int epoch) {
        int size = datasetSize / mlp->layers[0];
        for (int i = 0; i < epoch; i++) {
            int exemple_number = rand() % size;
            int predicted = predict_mlp_class_model(mlp, dataset + exemple_number * mlp->layers[0]);
            int expected = expeced_outputs[exemple_number];

            mlp->delta[mlp->layersSize-1][0] = (1 - pow(predicted, 2)) * (predicted - expected);

            for(int l = mlp->layersSize - 1; l > 0; l--){
                for(int k = mlp->layers[l]; k >= 0; k--){
                    for(int j = 0; j < mlp->layers[l - 1]; j++){
                        mlp->delta[l][k] += 1;
                    }
                    mlp->delta[l][k] *= (1 - pow(mlp->result[l][k], 2));
                }
            }
        }
    }
}

int main()
{
    auto dataset = new double[6] {-10.0, 0.0, 1.0, 0.0, 5.0, 1.0};
    auto expectedOutputs = new double[3] {1.0, 0.0, 1.0};

    /*cout << "bonjour";
    auto m = create_linear_model(2);
    train_linear_class_model(m, dataset, expectedOutputs, 2, 6, 0.0001 , 100000);

    cout << "bonjour";
    cout << predict_linear_class_model(m, dataset, 2) << endl;
    cout << predict_linear_class_model(m, dataset + 2, 2) << endl;
    cout << predict_linear_class_model(m, dataset + 4, 2) << endl;

    for(int i = 0; i < 3; i++){
        cout << m[i] << endl;
    }*/

    MLP* mlp = create_mlp_model(new int[5] {2, 3, 2, 2, 1}, 5);
    cout << mlp->model[1][1][0] << endl;
    cout << predict_mlp_class_model(mlp, new double[2] {2, 3}) << endl;
    std::cout << "feoizjfozeijfoizej" << endl;
    return 0;
}