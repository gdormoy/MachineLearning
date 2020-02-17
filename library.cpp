#include "library.h"
#include <stdlib.h>
#include <time.h>

#if linux
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"{
    DLLEXPORT double* create_linear_model(int numberOfParams) {
        srand(time(NULL));
        double model [numberOfParams + 1];
        for(int i = 0; i < numberOfParams + 1; i++){
            model[i] = rand() % 1;
        }
        return model;
    }

    DLLEXPORT double* train_linear_model(double* model, double* dataset) {

        return model;
    }
}