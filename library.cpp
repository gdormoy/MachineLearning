#include "library.h"
#if linux
#define DLLEXPORT
#else
#define DLLEXPORT __declspec(dllexport)
#endif

extern "C"{
    DLLEXPORT int GiveMe42FromC() {
        return 42;
    }
}