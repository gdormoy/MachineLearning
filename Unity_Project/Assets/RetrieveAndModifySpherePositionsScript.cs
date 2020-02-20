using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class RetrieveAndModifySpherePositionsScript : MonoBehaviour
{

    [DllImport("machine_learning_lib")]
    private static extern IntPtr create_linear_model(int numberOfParams);

    [DllImport("machine_learning_lib")]
    private static extern void train_linear_model(IntPtr model, double[] dataset, double[] expected_output, int numberOfParams,  int datasetSize);

    [DllImport("machine_learning_lib")]
    private static extern double predict_linear_model(IntPtr model, double[] param, int numberOfParams);

    [DllImport("machine_learning_lib")]
    private static extern double predict_linear_class_model(IntPtr model, double[] param, int numberOfParams);

    [DllImport("machine_learning_lib")]
    private static extern double train_linear_class_model(IntPtr model, double[] dataset, double[] expected_output,
        int numberOfParams,  int datasetSize, double step, int epoch);

    public Transform[] trainingSpheres;

    public Transform[] testSpheres;

    private double[] trainingInputs;

    private double[] trainingExpectedOutputs;

    private IntPtr model;

    public int epoch = 1000;

    public void ReInitialize()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                0f,
                testSpheres[i].position.z);
        }
    }
    
    public void CreateModel()
    {
        model = create_linear_model(2);
        Debug.Log(model);
    }

    public void Train()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.00001, epoch);
        Debug.Log("model is training");
    }

    public void PredictOnTestSpheres()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            Debug.Log($"x: {input[0]}");
            Debug.Log($"z: {input[1]}");
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_class_model(model, input, 2);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
    }

    public void TrainKOSoft()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.00001, epoch);
        Debug.Log("model is training");
    }

    public void PredictOnTestSpheresKOSoft()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            Debug.Log($"x: {input[0]}");
            Debug.Log($"z: {input[1]}");
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_class_model(model, input, 2);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
    }

    public void TrainXOR()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length/2; i++)
        {
            trainingInputs[2 * i] = (trainingSpheres[i].position.x * trainingSpheres[i].position.z) * -1;
            // trainingInputs[2 * i + 1] = Math.Pow(trainingSpheres[i].position.z,2);
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 1, trainingInputs.Length, 0.000001, epoch);
        Debug.Log("model is training");
         
    }

    public void PredictOnTestSpheresXOR()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {(testSpheres[i].position.x * testSpheres[i].position.z) * -1};
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_class_model(model, input, 1);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
    }

    public void TrainCross()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = 1 / (Math.Pow(trainingSpheres[i].position.x,2) * Math.Pow(trainingSpheres[i].position.z,2));
            trainingInputs[2 * i + 1] = 1 / (Math.Pow(trainingSpheres[i].position.z,2) * Math.Pow(trainingSpheres[i].position.x,2));
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.00001, epoch);
        Debug.Log("model is training");
         
    }

    public void PredictOnTestSpheresCross()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {1 / (Math.Pow(testSpheres[i].position.x,2) * Math.Pow(testSpheres[i].position.z,2)),
                                      1 / (Math.Pow(testSpheres[i].position.z,2) * Math.Pow(testSpheres[i].position.x,2))};
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_class_model(model, input, 2);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
    }

        public void TrainCrossIF()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            double x = Math.Abs(trainingSpheres[i].position.x);
            double z = Math.Abs(trainingSpheres[i].position.z);
            if(x < 6 && x > 2 && z > 2 && z < 8) {
                trainingInputs[2 * i] = x;
                trainingInputs[2 * i + 1] = z;
            } else {
                trainingInputs[2 * i] = trainingSpheres[i].position.x;
                trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
            }
            // trainingInputs[2 * i] = 1 / (Math.Pow(trainingSpheres[i].position.x,2) * Math.Pow(trainingSpheres[i].position.z,2));
            // trainingInputs[2 * i + 1] = 1 / (Math.Pow(trainingSpheres[i].position.z,2) * Math.Pow(trainingSpheres[i].position.x,2));
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.0001, epoch);
        Debug.Log("model is training");
         
    }

    public void PredictOnTestSpheresCrossIF()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            
            double x = Math.Abs(testSpheres[i].position.x);
            double z = Math.Abs(testSpheres[i].position.z);
            // var input = new double[] {x, z};
            if(x < 6 && x > 2 && z > 2 && z < 8) {
                var input = new double[] {x, z};
                Debug.Log($"input length: {input.Length}");
                var predictedY = (float) predict_linear_class_model(model, input, 2);
                Debug.Log($"predict: {predictedY}");
                testSpheres[i].position = new Vector3(
                    (float) x,
                    predictedY,
                    (float) z);
            } else {
                var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
                Debug.Log($"input length: {input.Length}");
                var predictedY = (float) predict_linear_class_model(model, input, 2);
                Debug.Log($"predict: {predictedY}");
                testSpheres[i].position = new Vector3(
                    testSpheres[i].position.x,
                    predictedY,
                    testSpheres[i].position.z);
            }
            
        }
    }

    public void TrainRegression()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = trainingSpheres[i].position.x;
            trainingInputs[2 * i + 1] = trainingSpheres[i].position.z;
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length);
        Debug.Log("model is training");
    }

    public void PredictOnTestSpheresRegression()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            Debug.Log($"x: {input[0]}");
            Debug.Log($"z: {input[1]}");
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_model(model, input, 2);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
    }

    public void ReleaseModel()
    {
        // FreeLinearModel(model);
    }
}