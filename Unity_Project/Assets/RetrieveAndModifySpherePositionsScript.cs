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
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.0001, 100000);
        Debug.Log("model is training");
    }
    public void TrainXOR()
    {
        trainingInputs = new double[trainingSpheres.Length * 2];
        trainingExpectedOutputs = new double[trainingSpheres.Length];

        for (var i = 0; i < trainingSpheres.Length; i++)
        {
            trainingInputs[2 * i] = Math.Sign(trainingSpheres[i].position.x);
            trainingInputs[2 * i + 1] = Math.Sign(trainingSpheres[i].position.z);
            trainingExpectedOutputs[i] = trainingSpheres[i].position.y;
        }
        
        train_linear_class_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingInputs.Length, 0.0001, 100000);
        Debug.Log("model is training");
        for (var i = 0; i < trainingSpheres.Length; i++){
            trainingSpheres[i].position = new Vector3(
                (float) Math.Pow(trainingSpheres[i].position.x,2),
                (float) Math.Pow(trainingSpheres[i].position.y,2),
                (float) Math.Pow(trainingSpheres[i].position.z,2)
            );
        }
         
    }

    public void PredictOnTestSpheresXOR()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {Math.Sign(testSpheres[i].position.x), Math.Sign(testSpheres[i].position.z)};
            Debug.Log($"input length: {input.Length}");
            var predictedY = (float) predict_linear_class_model(model, input, 2);
            Debug.Log($"predict: {predictedY}");
            testSpheres[i].position = new Vector3(
                testSpheres[i].position.x,
                predictedY,
                testSpheres[i].position.z);
        }
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

    public int XORFunc(double x, double y){
        int res = (byte) x ^ (byte) y;
        Debug.Log($"xor = {res}");
        return res;
    }
}