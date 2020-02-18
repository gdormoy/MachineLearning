using System;
using System.Runtime.InteropServices;
using UnityEngine;
using Random = UnityEngine.Random;

public class RetrieveAndModifySpherePositionsScript : MonoBehaviour
{

    [DllImport("machine_learning_lib")]
    private static extern IntPtr create_linear_model(int numberOfParams);

    [DllImport("machine_learning_lib")]
    private static extern IntPtr train_linear_model(IntPtr model, double[] dataset, double[] expected_output, int numberOfParams,  int datasetSize);

    [DllImport("machine_learning_lib")]
    private static extern double predict_linear_model(IntPtr model, double[] param, int numberOfParams);

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
        
        train_linear_model(model, trainingInputs, trainingExpectedOutputs, 2, trainingSpheres.Length);
    }

    public void PredictOnTestSpheres()
    {
        for (var i = 0; i < testSpheres.Length; i++)
        {
            var input = new double[] {testSpheres[i].position.x, testSpheres[i].position.z};
            var predictedY = (float) predict_linear_model(model, input, 2);
            // var predictedY = Random.Range(-5, 5);
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