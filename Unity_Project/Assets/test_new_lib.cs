using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class test_new_lib : MonoBehaviour
{
    [DllImport("machine_learning_lib")]
    private static extern double create_linear_model();
    
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log($"MyCDll2 : {create_linear_model()}");
    }
}
