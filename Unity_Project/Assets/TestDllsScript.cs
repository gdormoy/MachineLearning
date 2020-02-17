using System.Runtime.InteropServices;
using UnityEngine;

public class TestDllsScript : MonoBehaviour
{
    [DllImport("2020_5A_AL1_CppDllForUnity")]
    private static extern int GiveMe42FromC();
    
    [DllImport("_2020_5A_AL1_MyRustDllForUnity")]
    private static extern int GiveMe42FromRust();
    
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log($"MyCDll : {GiveMe42FromC()}");
        Debug.Log($"MyRustDll : {GiveMe42FromRust()}");
    }
}