using System.Runtime.InteropServices;
using UnityEngine;

public class TestDllsScript : MonoBehaviour
{
    [DllImport("2020_5A_AL1_CppDllForUnity")]
    private static extern int GiveMe42FromC();
    
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log($"MyCDll : {GiveMe42FromC()}");
    }
}