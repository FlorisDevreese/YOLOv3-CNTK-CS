using CNTK;
using System;
using System.IO;

namespace YOLOv3
{
    class Program
    {
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.GPUDevice(0);
            // Understand the inner workings of YOLO v3 by reading this: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b

            // get the building blocks
            string configFilePath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName, "config/yolov3.cfg");
            var blocks = Darknet.ParseConfigFile(configFilePath);

            // use the blocks to construct cntk modules for the blocks present in the config file

            // Testing if the CNTK dependecy works
            Variable features = new Variable();
            double convWScale = 0.26;
            var convParams = new Parameter(new int[] { 3, 3, 3, 64 }, DataType.Float, CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), device);
            Function convFunction = CNTKLib.ReLU(CNTKLib.Convolution(convParams, features, new int[] { 1, 1, 3 } /* strides */));

            Console.WriteLine("\nEnd of program -> Press any key to close the program");
            Console.ReadKey();
        }
    }
}
