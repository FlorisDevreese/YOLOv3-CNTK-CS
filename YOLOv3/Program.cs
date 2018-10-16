using CNTK;
using System;
using System.IO;

namespace YOLOv3
{
    class Program
    {
        static void Main(string[] args)
        {
            // Explenation of YOLO v3 here: https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b
            // Description of some implementation issues: https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe

            // get device to run YOLO on
            var device = DeviceDescriptor.GPUDevice(0);
            Console.WriteLine($"======== running YOLO on {device.Type} ========");
            
            // get the network config
            string configFilePath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.Parent.FullName, "config/yolov3.cfg");
            var blocks = Darknet.ParseConfigFile(configFilePath);

            // create network
            var network = Darknet.CreateNetwork(blocks, out Variable input, device);

            // use the blocks to construct cntk modules for the blocks present in the config file
                       
            Console.WriteLine("\nEnd of program -> Press any key to close the program");
            Console.ReadKey();
        }
    }
}
