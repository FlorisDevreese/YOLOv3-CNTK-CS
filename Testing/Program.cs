using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using Testing.Tests;

namespace Testing
{
    class Program
    {
        public static string SampleFolderPath = Path.Combine(Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.Parent.FullName, "SampleImages");

        /// <summary>
        /// This is a project to quickly test out functions
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.GPUDevice(0);

            Upsample.Test(SampleFolderPath, device);
        }
    }
}
