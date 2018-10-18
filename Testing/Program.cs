using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
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

            var testImage = new Bitmap(Bitmap.FromFile(Path.Join(SampleFolderPath, "testImage.png")));

            var result = Upsample.Test(testImage, device);

            result.Save(Path.Join(SampleFolderPath, "upsample output.bmp")); // check image manually
        }
    }
}
