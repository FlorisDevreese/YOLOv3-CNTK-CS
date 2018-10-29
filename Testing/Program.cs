﻿using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Testing.Tests;

namespace Testing
{
    class Program
    {
        public static string RootPath = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.Parent.Parent.FullName;

        /// <summary>
        /// This is a project to quickly test out functions
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.GPUDevice(0);

            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\cicles.png")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\jump.png")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\1.jpg")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\3.jpg")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\5.jpg")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\6.png")));
            //var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\7.jpg")));
            var testImage = new Bitmap(Image.FromFile(Path.Join(RootPath, @"Test images\dog-cycle-car.png")));

            //// Upsample
            //var result = Upsample.Test(testImage, device);
            //result.Save(Path.Join(RootPath, @"Output\upsample output.bmp"));

            // ONNX models
            var result = OnnxModelModels.TestTinyYoloV2(testImage, device);
            result.Save(Path.Join(RootPath, @"Output\Tiny Yolov2 output.bmp"));

            //OnnxModelModels.RetrainTinyYoloV2(device);

            // Testing Simple networks
            //SimpleNetworks.LogisticRegression(device);
        }
    }
}
