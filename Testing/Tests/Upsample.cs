using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using Testing.Utils;

namespace Testing.Tests
{
    public class Upsample
    {
        /// <summary>
        /// Tryout for different upsample functions
        /// </summary>
        /// <param name="device"></param>
        public static void Test(string sampleFolderPath, DeviceDescriptor device)
        {
            int width = 320;
            int height = 320;
            int channels = 3;

            var input = CNTKLib.InputVariable(new[] { width, height, channels }, DataType.Float, "0: input");

            // upsampling approach 1 (from Frank), based on https://stackoverflow.com/questions/43079648/cntk-how-to-define-upsampling2d
            // this doesn't work. It enlarges the dimensions, but hoesn't upsample the image content. It just shows 4 times the original picture
            //var xr = CNTKLib.Reshape(input, new[] { width, 1, height, 1, channels });
            //var xx = CNTKLib.Splice(new VariableVector(new Variable[] { xr, xr }), new Axis(3));
            //var xy = CNTKLib.Splice(new VariableVector(new Variable[] { xx, xx }), new Axis(1));
            //var model = CNTKLib.Reshape(xy, new[] { width * 2, height * 2, channels });

            // upsampling approach 2 (from David), based on https://stackoverflow.com/questions/43079648/cntk-how-to-define-upsampling2d
            // this doesn't work eather. I can't figure out what the parameters of CNTKLib.ConcolutionTranspose(...) should be. More info here: https://github.com/Microsoft/CNTK/issues/2939
            //var convolutionMap = new Constant(new[] { 3, 3, 3, 3 }, DataType.Float, 1f);
            //var model = CNTKLib.ConvolutionTranspose(convolutionMap, input, new[] { 2, 2, 1 }, new BoolVector() { false }, new BoolVector() { true, true, false });

            // Upsampling approach 3. This is just simple max unpooling. This works, but doesn't return the desired output because only one out of every 4 pixels isn't black
            var poolingLayer = CNTKLib.Pooling(input, PoolingType.Max, new[] { 3, 3 }, new[] { 2, 2 }, new BoolVector() { true });
            var model = CNTKLib.Unpooling(poolingLayer, input, PoolingType.Max, new[] { 2, 2 }, new[] { 2, 2 });

            // create the input for the model
            var modelInput = model.Arguments[0];
            var modelOutput = model.Output;

            var validationMap = Path.Combine(sampleFolderPath, "val_map.txt");
            var deserializerConfiguration = CNTKLib.ImageDeserializer(validationMap, "labels", (uint)2, "image");

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration });
            MinibatchSource minibatchSource = CNTKLib.CreateCompositeMinibatchSource(config);

            var featureStreamInfo = minibatchSource.StreamInfo("image");
            var minibatchData = minibatchSource.GetNextMinibatch(1, device);
            var inputDataMap = new Dictionary<Variable, Value>() { { modelInput, minibatchData[featureStreamInfo].data } };
            var outputDataMap = new Dictionary<Variable, Value>() { { modelOutput, null } };

            // Run the model
            model.Evaluate(inputDataMap, outputDataMap, device);

            // get the output of the model
            var output = outputDataMap[modelOutput];
            var outputArray = output.GetDenseData<float>(modelOutput)[0];
            var outputWidth = output.Shape.Dimensions[0];
            var outputHeight = output.Shape.Dimensions[1];

            // Convert the output of the model into an image
            var outputImage = Helper.CreateBitmapFromOutput(outputArray, outputWidth, outputHeight);

            // write the image to file (so it can be manually checked)
            outputImage.Save(Path.Join(sampleFolderPath, "upsample output.bmp"));
        }
    }
}
