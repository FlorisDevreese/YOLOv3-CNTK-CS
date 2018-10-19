using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
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
        public static Bitmap Test(Bitmap inputImage, DeviceDescriptor device)
        {
            var input = CNTKLib.InputVariable(new[] { 320, 320, 3 }, DataType.Float, "0: input");

            // upsampling approach 1 (from Frank), based on https://stackoverflow.com/questions/43079648/cntk-how-to-define-upsampling2d
            // this doesn't work. It enlarges the dimensions, but hoesn't upsample the image content. It just shows 4 times the original picture
            //var xr = CNTKLib.Reshape(input, new[] { 320, 1, 320, 1, 3 });
            //var xx = CNTKLib.Splice(new VariableVector(new Variable[] { xr, xr }), new Axis(3));
            //var xy = CNTKLib.Splice(new VariableVector(new Variable[] { xx, xx }), new Axis(1));
            //var model = CNTKLib.Reshape(xy, new[] { 320 * 2, 320 * 2, 3 });

            // upsampling approach 2 (from David), based on https://stackoverflow.com/questions/43079648/cntk-how-to-define-upsampling2d
            // this doesn't work eather. I can't figure out what the parameters of CNTKLib.ConcolutionTranspose(...) should be. More info here: https://github.com/Microsoft/CNTK/issues/2939
            //var convolutionMap = new Constant(new[] { 3, 3, 3, 3 }, DataType.Float, 1f);
            //var model = CNTKLib.ConvolutionTranspose(convolutionMap, input, new[] { 2, 2, 1 }, new BoolVector() { false }, new BoolVector() { true, true, false });

            // Upsampling approach 3. This is just simple max unpooling. This works, but doesn't return the desired output because only one out of every 4 pixels isn't black
            var poolingLayer = CNTKLib.Pooling(input, PoolingType.Max, new[] { 3, 3 }, new[] { 2, 2 }, new BoolVector() { true });
            var model = CNTKLib.Unpooling(poolingLayer, input, PoolingType.Max, new[] { 2, 2 }, new[] { 2, 2 });

            // create the input for the model
            var modelInput = model.Arguments.First();
            var modelOutput = model.Output;

            var inputMap = Helper.CreateInputDataMap(inputImage, input, device);
            var outputDataMap = new Dictionary<Variable, Value>() { { modelOutput, null } };

            // Run the model
            model.Evaluate(inputMap, outputDataMap, device);

            // get the output of the model
            var output = outputDataMap[modelOutput];
            var outputArray = output.GetDenseData<float>(modelOutput)[0];
            var outputWidth = output.Shape.Dimensions[0];
            var outputHeight = output.Shape.Dimensions[1];

            // Convert the output of the model into an image
            return Helper.CreateBitmapFromOutput(outputArray, outputWidth, outputHeight);
        }
    }
}
