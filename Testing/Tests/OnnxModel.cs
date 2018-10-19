using CNTK;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Testing.Utils;

namespace Testing.Tests
{
    public class OnnxModel
    {
        public static void TestModel(string modelPath, Bitmap image, DeviceDescriptor device)
        {
            Function model = Function.Load(modelPath, device, ModelFormat.ONNX);

            var inputVariable = model.Arguments.First();
            var modelOutput = model.Output;

            var inputMap = Helper.CreateInputDataMap(image, inputVariable, device);
            var outputDataMap = new Dictionary<Variable, Value>() { { modelOutput, null } };

            model.Evaluate(inputMap, outputDataMap, device);

            var output = outputDataMap[modelOutput];
            var outputArray = output.GetDenseData<float>(modelOutput).First();

            // todo parse the output
        }
    }
}
