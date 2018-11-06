using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Testing.Utils;

namespace Testing.Tests
{
    public class OpertationTester
    {
        public static void TestElementTimes()
        {
            var device = DeviceDescriptor.GPUDevice(0);
            // todo put in different values
            CNTKDictionary testInitializer = new CNTKDictionary();

            Parameter leftOperand = new Parameter(new int[] { 2, 2 }, 1f, device, "left");
            NDArrayView initValues = new NDArrayView(new int[] { 2, 2 }, new float[] { 0f, 1f, 2f, 3f }, device);
            leftOperand.SetValue(initValues);
            // leftOperand looks like:
            // 0  1
            // 4  9

            Parameter rightOperand = new Parameter(new int[] { 2, 2, 2 }, 1f, device, "right");
            initValues = new NDArrayView(new int[] { 2, 2, 2 }, new float[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f}, device);
            rightOperand.SetValue(initValues);
            // rightOperand looks like:
            // 0  1 |  4  5
            // 2  3 |  6  7

            Function model = CNTKLib.ElementTimes(leftOperand, rightOperand);

            var inputVariable = model.Inputs.First();
            var inputMap = new Dictionary<Variable, Value>();

            var outputVariable = model.Output;
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVariable, null } };

            model.Evaluate(inputMap, outputDataMap, device);

            var output = outputDataMap[outputVariable];
            var outputArray = output.GetDenseData<float>(outputVariable).First();
            // output looks like:
            // 0  1 |  0  5
            // 4  9 | 12 21

            // conclusion of this test: CNTKLib.ElementTimes works as espected :-)
        }
    }
}
