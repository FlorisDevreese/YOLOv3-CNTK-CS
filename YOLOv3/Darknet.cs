using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Linq;
using CNTK;

namespace YOLOv3
{
    public class Darknet
    {
        //Hyper parameters
        private const double bnTimeConst = 4096;
        private const double leakReLuAplha = 0.1;

        /// <summary>
        /// Converts a config file into a list of Dictionaries (blocks) which describe the network to be built.
        /// </summary>
        public static List<Dictionary<string, string>> ParseConfigFile(string configFilePath)
        {
            var allBlocks = new List<Dictionary<string, string>>();
            var currentBlock = new Dictionary<string, string>();

            using (StreamReader configFile = File.OpenText(configFilePath))
            {
                string line;
                while ((line = configFile.ReadLine()) != null)
                {
                    line = line.Trim();

                    // check if this line is the start of a new block
                    Match newBlockRegex = Regex.Match(line, @"^\[(.+)\]");
                    if (newBlockRegex.Success)
                    {
                        // create new block, and add it to the allBlocks list
                        currentBlock = new Dictionary<string, string>
                        {
                            ["type"] = newBlockRegex.Groups[1].Value
                        };
                        allBlocks.Add(currentBlock);
                    }

                    // check if this line is just a key value pair, and not an empty line or comment line
                    Match keyValueRegex = Regex.Match(line, @"^([^#].+)=(.+)");
                    if (keyValueRegex.Success)
                    {
                        // add key value pair to the current block
                        string key = keyValueRegex.Groups[1].Value.Trim();
                        string value = keyValueRegex.Groups[2].Value.Trim();
                        currentBlock[key] = value;
                    }
                }
            }

            return allBlocks;
        }

        /// <summary>
        /// Creates CNTK modules for each block in the allBlocks list
        /// </summary>
        /// <param name="allBlocks"></param>
        public static Function CreateNetwork(List<Dictionary<string, string>> allBlocks, out Variable input, DeviceDescriptor device)
        {
            // a list of all network layers so that
            var networkLayers = new List<Function>();

            // the first block is the 'net' block with information about the input and the pre-processing
            var netBlock = allBlocks.FirstOrDefault();
            int batch = GetParameterValue<int>(netBlock, "batch", "net");
            int subdivisions = GetParameterValue<int>(netBlock, "subdivisions", "net");
            int width = GetParameterValue<int>(netBlock, "width", "net");
            int height = GetParameterValue<int>(netBlock, "height", "net");
            int channels = GetParameterValue<int>(netBlock, "channels", "net");
            int burnIn = GetParameterValue<int>(netBlock, "burn_in", "net");
            int maxBatches = GetParameterValue<int>(netBlock, "max_batches", "net");
            double momentum = GetParameterValue<double>(netBlock, "momentum", "net");
            double decay = GetParameterValue<double>(netBlock, "decay", "net");
            double angle = GetParameterValue<double>(netBlock, "angle", "net");
            double saturation = GetParameterValue<double>(netBlock, "saturation", "net");
            double exposure = GetParameterValue<double>(netBlock, "exposure", "net");
            double hue = GetParameterValue<double>(netBlock, "hue", "net");
            double learningRate = GetParameterValue<double>(netBlock, "learning_rate", "net");
            string policy = GetParameterValue<string>(netBlock, "policy", "net");
            string steps = GetParameterValue<string>(netBlock, "steps", "net");
            string scales = GetParameterValue<string>(netBlock, "scales", "net");

            // create the input variable
            input = CNTKLib.InputVariable(new [] { width, height, 3 }, DataType.Float, "0: input");
            // todo is line below necessary?
            //var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
            networkLayers.Add(input);

            // the other blocks are actural layers in the network
            for (int i = 1; i < allBlocks.Count; i++)
            {
                var block = allBlocks[i];
                var priorLayer = (Variable)networkLayers.LastOrDefault();
                int priorFilers = priorLayer.Shape.Dimensions[2];

                if (!block.TryGetValue("type", out string type))
                    throw new Exception("block contains no 'type' value");

                if ("convolutional".Equals(type))
                {
                    // WARNING: this implmentation isn't tested yet. It could also differ from the documentation concerning the bias param.

                    // get all parameters of the block
                    bool batchNormalize = GetParameterValue<bool>(block, "batch_normalize", type);
                    // the 'pad' config is ignored because it's always 1 (=true)
                    int newFilters = GetParameterValue<int>(block, "filters", type);
                    int size = GetParameterValue<int>(block, "size", type);
                    int stride = GetParameterValue<int>(block, "stride", type);
                    string activation = GetParameterValue<string>(block, "activation", type);

                    // add convolution
                    var convParameter = new Parameter(new [] { size, size, priorFilers, newFilters }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device); // todo does this need a initializer?
                    var newLayer = CNTKLib.Convolution(convParameter, priorLayer, new [] { stride, stride, priorFilers });
                    newLayer.SetName(i + ": Convolution");

                    // add batch normalization
                    if (batchNormalize)
                    {
                        var biasParams = new Parameter(new[] { NDShape.InferredDimension }, 0.0f, device, "");
                        var scaleParams = new Parameter(new[] { NDShape.InferredDimension }, 0.0f, device, "");
                        var runningMean = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device);
                        var runningInvStd = new Constant(new[] { NDShape.InferredDimension }, 0.0f, device);
                        var runningCount = Constant.Scalar(0.0f, device);
                        newLayer = CNTKLib.BatchNormalization(newLayer, scaleParams, biasParams, runningMean, runningInvStd, runningCount, true, bnTimeConst, 0.0, 1e-5 /* epsilon */);
                        newLayer.SetName(i + ": Batch normaization");
                    }

                    // add activation
                    if ("leaky".Equals(activation))
                    {
                        newLayer = CNTKLib.LeakyReLU(newLayer, leakReLuAplha, i + ": Activation");
                    }

                    networkLayers.Add(newLayer);
                }
                else if ("upsample".Equals(type))
                {
                    // get all parameters of the block
                    int stride = GetParameterValue<int>(block, "stride", type);
                    int priorWidht = priorLayer.Shape.Dimensions[1];
                    int priorheight = priorLayer.Shape.Dimensions[1];

                    // A correct implementation of bilinear upsampling in CNTK isn't found yet.
                    // I suppose bilinear upsampling doesn't exist in CNTK. See: https://github.com/Microsoft/CNTK/issues/3045

                    // todo continue work here: Found a way to do bilinear upsampling

                }
                else if ("shortcut".Equals(type))
                {
                    // get all parameters of the block
                    int from = GetParameterValue<int>(block, "from", type);
                    string activation = GetParameterValue<string>(block, "activation", type);

                    // todo build the layer

                }
                else if ("route".Equals(type))
                {
                    // get all parameters of the block
                    string layers = GetParameterValue<string>(block, "layers", type);

                    // todo build the layer

                }
                else if ("yolo".Equals(type))
                {
                    // get all parameters of the block
                    bool random = GetParameterValue<bool>(block, "random", type);
                    int classes = GetParameterValue<int>(block, "classes", type);
                    int num = GetParameterValue<int>(block, "num", type);
                    double jitter = GetParameterValue<double>(block, "jitter", type);
                    double ignoreThreshold = GetParameterValue<double>(block, "ignore_thresh", type);
                    double truthThreshold = GetParameterValue<double>(block, "truth_thresh", type);
                    string mask = GetParameterValue<string>(block, "mask", type);
                    string anchors = GetParameterValue<string>(block, "anchors", type);

                    // todo build the layer

                }
            }

            //todo print out the builded network

            return networkLayers.LastOrDefault();
        }

        /// <summary>
        /// Gets the value associated to the key in the dictonary, and converts it to type T.
        /// Will throw an Exception when the conversion failed.
        /// </summary>
        /// <typeparam name="T">Type to which you want to convert</typeparam>
        /// <param name="block">Dictonary where to look up te key</param>
        /// <param name="key"></param>
        /// <param name="blockType">Type of the block</param>
        /// <returns></returns>
        private static T GetParameterValue<T>(Dictionary<string, string> block, string key, string blockType)
        {
            Type conversionType = typeof(T);

            block.TryGetValue(key, out string value);

            if (conversionType.Equals(typeof(bool)))
                return (T)(object)(value != null && "1".Equals(value));

            if (value == null)
                throw new Exception("'" + blockType + "' block contains no '" + key + "' value");

            if (conversionType.Equals(typeof(string)))
                return (T)(object)value;
            if (conversionType.Equals(typeof(int)))
            {
                if (!int.TryParse(value, out int returnValue))
                    throw new Exception("'" + blockType + "' block contains no integer '" + key + "' value");

                return (T)(object)returnValue;
            }
            if (conversionType.Equals(typeof(double)))
            {
                if (!double.TryParse(value, out double returnValue))
                    throw new Exception("'" + blockType + "' block contains no double '" + key + "' value");

                return (T)(object)returnValue;
            }

            throw new Exception("A converstion to type " + conversionType + " is not supported");
        }
    }
}
