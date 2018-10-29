using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Testing.Models;
using Testing.Utils;

namespace Testing.Tests
{
    public class OnnxModelModels
    {
        public static string TinyYoloV2ModelPath = Path.Join(Program.RootPath, @"ONNX model test app\Assets\Models\Tiny-YOLOv2.onnx");

        /// <summary>
        /// Runs existing tiny YOLO v2 ONNX model on image
        /// </summary>
        /// <param name="image">Image on which to run the model on</param>
        /// <param name="device">Device to rin the model on</param>
        /// <returns>Original image annotated with the results of the model</returns>
        public static Bitmap TestTinyYoloV2(Bitmap image, DeviceDescriptor device)
        {
            // the labels and anchor box coordinates are dependent on how the model is learned.
            string[] labels = new string[] { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
            float[,] anchorCoordintes = { { 1.08f, 1.19f }, { 3.42f, 4.41f }, { 6.63f, 11.38f }, { 9.42f, 5.11f }, { 16.62f, 10.52f } }; // pairs of height and weight for every anchor box

            Function model = Function.Load(TinyYoloV2ModelPath, device, ModelFormat.ONNX);
            // Add a layer that converts yolo output
            model = YoloDetectionLayer(model, anchorCoordintes, labels.Length, device);

            var inputVariable = model.Arguments.First();
            var outputVariable = model.Output;

            var inputMap = Helper.CreateInputDataMap(image, inputVariable, device);
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVariable, null } };

            model.Evaluate(inputMap, outputDataMap, device);

            var output = outputDataMap[outputVariable];
            var outputArray = output.GetDenseData<float>(outputVariable).First();

            return VisualizeOutput(image, outputArray, labels, anchorCoordintes, outputVariable.Shape[0], outputVariable.Shape[1], 0.3f);
        }

        /// <summary>
        /// Converts YOLO output to more meaningfull values.
        /// Values like "tx, ty, tw, th, tc, an the class scores" will be converted more "real" values. This means:
        ///  - Probability values will be softmaxed
        ///  - tx, ty, tw, ... values will be converted to cell relative values. This means when the output value for x is 0.5 , the x coordinate is in the middel of that cell
        /// </summary>
        /// <param name="input"></param>
        /// <returns>Function that converts YOLO output to more meaningfull values</returns>
        private static Function YoloDetectionLayer(Function input, float[,] anchorCoordintes, int classes, DeviceDescriptor device)
        {
            VariableVector convertedVariables = new VariableVector();

            int anchors = anchorCoordintes.GetLength(0);
            for (int i = 0; i < anchors; i++)
            {
                int anchorOffset = (5 + classes) * i;

                // sigmoid the first 2 channels of this anchor. These channels contain tx and ty
                Function centerSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 }, $"Slice tx, ty for anchor box {i}");
                Function centerSigmoid = CNTKLib.Sigmoid(centerSlice, $"Sigmoid tx, ty for anchor box {i}");
                convertedVariables.Add(centerSigmoid);

                // convert next channel: e^(tw)*anchorCoordinates
                Function widthSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 3 }, $"Slice tw for anchor box {i}");
                Function widthExponentials = CNTKLib.Exp(widthSlice, $"Exponentiate tw for anchor box {i}");
                CNTKDictionary anchorWidth = CNTKLib.ConstantInitializer(anchorCoordintes[i, 0]);
                Parameter anchorWidthParameter = new Parameter(widthSlice.Output.Shape.SubShape(0, 3), DataType.Float, anchorWidth, device, $"anchor width for anchor box {i}");
                Function convertedWidth = CNTKLib.ElementTimes(widthExponentials, anchorWidthParameter, $"Multiply tw times the anchor box width for anchor box {i}");
                convertedVariables.Add(convertedWidth);

                // convert next channel: e^(th)*anchorCoordinates
                Function heightSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 3 }, new IntVector() { anchorOffset + 4 }, $"Slice th for anchor box {i}");
                Function heigthExponentials = CNTKLib.Exp(heightSlice, $"Exponentiate th for anchor box {i}");
                CNTKDictionary anchoHeight = CNTKLib.ConstantInitializer(anchorCoordintes[i, 1]);
                Parameter anchorHeightParameter = new Parameter(heightSlice.Output.Shape.SubShape(0, 3), DataType.Float, anchoHeight, device, $"anchor height for anchor box {i}");
                Function convertedHeight = CNTKLib.ElementTimes(heigthExponentials, anchorHeightParameter, $"Multiply th times the anchor box width for anchor box {i}");
                convertedVariables.Add(convertedHeight);

                // sigmoid the next channel. This channel contains tc
                Function objectnessSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 }, $"Slice tx for anchor box {i}");
                Function objectnessSigmoid = CNTKLib.Sigmoid(objectnessSlice, $"Sigmoid tc for anchor box {i}");
                convertedVariables.Add(objectnessSigmoid);

                // Softmax the remaining layers of this anchor. These layers contain the c1, c2, ... values
                Function classesSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes }, $"Slice all classes for anchor box {i}");
                Function classSoftMax = CNTKLib.Softmax(classesSlice, new Axis(2), $"Softmax classes values for anchor box {i}");
                convertedVariables.Add(classSoftMax);
            }

            // concatenate the converted layers back in the right order
            return CNTKLib.Splice(convertedVariables, new Axis(2), "Splice all converted layers back together");
        }

        /// <summary>
        /// Processes the output of the network on to the image. 
        /// This function assumes that the tx, ty, tw, th, tc, c0, C1, ... values of the network are already converted to relative values
        /// </summary>
        /// <param name="image">original image on which the result is drawn</param>
        /// <param name="outputData">output data of the network</param>
        /// <param name="labels">string array with all the names of the classes</param>
        /// <param name="gridWidth">Number of cells the width of the image is devided in</param>
        /// <param name="gridHeight">Number of cells the height of the image is devided in</param>
        /// <param name="anchors">number of bounding boxes per cell</param>
        /// <param name="threshold">certainty threshold for the objectness and classness of a bounding box</param>
        /// <returns>annotated image</returns>
        private static Bitmap VisualizeOutput(Bitmap image, IList<float> outputData, string[] labels, float[,] anchorCoordintes, int gridWidth, int gridHeight, float threshold)
        {
            int bbLength = (5 + labels.Length);
            int cellWidth = image.Width / gridWidth;
            int cellHeight = image.Height / gridHeight;

            int anchors = anchorCoordintes.GetLength(0); // get the length of the first dimension

            // validate input
            if (outputData.Count != gridWidth * gridHeight * anchors * bbLength)
                throw new ArgumentException("The count of outputData.Count should be equal to: gridWidth * gridHeight * anchors * (5 + labels.Length).");

            // convert outputData (formatted in CHW) into list of bounding boxes. More info about CHW format: https://docs.microsoft.com/en-us/cognitive-toolkit/Archive/CNTK-Evaluate-Image-Transforms
            IList<BoundingBox> boundingBoxes = new List<BoundingBox>();
            int channelOffset = gridWidth * gridHeight;
            for (int cy = 0; cy < gridHeight; cy++)
            {
                for (int cx = 0; cx < gridWidth; cx++)
                {
                    for (int b = 0; b < anchors; b++)
                    {
                        int bbOffset = channelOffset * bbLength * b;

                        // get the objectness of the bounding box and skip this bounding box when the objectness is below the theshold
                        var objectness = outputData[bbOffset + channelOffset * 4 + cy * gridWidth + cx];
                        if (objectness < threshold)
                            continue;

                        // dx and dy are relative to the grid cell. So a dx value of 0.5 means that x lays in the middle of grid cell cx, cy
                        var dx = outputData[bbOffset + channelOffset * 0 + cy * gridWidth + cx];
                        var dy = outputData[bbOffset + channelOffset * 1 + cy * gridWidth + cx];

                        // dw and dh are relative to the grid cell. So a dw values of 1.5 meand that the width of the object is 1.5 times the width of the grid cell
                        var dw = outputData[bbOffset + channelOffset * 2 + cy * gridWidth + cx];
                        var dh = outputData[bbOffset + channelOffset * 3 + cy * gridWidth + cx];

                        var x = (cx + dx) * cellWidth; // remember: x is the center of the bounding box
                        var y = (cy + dy) * cellHeight; // remember: y is the center of the bounding box
                        var width = dw * cellWidth;
                        var height = dh * cellHeight;

                        var classes = new float[labels.Length];
                        for (var i = 0; i < labels.Length; i++)
                            classes[i] = outputData[bbOffset + channelOffset * (5 + i) + cy * gridWidth + cx];

                        var topScore = 0f;
                        int topClass = 0;
                        for (int i = 0; i < classes.Length; i++)
                        {
                            if (classes[i] > topScore)
                            {
                                topScore = classes[i];
                                topClass = i;
                            }
                        }

                        // skip this bounding box if the class certenty topscore is smaller then the threshold
                        if (topScore < threshold)
                            continue;

                        boundingBoxes.Add(new BoundingBox()
                        {
                            Confidence = topScore,
                            X = (x - width / 2),
                            Y = (y - height / 2),
                            Width = width,
                            Height = height,
                            Label = labels[topClass]
                        });

                    }
                }
            }

            // filter the bounding boxes by non-max suppression.
            boundingBoxes = Helper.NonMaxSuppression(boundingBoxes, 5, .5f);

            // draw bounding boxes on the image
            foreach (BoundingBox bb in boundingBoxes)
            {
                int x = (int)bb.X;
                int y = (int)bb.Y;
                int width = (int)bb.Width;
                int height = (int)bb.Height;

                // crop the bounding box so it doesn't exceed the dimentions of the image
                if (x < 0)
                {
                    width = width + x;
                    x = 0;
                }
                if (y < 0)
                {
                    height = height + y;
                    y = 0;
                }
                if (x + width > image.Width)
                {
                    width = image.Width - x;
                }
                if (y + height > image.Height)
                {
                    height = image.Height - y;
                }

                Graphics g = Graphics.FromImage(image);

                g.DrawRectangle(new Pen(Brushes.Red, 4f), new Rectangle(x, y, width, height));
                g.DrawString(bb.Label + " " + (bb.Confidence * 100f).ToString("0.00"), new Font("Tahoma", 8), Brushes.Red, x, y);

                g.Flush();
            }

            return image;
        }
    }

    ///// <summary>
    ///// Retrains existing tiny YOLO v2 onnx model on new data
    ///// </summary>
    ///// <param name="device"></param>
    //public static void RetrainTinyYoloV2(DeviceDescriptor device)
    //{
    //    // load model
    //    Function model = Function.Load(TinyYoloV2ModelPath, device, ModelFormat.ONNX);

    //    // get inputVariable and outputVariable of the model
    //    var inputVariable = model.Arguments.First();
    //    var outputVariable = model.Output;

    //    // create a labelVariable
    //    Variable labelVariable = Variable.InputVariable(outputVariable.Shape, outputVariable.DataType);

    //    // create loss and evaluationError Functions
    //    // todo: create the right loss and evaluation functions
    //    var loss = CNTKLib.CrossEntropyWithSoftmax(outputVariable, labelVariable);
    //    var evalError = CNTKLib.ClassificationError(outputVariable, labelVariable);


    //    // create a trainer
    //    // todo: set the right learning parameters
    //    TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.02, 1);
    //    IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(model.Parameters(), learningRatePerSample) };
    //    var trainer = Trainer.CreateTrainer(outputVariable, loss, evalError, parameterLearners);

    //    // create minibatchSource
    //    // todo: create minibatch transformet to transform the annotated data to the label variable format
    //    //var minibatchSource = MinibatchSource.TextFormatMinibatchSource(Path.Combine(ImageDataFolder, "Train_cntk_text.txt"), streamConfigurations, MinibatchSource.InfinitelyRepeat);

    //    // train the network




    //    // save model
    //    // todo: Edit function below so it saves in ModelFormat.ONNX. See: https://github.com/Microsoft/CNTK/issues/3412
    //    //model.Save(Path.Join(Program.RootPath, @"Output\Retrained tiny YOLO v2.onnx"));
    //}

    ///// <summary>
    ///// Creates the Yolo v2 loss function as described in chapter 2.2 of the paper here: https://pjreddie.com/media/files/papers/yolo_1.pdf
    ///// </summary>
    ///// <param name="prediction"></param>
    ///// <param name="labels"></param>
    ///// <returns></returns>
    //private Function YoloV2Loss(Variable prediction, Variable labels)
    //{
    //    float coordinatesLambda = 5f;
    //    float noObjectLambda = .5f;

    //    Function totalLoss;

    //    // todo: continue work here after you added layers to the network that convert the output to real values

    //    Function centerCorrdinatesLoss;
    //    Function heightWidthLoss;
    //    totalLoss = CNTKLib.Plus(centerCorrdinatesLoss, heightWidthLoss);
    //    Function objectnessLoss;
    //    totalLoss = CNTKLib.Plus(totalLoss, objectnessLoss);
    //    Function noObjectnessLoss;
    //    totalLoss = CNTKLib.Plus(totalLoss, noObjectnessLoss);
    //    Function classificationLoss;
    //    totalLoss = CNTKLib.Plus(totalLoss, classificationLoss);

    //    // todo: check if this should return a single value that can be minimized 
    //    // if so, then return the sum of totalLoss (or something like that)
    //    return null;
    //}
}
