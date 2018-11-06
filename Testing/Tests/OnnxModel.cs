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
            float threshold = 0.3f;

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

            return VisualizeOutput(image, outputArray, labels, anchorCoordintes, outputVariable.Shape[0], outputVariable.Shape[1], threshold);
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
                CNTKDictionary anchorWidth = CNTKLib.ConstantInitializer(anchorCoordintes[i, 0]); // here, and in the line below, we create a one channel parameter filled with the width of the anchor box
                Parameter anchorWidthParameter = new Parameter(widthSlice.Output.Shape.SubShape(0, 3), DataType.Float, anchorWidth, device, $"anchor width for anchor box {i}");
                Function convertedWidth = CNTKLib.ElementTimes(widthExponentials, anchorWidthParameter, $"Multiply tw times the anchor box width for anchor box {i}");
                convertedVariables.Add(convertedWidth);

                // convert next channel: e^(th)*anchorCoordinates
                Function heightSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 3 }, new IntVector() { anchorOffset + 4 }, $"Slice th for anchor box {i}");
                Function heigthExponentials = CNTKLib.Exp(heightSlice, $"Exponentiate th for anchor box {i}");
                CNTKDictionary anchorHeight = CNTKLib.ConstantInitializer(anchorCoordintes[i, 1]); // here, and in the line below, we create a one channel parameter filled with the height of the anchor box
                Parameter anchorHeightParameter = new Parameter(heightSlice.Output.Shape.SubShape(0, 3), DataType.Float, anchorHeight, device, $"anchor height for anchor box {i}");
                Function convertedHeight = CNTKLib.ElementTimes(heigthExponentials, anchorHeightParameter, $"Multiply th times the anchor box width for anchor box {i}");
                convertedVariables.Add(convertedHeight);

                // sigmoid the next channel. This channel contains tc
                Function objectnessSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 }, $"Slice tc for anchor box {i}");
                Function objectnessSigmoid = CNTKLib.Sigmoid(objectnessSlice, $"Sigmoid tc for anchor box {i}");
                convertedVariables.Add(objectnessSigmoid);

                // Softmax the remaining channels of this anchor. These channels contain the c1, c2, ... values
                Function classesSlice = CNTKLib.Slice(input, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes }, $"Slice all classes for anchor box {i}");
                Function classSoftMax = CNTKLib.Softmax(classesSlice, new Axis(2), $"Softmax classes values for anchor box {i}");
                convertedVariables.Add(classSoftMax);
            }

            // concatenate the converted channels back in the right order
            return CNTKLib.Splice(convertedVariables, new Axis(2), "Splice all converted channels back together");
        }

        /// <summary>
        /// Processes the output of the network on to the image. 
        /// This function assumes that the tx, ty, tw, th, tc, c0, c1, ... values of the network are already converted to relative values
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

        /// <summary>
        /// Retrains existing tiny YOLO v2 onnx model on new data
        /// </summary>
        /// <param name="device"></param>
        public static void RetrainTinyYoloV2(DeviceDescriptor device)
        {
            string[] labels = new string[] { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
            float[,] anchorCoordintes = { { 1.08f, 1.19f }, { 3.42f, 4.41f }, { 6.63f, 11.38f }, { 9.42f, 5.11f }, { 16.62f, 10.52f } }; // pairs of height and weight for every anchor box
            float threshold = 0.3f;

            // load model
            Function model = Function.Load(TinyYoloV2ModelPath, device, ModelFormat.ONNX);
            model = YoloDetectionLayer(model, anchorCoordintes, labels.Length, device);

            // get inputVariable and outputVariable of the model
            var inputVariable = model.Arguments.First();
            var outputVariable = model.Output;

            // create a labelVariable
            Variable labelVariable = Variable.InputVariable(outputVariable.Shape, outputVariable.DataType);

            // create loss and evaluationError Functions
            var loss = YoloV2Loss(outputVariable, labelVariable, anchorCoordintes.GetLength(0), labels.Length, device);
            var evalError = YoloV2EvaluationError(outputVariable, labelVariable, anchorCoordintes.GetLength(0), labels.Length, threshold, device);

            // create a trainer
            // todo: set the right learning parameters
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.0001, 1);
            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(model.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(outputVariable, loss, evalError, parameterLearners);

            // create minibatchSource
            // before any continuation is possible this issue has to be resolved: https://github.com/Microsoft/CNTK/issues/3497

            // based on https://docs.microsoft.com/en-us/cognitive-toolkit/brainscript-and-python---understanding-and-extending-readers#configuring-a-reader-minibatch-source-in-python
            // based on https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_feed_data.ipynb

            // todo find a way to define a userDeserializar. This is blocking. See: https://cntk.ai/pythondocs/cntk.io.html#cntk.io.UserMinibatchSource & https://www.cntk.ai/pythondocs/Manual_How_to_write_a_custom_deserializer.html
            // todo: lookup the code for FasterRCNN, and replicate that in c#. See: https://github.com/Microsoft/CNTK/tree/release/latest/Examples/Image/Detection

            var deserializerConfiguration = CNTKLib.ImageDeserializer("mapFilePath", "noUse", 0, "features"); // todo: maybe add streamDef in the list. see: https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_feed_data.ipynb
            // todo: maybe use the line below, but set sparse to true, and transform the label date pre training
            var labelDeserializer = CNTKLib.CTFDeserializer("path", new StreamConfigurationVector(new List<StreamConfiguration>() { new StreamConfiguration("label", 10, false)})); // todo this must be a custom deserializer

            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { deserializerConfiguration, labelDeserializer }) // todo: lookup how to sync the two deserializers
            {
                MaxSweeps = MinibatchSource.InfinitelyRepeat
            };

            var minibatchSource =  CNTKLib.CreateCompositeMinibatchSource(config);




            // todo: create minibatch transformet to transform the annotated data to the label variable format

            // todo: make sure that the input is transposed from HWC to CHW



            // train the network




            // save model
            // todo: Edit function below so it saves in ModelFormat.ONNX. See: https://github.com/Microsoft/CNTK/issues/3412
            //model.Save(Path.Join(Program.RootPath, @"Output\Retrained tiny YOLO v2.onnx"));
        }

        /// <summary>
        /// Creates the Yolo v2 loss function as described in chapter 2.2 of the paper here: https://pjreddie.com/media/files/papers/yolo_1.pdf
        /// The problem of this loss function is that the penalization for a bounding box with high IOU is the same as the penalization for one with a low IOU, as long as bouth bounding boxes are centered in a other grid/anchor cells
        /// Some other examples of the loss function do take the IOU int account. see:
        ///     - https://github.com/santoshgsk/yolov2-pytorch/blob/master/yolotorch.ipynb
        ///     - https://github.com/pjreddie/darknet/blob/master/src/detection_layer.c#L50
        /// </summary>
        /// <param name="prediction">Prediction made by the model</param>
        /// <param name="truth">Ground truth data</param>
        /// <param name="anchors">Number of anchors used by the model</param>
        /// <param name="classes">Number of label classes used by the model</param>
        /// <param name="device"></param>
        /// <returns>Yolo v2 loss function</returns>
        private static Function YoloV2Loss(Variable prediction, Variable truth, int anchors, int classes, DeviceDescriptor device)
        {
            float coordinatesLambda = 5f;
            float noObjectLambda = .5f;

            // we create a new function with the same dimension as the prediction variable. 
            // This new "delta" function (see after the for loop) contains all errors which are calculated by the Yolo v2 loss function. See here: https://pjreddie.com/media/files/papers/yolo_1.pdf
            // For example: 
            //  When a particular cell value in the prediction Varibale contain the dx coordinate, 
            //  then this cell in the "delta" function will contain the squared x-error (when there was an object detection for that anchor box)

            VariableVector deltaChannels = new VariableVector();
            for (int i = 0; i < anchors; i++)
            {
                int anchorOffset = (5 + classes) * i;

                // objectnessTruth is the "1^(obj)" from loss function in the paper
                Function objectnessTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                NDShape singleChannelShape = objectnessTruth.Output.Shape.SubShape(0, 3);

                // noobjectnessTruth is the "1^(noobj)" from loss function in the paper
                Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), device);
                Function noobjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);

                // coordinatesLambdaChannel is the LAMBDA(coord) from the loss function in the paper
                Parameter coordinatesLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(coordinatesLambda), device);

                // loss due to differences in centers
                Function centerPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerDifference = CNTKLib.Minus(centerPrediction, centerTruth);
                Function centerLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(centerDifference)));
                deltaChannels.Add(centerLoss); // Add 1 Function of 2 channels to the vector

                // loss due to differences in width/height
                Function dimensionPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionSqrtDifference = CNTKLib.Minus(CNTKLib.Sqrt(dimensionPrediction), CNTKLib.Sqrt(dimensionTruth));
                Function dimensionLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(dimensionSqrtDifference)));
                deltaChannels.Add(dimensionLoss); // Add 1 Function of 2 channels to the vector

                // noObjectLambdaChannel is the LAMBDA(noobj) from the loss function in the paper
                Parameter noObjectLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(noObjectLambda), device);
                // objectnessWeight is the combination of 1^(obj) + LAMBDA(noobj) * 1^(noobj)
                Function objectnessWeight = CNTKLib.Plus(objectnessTruth, CNTKLib.ElementTimes(noObjectLambdaChannel, noobjectnessTruth)); // todo this could potentially be wrong as the documentation describes that this is an element wise BINARY addition

                // loss due to difference in objectness
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessDifference = CNTKLib.Minus(objectnessPrediction, objectnessTruth);
                Function objectnessLoss = CNTKLib.ElementTimes(objectnessWeight, CNTKLib.Square(objectnessDifference));
                deltaChannels.Add(objectnessLoss); // Add 1 Function of 1 channels to the vector

                // loss due to difference in class predictions
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes });
                Function classesTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes });
                Function classesDifference = CNTKLib.Minus(classesPrediction, classesTruth);
                Function classesLoss = CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(classesDifference));
                deltaChannels.Add(classesLoss); // Add 1 Function of "classes" channels to the vector
            }

            // As described above the delta function contains all losses
            Function delta = CNTKLib.Splice(deltaChannels, new Axis(2));

            //// todo: if the loss Function can't be a multidimensional tensor, then reduce the dimension by taking the sum
            //Function totalSum = CNTKLib.ReduceSum(delta, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });
            //return CNTKLib.Reshape(totalSum, new int[] { 1 });

            return delta;
        }

        /// <summary>
        /// Creates the evaluation error function, which returns the ratio of # bad detections / # good detections
        /// A good detection is when:
        ///     1. True Positive: Correctly detect the class in the grid cell/ anchor box
        ///     2. True Negative: Correctly detect no object in the grid cell/ anchor box
        /// A bad detection is wehn:
        ///     1. False Negative: Incoreectly detect no object, when an objecti is present in the grid cell/ anchor box
        ///     2. False Positive: Incoreectly detects an object when no object is present in the grid cell/ anchor box
        ///     3. Missclassification: Correctly detects an object, but misclassifies it in the grid cell/ anchor box
        /// Please note that this evaluation function doesn't take IOU (Intersection over Union) into account.
        ///     In fact it doesn't take any overlap into account at all.
        /// </summary>
        /// <param name="prediction">Prediction made by the model</param>
        /// <param name="truth">Ground truth data</param>
        /// <param name="anchors">Number of anchors used by the model</param>
        /// <param name="classes">Number of label classes used by the model</param>
        /// <param name="threshold">certainty threshold for the objectness and classness of a bounding box</param>
        /// <param name="device"></param>
        /// <returns>Yolo v2 evaluation error function</returns>
        private static Function YoloV2EvaluationError(Variable prediction, Variable truth, int anchors, int classes, float threshold, DeviceDescriptor device)
        {
            // thresholdChannel is a one channel parameter filled with the threshold value
            NDShape singleChannelShape = new int[] { prediction.Shape[0], prediction.Shape[1], 1 };
            Parameter thresholdChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(threshold), device);
            Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), device);

            VariableVector truePredictionChannels = new VariableVector();
            VariableVector falsePredictionChannels = new VariableVector();
            for (int i = 0; i < anchors; i++)
            {
                int anchorOffset = (5 + classes) * i;

                Function objectnessTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                objectnessPrediction = CNTKLib.GreaterEqual(objectnessPrediction, thresholdChannel); // convert to 0 where the value is below threshold, and 1 if it is above threshold

                Function noObjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);
                Function noObjectnessPrediction = CNTKLib.Minus(onesChannel, objectnessPrediction);

                // todo class prediction should be set to false when it is below threshold
                Function classesTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes });
                classesTruth = CNTKLib.Argmax(classesTruth, new Axis(2)); // Reduce the tensor to one channel; filled with the index of the maximum value (= the index of the class)
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 5 }, new IntVector() { anchorOffset + 5 + classes });
                classesPrediction = CNTKLib.Argmax(classesPrediction, new Axis(2)); // Reduce the tensor to one channel; filled with the index of the maximum value (= the index of the class)

                // True Positives (tensor contains 1 if yes, 0 otherwise) 
                Function truePositives = CNTKLib.ElementTimes(objectnessTruth, objectnessPrediction); // should detect an object, and detects an object
                truePositives = CNTKLib.ElementTimes(truePositives, CNTKLib.Equal(classesTruth, classesPrediction)); // and should detecect the same class

                // True Negatives (tensor contains 1 if yes, 0 otherwise)
                Function trueNegatives = CNTKLib.ElementTimes(noObjectnessTruth, noObjectnessPrediction); // should detect no object, and detects no object

                // False Negatives (tensor contains 1 if yes, 0 otherwise)
                Function falseNegatives = CNTKLib.ElementTimes(objectnessTruth, noObjectnessPrediction); // should detect an object, but detects no object

                // False Positives (tensor contains 1 if yes, 0 otherwise)
                Function falsePositives = CNTKLib.ElementTimes(noObjectnessTruth, objectnessPrediction); // should detect no object, but detects an object

                // Missclassification (tensor contains 1 if yes, 0 otherwise)
                Function missclassifications = CNTKLib.ElementTimes(objectnessTruth, objectnessPrediction); // should detect an object, and detects an object
                missclassifications = CNTKLib.ElementTimes(missclassifications, CNTKLib.NotEqual(classesTruth, classesPrediction)); // but detects a different class

                truePredictionChannels.Add(truePositives);
                truePredictionChannels.Add(trueNegatives);

                falsePredictionChannels.Add(falseNegatives);
                falsePredictionChannels.Add(falsePositives);
                falsePredictionChannels.Add(missclassifications);
            }

            Function allGoodPredictions = CNTKLib.Splice(truePredictionChannels, new Axis(2));
            Function allbadPredictions = CNTKLib.Splice(falsePredictionChannels, new Axis(2));

            // sum the variable vector
            Function goodPredictionSum = CNTKLib.ReduceSum(allGoodPredictions, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });
            Function badPredictionSum = CNTKLib.ReduceSum(allbadPredictions, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });

            // fail safe: The value of goodPredictionSum isn't allowed to be 0, because the division at the end would fail.
            Parameter one = new Parameter(new int[] { 1, 1, 1 }, DataType.Float, CNTKLib.ConstantInitializer(1f), device);
            goodPredictionSum = CNTKLib.ElementMax(goodPredictionSum, one, "Fail safe");

            // device the summations (Take into account the case where there are 0 true predictions
            return CNTKLib.ElementDivide(badPredictionSum, goodPredictionSum);
        }
    }
}
