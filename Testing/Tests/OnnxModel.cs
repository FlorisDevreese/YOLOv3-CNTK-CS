using CNTK;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Testing.Models;
using Testing.Utils;

namespace Testing.Tests
{
    public class OnnxModel
    {
        public static Bitmap TestModel(string modelPath, Bitmap image, DeviceDescriptor device)
        {
            Function model = Function.Load(modelPath, device, ModelFormat.ONNX);

            var inputVariable = model.Arguments.First();
            var outputVariable = model.Output;

            var inputMap = Helper.CreateInputDataMap(image, inputVariable, device);
            var outputDataMap = new Dictionary<Variable, Value>() { { outputVariable, null } };

            model.Evaluate(inputMap, outputDataMap, device);

            var output = outputDataMap[outputVariable];
            var outputArray = output.GetDenseData<float>(outputVariable).First();

            // the labels and anchor box coordinates are dependent on how the model is learned.
            string[] labels = new string[] { "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
            float[,] anchorCoordintes = { { 1.08f, 1.19f }, { 3.42f, 4.41f }, { 6.63f, 11.38f }, { 9.42f, 5.11f }, { 16.62f, 10.52f } }; // pairs of height and weight for every anchor box

            return VisualizeResults(image, outputArray, labels, anchorCoordintes, 13, 13, 5, 0.3f);
        }

        /// <summary>
        /// Processes the output of the network on to the image
        /// </summary>
        /// <param name="image">original image on which the result is drawn</param>
        /// <param name="outputData">output data of the network</param>
        /// <param name="labels">string array with all the names of the classes</param>
        /// <param name="gridWidth">Number of cells the width of the image is devided in</param>
        /// <param name="gridHeight">Number of cells the height of the image is devided in</param>
        /// <param name="bbPerCell">number of bounding boxes per cell</param>
        /// <param name="threshold">certainty threshold for the objectness and classness of a bounding box</param>
        /// <returns>annotated image</returns>
        private static Bitmap VisualizeResults(Bitmap image, IList<float> outputData, string[] labels, float[,] anchorCoordintes, int gridWidth, int gridHeight, int bbPerCell, float threshold)
        {
            int bbLength = (5 + labels.Length);
            int cellWidth = image.Width / gridWidth;
            int cellHeight = image.Height / gridHeight;

            // validate input
            if (outputData.Count != gridWidth * gridHeight * bbPerCell * bbLength)
                throw new ArgumentException("The count of outputData.Count should be equal to: gridWidth * gridHeight * bbPerCell * (5 + labels.Length).");

            // convert outputData (formatted in CHW) into list of bounding boxes. More info about CHW format: https://docs.microsoft.com/en-us/cognitive-toolkit/Archive/CNTK-Evaluate-Image-Transforms
            IList<BoundingBox> boundingBoxes = new List<BoundingBox>();
            int channelOffset = gridWidth * gridHeight;
            for (int cy = 0; cy < gridHeight; cy++)
            {
                for (int cx = 0; cx < gridWidth; cx++)
                {
                    for (int b = 0; b < bbPerCell; b++)
                    {
                        int bbOffset = channelOffset * bbLength * b;

                        // get the objectness of the bounding box and skip this bounding box when the objectness is below the theshold
                        var tc = outputData[bbOffset + channelOffset * 4 + cy * gridWidth + cx];
                        var objectness = Helper.Sigmoid(tc);
                        if (objectness < threshold)
                            continue;

                        var tx = outputData[bbOffset + channelOffset * 0 + cy * gridWidth + cx];
                        var ty = outputData[bbOffset + channelOffset * 1 + cy * gridWidth + cx];
                        var tw = outputData[bbOffset + channelOffset * 2 + cy * gridWidth + cx];
                        var th = outputData[bbOffset + channelOffset * 3 + cy * gridWidth + cx];

                        var x = (cx + Helper.Sigmoid(tx)) * cellWidth;
                        var y = (cy + Helper.Sigmoid(ty)) * cellHeight;
                        var width = (float)Math.Exp(tw) * cellWidth * anchorCoordintes[b, 0];
                        var height = (float)Math.Exp(th) * cellHeight * anchorCoordintes[b, 1];

                        var classes = new float[labels.Length];
                        for (var i = 0; i < labels.Length; i++)
                            classes[i] = outputData[bbOffset + channelOffset * (5 + i) + cy * gridWidth + cx];

                        var results = Helper.Softmax(classes);

                        var topScore = 0f;
                        int topClass = 0;
                        for (int i = 0; i < results.Length; i++)
                        {
                            if (results[i] > topScore)
                            {
                                topScore = results[i];
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
}
