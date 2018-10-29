using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Testing.Models;

namespace Testing.Utils
{
    public static class Helper
    {
        /// <summary>
        /// Deserialize the output of a network into an image.
        /// The output of a network is not serialized as RGB RGB RGB RGB...
        /// but is instead serialized as CHW (Channel Height Width (= BBBB..., GGGG..., RRRR...))
        /// So the RGB value of pixel x,y is:
        /// Red = outputArray[width * height * 2 + y * width + x]
        /// Green = outputArray[width * height + y * width + x]
        /// Blue = outputArray[y * width + x]
        /// </summary>
        /// <param name="modelOutput"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public static Bitmap CreateBitmapFromOutput(IList<float> outputArray, int width, int height)
        {
            Bitmap outputImage = new Bitmap(width, height);

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    byte b = (byte)outputArray[y * width + x];
                    byte g = (byte)outputArray[width * height + y * width + x];
                    byte r = (byte)outputArray[width * height * 2 + y * width + x];
                    Color color = Color.FromArgb(r, g, b);
                    outputImage.SetPixel(x, y, color);
                }
            }
            return outputImage;
        }

        /// <summary>
        /// Resizes an image
        /// </summary>
        /// <param name="image">The image to resize</param>
        /// <param name="width">New width in pixels</param>
        /// <param name="height">New height in pixels</param>
        /// <param name="useHighQuality">Resize quality</param>
        /// <returns>The resized image</returns>
        public static Bitmap Resize(this Bitmap image, int width, int height, bool useHighQuality)
        {
            var newImg = new Bitmap(width, height);

            using (var g = Graphics.FromImage(newImg))
            {
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                if (useHighQuality)
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                }
                else
                {
                    g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Default;
                    g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.Default;
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Default;
                }

                var attributes = new ImageAttributes();
                attributes.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                g.DrawImage(image, new Rectangle(0, 0, width, height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);
            }

            return newImg;
        }

        /// <summary>
        /// Extracts image pixels in CHW (Channel Height Width
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in HWC order</returns>
        public static List<float> ExtractCHW(this Bitmap image)
        {
            var features = new List<float>(image.Width * image.Height * 3);
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < image.Height; h++)
                {
                    for (int w = 0; w < image.Width; w++)
                    {
                        var pixel = image.GetPixel(w, h);
                        float v = c == 0 ? pixel.B : c == 1 ? pixel.G : pixel.R;

                        features.Add(v);
                    }
                }
            }

            return features;
        }

        /// <summary>
        /// Creates an input dictionary from the image for the input variable.
        /// Will transfor the image to comply with the input variable constraints
        /// </summary>
        /// <param name="image"></param>
        /// <param name="inputVariable"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        public static Dictionary<Variable, Value> CreateInputDataMap(Bitmap image, Variable inputVariable, DeviceDescriptor device)
        {
            // Get shape data for the input variable, and optionally remove the free dimension
            NDShape variableShape = inputVariable.Shape.SubShape(0, 3);

            int inputWidth = variableShape[0];
            int inputHeight = variableShape[1];

            // Image preprocessing to match input requirements of the model.
            var resized = image.Resize(inputWidth, inputHeight, true);
            List<float> resizedCHW = resized.ExtractCHW();

            // Create input data map
            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(variableShape, resizedCHW, device);
            inputDataMap.Add(inputVariable, inputVal);

            return inputDataMap;
        }

        public static float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        /// <summary>
        /// Converts the array of (arbitrary) values into an array of probabilities
        /// info see: https://en.wikipedia.org/wiki/Softmax_function
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();

            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        /// <summary>
        /// More info about non-max suppression here: https://www.youtube.com/watch?v=A46HZGR5fMw&list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud&t=6s&index=30
        /// </summary>
        /// <param name="boxes"></param>
        /// <param name="limit"></param>
        /// <param name="threshold"></param>
        /// <returns></returns>
        public static IList<BoundingBox> NonMaxSuppression(IList<BoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];

            for (var i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;

            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                                .OrderByDescending(b => b.Box.Confidence)
                                .ToList();

            var results = new List<BoundingBox>();

            for (var i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);

                    if (results.Count >= limit)
                        break;

                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;

                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;

                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }

                    if (activeCount <= 0)
                        break;
                }
            }

            return results;
        }

        private static float IntersectionOverUnion(RectangleF a, RectangleF b)
        {
            var areaA = a.Width * a.Height;

            if (areaA <= 0)
                return 0;

            var areaB = b.Width * b.Height;

            if (areaB <= 0)
                return 0;

            var minX = Math.Max(a.Left, b.Left);
            var minY = Math.Max(a.Top, b.Top);
            var maxX = Math.Min(a.Right, b.Right);
            var maxY = Math.Min(a.Bottom, b.Bottom);

            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

            return intersectionArea / (areaA + areaB - intersectionArea);
        }
    }
}
