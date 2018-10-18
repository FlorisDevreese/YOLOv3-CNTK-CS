using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Testing.Utils
{
    public static class Helper
    {
        /// <summary>
        /// Deserialize the output of a network into an image.
        /// The output of a network is not serialized as RGB RGB RGB RGB...
        /// but is instead serialized as: BBBB..., GGGG..., RRRR...
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
            // todo: look if you can't convert this to the way it is done in the UWP project
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
        /// Extracts image pixels in CHW using parallelization
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in CHW order</returns>
        public static List<float> ParallelExtractCHW(this Bitmap image)
        {
            // We use local variables to avoid contention on the image object through the multiple threads.
            int channelStride = image.Width * image.Height;
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            var features = new byte[imageWidth * imageHeight * 3];
            var bitmapData = image.LockBits(new System.Drawing.Rectangle(0, 0, imageWidth, imageHeight), ImageLockMode.ReadOnly, image.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
            byte[] rgbValues = new byte[bytes];

            int stride = bitmapData.Stride;

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

            // The mapping depends on the pixel format
            // The mapPixel lambda will return the right color channel for the desired pixel
            Func<int, int, int, int> mapPixel = GetPixelMapper(image.PixelFormat, stride);

            Parallel.For(0, imageHeight, (int h) =>
            {
                Parallel.For(0, imageWidth, (int w) =>
                {
                    Parallel.For(0, 3, (int c) =>
                    {
                        features[channelStride * c + imageWidth * h + w] = rgbValues[mapPixel(h, w, c)];
                    });
                });
            });

            image.UnlockBits(bitmapData);

            return features.Select(b => (float)b).ToList();
        }

        /// <summary>
        /// Returns a function for extracting the R-G-B values properly from an image based on its pixel format
        /// </summary>
        /// <param name="pixelFormat">The image's pixel format</param>
        /// <param name="heightStride">The stride (row byte count)</param>
        /// <returns>A function with signature (height, width, channel) returning the corresponding color value</returns>
        private static Func<int, int, int, int> GetPixelMapper(PixelFormat pixelFormat, int heightStride)
        {
            switch (pixelFormat)
            {
                case PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * heightStride + w * 4 + c;  // bytes are B-G-R-A
                case PixelFormat.Format24bppRgb:
                default:
                    return (h, w, c) => h * heightStride + w * 3 + c;  // bytes are B-G-R
            }
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
            // Get shape data for the input variable
            NDShape inputShape = inputVariable.Shape;
            int imageWidth = inputShape[0];
            int imageHeight = inputShape[1];

            // Image preprocessing to match input requirements of the model.
            var resized = image.Resize(imageWidth, imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            var inputDataMap = new Dictionary<Variable, Value>();
            var inputVal = Value.CreateBatch(inputShape, resizedCHW, device);
            inputDataMap.Add(inputVariable, inputVal);

            return inputDataMap;
        }
    }
}
