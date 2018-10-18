using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

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
    }
}
