using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Media.MediaProperties;

namespace ONNX_model_test_app.Models
{
    public class ResolutionWrapper
    {
        public string Text { get; set; }
        public VideoEncodingProperties VideoProperties { get; set; }
    }
}
