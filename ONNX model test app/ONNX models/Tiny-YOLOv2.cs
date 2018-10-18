using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.AI.MachineLearning; // ATTENTION minimum Windows 10 version 1809 (10.0; build 17763) needed
using System.Linq;

namespace ONNX_model_test_app.ONNX_models
{
    public sealed class TinyYoloV2ModelInput
    {
        public VideoFrame Image { get; set; }
    }

    public sealed class TinyYoloV2ModelOutput
    {
        public IList<float> Grid { get; set; }

        public TinyYoloV2ModelOutput()
        {
            Grid = new List<float>();
        }
    }

    // todo the functionality below is only availble after the Windows 10 october update. 
    // info on implementation: https://docs.microsoft.com/en-us/windows/ai/integrate-model & https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning.learningmodel.inputfeatures
    // examples see: C:\Users\flori\Documents\Code\Windows-Machine-Learning
    public sealed class TinyYoloV2Model
    {
        private LearningModel learningModel;
        public static async Task<TinyYoloV2Model> CreateModel(StorageFile file)
        {
            LearningModel model = await LearningModel.LoadFromStorageFileAsync(file);


            // todo investigate the input image description and the outputDescription.

            //Get input and output features of the model
            List<ILearningModelFeatureDescriptor> inputFeatures = model.InputFeatures.ToList();
            List<ILearningModelFeatureDescriptor> outputFeatures = model.OutputFeatures.ToList();

            // Retrieve the first input feature which is an image
            var _inputImageDescription = inputFeatures.FirstOrDefault(
                feature => feature.Kind == LearningModelFeatureKind.Image) as ImageFeatureDescriptor;

            // Retrieve the first output feature which is a tensor
            var _outputImageDescription = outputFeatures.FirstOrDefault(
                feature => feature.Kind == LearningModelFeatureKind.Tensor) as TensorFeatureDescriptor;

            return new TinyYoloV2Model() { learningModel = model };
        }
        public async Task<TinyYoloV2ModelOutput> EvaluateAsync(TinyYoloV2ModelInput input)
        {
            TinyYoloV2ModelOutput output = new TinyYoloV2ModelOutput();
            // todo decomment the lines below. + Make them work
            //LearningModelBinding binding = new LearningModelBinding(learningModel);
            //binding.Bind("image", input.Image);
            //binding.Bind("grid", output.Grid);
            //LearningModelEvaluationResult evalResult = await learningModel.EvaluateAsync(binding, string.Empty);
            return output;
        }
    }
}
