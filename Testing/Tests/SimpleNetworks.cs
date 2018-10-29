using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Testing.Tests
{
    public class SimpleNetworks
    {
        /// <summary>
        /// Logistic regression example copied from the Microsoft CNTK repo
        /// </summary>
        /// <param name="device"></param>
        public static void LogisticRegression(DeviceDescriptor device)
        {
            int inputDim = 3;
            int numOutputClasses = 2;

            Variable featureVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            Variable labelVariable = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            // create network
            var weightParam = new Parameter(new int[] { numOutputClasses, inputDim }, DataType.Float, 1, device, "w");
            var biasParam = new Parameter(new int[] { numOutputClasses }, DataType.Float, 0, device, "b");
            var classifierOutput = CNTKLib.Times(weightParam, featureVariable) + biasParam;

            var loss = CNTKLib.CrossEntropyWithSoftmax(classifierOutput, labelVariable);
            var evalError = CNTKLib.ClassificationError(classifierOutput, labelVariable);

            // prepare for training
            TrainingParameterScheduleDouble learningRatePerSample = new TrainingParameterScheduleDouble(0.02, 1);
            IList<Learner> parameterLearners =
                new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, loss, evalError, parameterLearners);

            int minibatchSize = 64;
            int numMinibatchesToTrain = 1000;
            int updatePerMinibatches = 50;

            // train the model
            for (int minibatchCount = 0; minibatchCount < numMinibatchesToTrain; minibatchCount++)
            {
                Value features, labels;
                GenerateValueData(minibatchSize, inputDim, numOutputClasses, out features, out labels, device);
                //TODO: sweepEnd should be set properly instead of false.
#pragma warning disable 618
                trainer.TrainMinibatch(
                    new Dictionary<Variable, Value>() { { featureVariable, features }, { labelVariable, labels } }, device);
#pragma warning restore 618
                PrintTrainingProgress(trainer, minibatchCount, updatePerMinibatches);
            }

            // test and validate the model
            int testSize = 100;
            Value testFeatureValue, expectedLabelValue;
            GenerateValueData(testSize, inputDim, numOutputClasses, out testFeatureValue, out expectedLabelValue, device);

            // GetDenseData just needs the variable's shape
            IList<IList<float>> expectedOneHot = expectedLabelValue.GetDenseData<float>(labelVariable);
            IList<int> expectedLabels = expectedOneHot.Select(l => l.IndexOf(1.0F)).ToList();

            var inputDataMap = new Dictionary<Variable, Value>() { { featureVariable, testFeatureValue } };
            var outputDataMap = new Dictionary<Variable, Value>() { { classifierOutput.Output, null } };
            classifierOutput.Evaluate(inputDataMap, outputDataMap, device);
            var outputValue = outputDataMap[classifierOutput.Output];
            IList<IList<float>> actualLabelSoftMax = outputValue.GetDenseData<float>(classifierOutput.Output);
            var actualLabels = actualLabelSoftMax.Select((IList<float> l) => l.IndexOf(l.Max())).ToList();
            int misMatches = actualLabels.Zip(expectedLabels, (a, b) => a.Equals(b) ? 0 : 1).Sum();

            Console.WriteLine($"Validating Model: Total Samples = {testSize}, Misclassify Count = {misMatches}");
        }

        private static void GenerateValueData(int sampleSize, int inputDim, int numOutputClasses,
            out Value featureValue, out Value labelValue, DeviceDescriptor device)
        {
            float[] features;
            float[] oneHotLabels;
            GenerateRawDataSamples(sampleSize, inputDim, numOutputClasses, out features, out oneHotLabels);

            featureValue = Value.CreateBatch<float>(new int[] { inputDim }, features, device);
            labelValue = Value.CreateBatch<float>(new int[] { numOutputClasses }, oneHotLabels, device);
        }

        private static void GenerateRawDataSamples(int sampleSize, int inputDim, int numOutputClasses,
            out float[] features, out float[] oneHotLabels)
        {
            Random random = new Random(0);

            features = new float[sampleSize * inputDim];
            oneHotLabels = new float[sampleSize * numOutputClasses];

            for (int sample = 0; sample < sampleSize; sample++)
            {
                int label = random.Next(numOutputClasses);
                for (int i = 0; i < numOutputClasses; i++)
                {
                    oneHotLabels[sample * numOutputClasses + i] = label == i ? 1 : 0;
                }

                for (int i = 0; i < inputDim; i++)
                {
                    features[sample * inputDim + i] = (float)GenerateGaussianNoise(3, 1, random) * (label + 1);
                }
            }
        }

        /// <summary>
        /// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        /// https://stackoverflow.com/questions/218060/random-gaussian-variables
        /// </summary>
        /// <returns></returns>
        static double GenerateGaussianNoise(double mean, double stdDev, Random random)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double stdNormalRandomValue = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * stdNormalRandomValue;
        }

        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                float trainLossValue = (float)trainer.PreviousMinibatchLossAverage();
                float evaluationValue = (float)trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropyLoss = {trainLossValue}, EvaluationCriterion = {evaluationValue}");
            }
        }
    }
}
