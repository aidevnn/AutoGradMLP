using System;
using System.Diagnostics;
namespace AutoGradMLP
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World! AutoGradient MultiLayers Neurals Network");

            Utils.random = new Random(123);

            double[] dataX = { 0, 0, 0, 1, 1, 0, 1, 1 };
            double[] dataY = { 0, 1, 1, 0 };
            var X = ND.CreateNDarray(dataX, 4, 2);
            var Y = ND.CreateNDarray(dataY, 4, 1);

            var MLP = new Chain(inNodes: 2)
                .AddDenseLayer(outNodes: 4)
                .AddTanhActivation()
                .AddDenseLayer(outNodes: 4)
                .AddTanhActivation()
                .AddDenseLayer(outNodes: 1)
                .AddSigmoidActivation();

            int epochs = 1000, displayEpoch = 100;
            var sw = Stopwatch.StartNew();
            for (int k = 0; k <= epochs; ++k)
            {
                MLP.Forward(X);
                var loss = MLP.Loss(Y);
                if (k % displayEpoch == 0)
                    Console.WriteLine($"Epochs:{k,5}/{epochs} loss:{loss:0.000000}");

                MLP.Backward(Y);
                MLP.UpdateWeightsSGD(0.1);
                MLP.ResetGradient();
            }

            Console.WriteLine($"Time:{sw.ElapsedMilliseconds,6} ms");
            Console.WriteLine();
            Console.WriteLine("Prediction");
            Console.WriteLine(MLP.Prediction(X));
        }
    }
}
