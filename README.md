# AutoGradMLP
Testing a simple autogradient calculation for a Neural Network

The code for XOR dummy dataset

```
double[] dataX = { 0, 0, 0, 1, 1, 0, 1, 1 };
double[] dataY = { 0, 1, 1, 0 };
var X = ND.CreateNDarray(dataX, 4, 2);
var Y = ND.CreateNDarray(dataY, 4, 1);

var MLP = new Chain(inNodes: 2)
    .AddDenseLayer(outNodes: 8)
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
```


The Output

```
Hello World! AutoGradient MultiLayers Neurals Network
Epochs:    0/1000 loss:0.153728
Epochs:  100/1000 loss:0.070206
Epochs:  200/1000 loss:0.031820
Epochs:  300/1000 loss:0.016029
Epochs:  400/1000 loss:0.009587
Epochs:  500/1000 loss:0.006502
Epochs:  600/1000 loss:0.004793
Epochs:  700/1000 loss:0.003739
Epochs:  800/1000 loss:0.003037
Epochs:  900/1000 loss:0.002540
Epochs: 1000/1000 loss:0.002174
Time:   111 ms

Prediction
[[  0.04370011]
 [  0.93628556]
 [  0.92778170]
 [  0.07860396]]

```

### Reference
The code was inspired from this Python repository https://github.com/evcu/numpy_autograd
