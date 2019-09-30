using System;
using System.Linq;

namespace AutoGradMLP
{
    public abstract class Function
    {
        public NDarray<double> Y, Grad;
        public abstract void Forward();
        public abstract void Backward(NDarray<double> dY);
        public abstract void PropResetGrad();

        public string Name = Guid.NewGuid().ToString();

        public void ResetGrad()
        {
            if (Grad != null)
                Grad = Grad.ApplyFunc(x => 0.0);
        }

        public void UpdateSGD(double lr)
        {
            Y = ND.Sub(Y, Grad.ApplyFunc(x => x * lr));
        }
    }

    public class Variable : Function
    {
        public Variable(string Name, NDarray<double> Y)
        {
            this.Name = Name;
            this.Y = Y;
            Grad = new NDarray<double>(Y.Shape);
        }

        public override void Backward(NDarray<double> dY) { Grad = dY; }

        public override void Forward() { }

        public override void PropResetGrad()
        {
            ResetGrad();
        }

        public void SetValue(NDarray<double> nDarray) => Y = nDarray;
    }

    public class UnaryFunction : Function
    {
        public UnaryFunction(Function function, Func<double, double> func, Func<double, double> grad)
        {
            this.function = function;
            this.func = func;
            this.grad = grad;
        }

        readonly Function function;
        readonly Func<double, double> func, grad;

        public override void Backward(NDarray<double> dY)
        {
            Grad = ND.Mul(Y.ApplyFunc(grad), dY);
            function.Backward(Grad);
        }

        public override void Forward()
        {
            function.Forward();
            Y = function.Y.ApplyFunc(func);

            if (Grad == null)
                Grad = new NDarray<double>(Y.Shape);
        }

        public override void PropResetGrad()
        {
            ResetGrad();
            function.PropResetGrad();
        }
    }

    public class SigFunc : UnaryFunction
    {
        public SigFunc(Function function) : base(function, x => 1.0 / (1.0 + Math.Exp(-x)), y => y * (1.0 - y))
        {

        }
    }

    public class TanhFunc : UnaryFunction
    {
        public TanhFunc(Function function) : base(function, Math.Tanh, y => 1.0 - y * y)
        {

        }
    }

    public class TransposeFunc : Function
    {
        public TransposeFunc(Function function)
        {
            this.function = function;
        }

        readonly Function function;

        public override void Backward(NDarray<double> dY)
        {
            Grad = ND.Add(Grad, dY.T);
            function.Backward(Grad);
        }

        public override void Forward()
        {
            function.Forward();
            Y = function.Y.T;

            if (Grad == null)
                Grad = new NDarray<double>(function.Y.Shape);
        }

        public override void PropResetGrad()
        {
            ResetGrad();
            function.PropResetGrad();
        }
    }

    public class AddFunc : Function
    {
        public AddFunc(Function left, Function right)
        {
            this.left = left;
            this.right = right;
        }

        readonly Function left, right;

        public override void Backward(NDarray<double> dY)
        {
            left.Grad = ND.AddBCleft(left.Grad, dY);
            right.Grad = ND.AddBCleft(right.Grad, dY);
            left.Backward(left.Grad);
            right.Backward(right.Grad);
        }

        public override void Forward()
        {
            left.Forward();
            right.Forward();
            Y = ND.Add(left.Y, right.Y);

            if (left.Grad == null) left.Grad = new NDarray<double>(left.Y.Shape);
            if (right.Grad == null) right.Grad = new NDarray<double>(right.Y.Shape);
        }

        public override void PropResetGrad()
        {
            ResetGrad();
            left.PropResetGrad();
            right.PropResetGrad();
        }
    }

    public class MulFunc : Function
    {
        public MulFunc(Function left, Function right)
        {
            this.left = left;
            this.right = right;
        }

        readonly Function left, right;
        NDarray<double> Yl, Yr;

        public override void Forward()
        {
            left.Forward();
            right.Forward();
            Yl = left.Y;
            Yr = right.Y;
            Y = ND.Mul(Yl, Yr);

            if (left.Grad == null) left.Grad = new NDarray<double>(left.Y.Shape);
            if (right.Grad == null) right.Grad = new NDarray<double>(right.Y.Shape);
        }

        public override void Backward(NDarray<double> dY)
        {
            left.Grad = ND.AddBCleft(left.Grad, ND.Mul(dY, Yr));
            right.Grad = ND.AddBCleft(right.Grad, ND.Mul(Yl, dY));
            left.Backward(left.Grad);
            right.Backward(right.Grad);
        }

        public override void PropResetGrad()
        {
            ResetGrad();
            left.PropResetGrad();
            right.PropResetGrad();
        }
    }

    public class DotFunc : Function
    {
        public DotFunc(Function left, Function right)
        {
            this.left = left;
            this.right = right;
        }

        readonly Function left, right;
        NDarray<double> Yl, Yr;

        public override void Forward()
        {
            left.Forward();
            right.Forward();
            Yl = left.Y;
            Yr = right.Y;
            Y = ND.Dot(Yl, Yr);

            if (left.Grad == null) left.Grad = new NDarray<double>(left.Y.Shape);
            if (right.Grad == null) right.Grad = new NDarray<double>(right.Y.Shape);
        }

        public override void Backward(NDarray<double> dY)
        {
            left.Grad = ND.AddBCleft(left.Grad, ND.Dot(dY, Yr.T));
            right.Grad = ND.AddBCleft(right.Grad, ND.Dot(Yl.T, dY));
            left.Backward(left.Grad);
            right.Backward(right.Grad);
        }

        public override void PropResetGrad()
        {
            ResetGrad();
            left.PropResetGrad();
            right.PropResetGrad();
        }
    }

    public class MSELoss
    {
        public double Loss(NDarray<double> y, NDarray<double> p) => ND.Sq(ND.Sub(p, y)).ApplyFunc(x => x * 0.5).Data.Average();

        public NDarray<double> Grad(NDarray<double> y, NDarray<double> p) => ND.Sub(p, y);
        public void Backward(NDarray<double> y, Function function)
        {
            var grad = Grad(y, function.Y);
            function.Backward(grad);
        }

        public void Backward(NDarray<double> y, Layer layer)
        {
            var grad = Grad(y, layer.Y);
            layer.Backward(grad);
        }
    }

    public abstract class Layer
    {
        public int InputNodes, OutputNodes;
        public abstract void UpdateWeightsSGD(double lr);
        public Function Function;

        public void Forward() => Function.Forward();
        public void Backward(NDarray<double> dY) => Function.Backward(dY);
        public void ResetGradient() => Function.PropResetGrad();

        public NDarray<double> Y => Function.Y;
    }

    public class InputLayer : Layer
    {
        public InputLayer(int inNodes)
        {
            InputNodes = OutputNodes = inNodes;
            Function = new Variable("inputs", ND.Zeros<double>(1));
        }

        public void SetValue(NDarray<double> X)
        {
            (Function as Variable).SetValue(X);
        }

        public override void UpdateWeightsSGD(double lr)
        {

        }
    }

    public class SigmoidLayer : Layer
    {
        private readonly Layer layer;

        public SigmoidLayer(Layer layer)
        {
            this.layer = layer;
            InputNodes = OutputNodes = layer.OutputNodes;
            Function = new SigFunc(layer.Function);
        }

        public override void UpdateWeightsSGD(double lr)
        {
            layer.UpdateWeightsSGD(lr);
        }
    }

    public class TanhLayer : Layer
    {
        private readonly Layer layer;

        public TanhLayer(Layer layer)
        {
            this.layer = layer;
            InputNodes = OutputNodes = layer.OutputNodes;
            Function = new TanhFunc(layer.Function);
        }

        public override void UpdateWeightsSGD(double lr)
        {
            layer.UpdateWeightsSGD(lr);
        }
    }

    public class DenseLayer : Layer
    {
        private readonly Layer layer;

        public DenseLayer(Layer layer, int outNodes)
        {
            this.layer = layer;
            InputNodes = layer.OutputNodes;
            OutputNodes = outNodes;

            double std = 2.0 / Math.Sqrt(InputNodes);
            weights = new Variable("weights", ND.Uniform(-std, std, InputNodes, OutputNodes));
            biases = new Variable("biases", ND.Zeros<double>(1, OutputNodes));

            Function = new AddFunc(new DotFunc(layer.Function, weights), biases);
        }

        readonly Variable weights, biases;

        public override void UpdateWeightsSGD(double lr)
        {
            weights.UpdateSGD(lr);
            biases.UpdateSGD(lr);
            layer.UpdateWeightsSGD(lr);
        }
    }

    public class Chain
    {
        InputLayer inputLayer;
        Layer layer;
        MSELoss MSELoss = new MSELoss();

        public Chain(int inNodes)
        {
            inputLayer = new InputLayer(inNodes);
            layer = inputLayer;
        }

        public Chain AddSigmoidActivation()
        {
            layer = new SigmoidLayer(layer);
            return this;
        }

        public Chain AddTanhActivation()
        {
            layer = new TanhLayer(layer);
            return this;
        }

        public Chain AddDenseLayer(int outNodes)
        {
            layer = new DenseLayer(layer, outNodes);
            return this;
        }

        public void Forward(NDarray<double> X = null)
        {
            if (X != null)
                inputLayer.SetValue(X);

            layer.Forward();
        }

        public double Loss(NDarray<double> y) => MSELoss.Loss(y, layer.Y);
        public void Backward(NDarray<double> y)
        {
            MSELoss.Backward(y, layer);
        }

        public void UpdateWeightsSGD(double lr) => layer.UpdateWeightsSGD(lr);
        public void ResetGradient() => layer.ResetGradient();

        public NDarray<double> Prediction(NDarray<double> X)
        {
            Forward(X);
            return layer.Y;
        }
    }
}