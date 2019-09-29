using System;
namespace AutoGradMLP
{

    public abstract class Operations<U>
    {
        public U One, Zero, Epsilon, Minvalue, Maxvalue;
        public string dtype;
        public abstract U Neg(U a);
        public abstract U Add(U a, U b);
        public abstract U Sub(U a, U b);
        public abstract U Mul(U a, U b);
        public abstract U Div(U a, U b);

        public abstract U Abs(U x);
        public abstract U Exp(U x);
        public U Inv(U x) => Div(One, x);
        public abstract U Log(U x);
        public U Sq(U x) => Mul(x, x);
        public abstract U Sqrt(U x);
        public abstract U Tanh(U x);
        public U DTanh(U x) => Sub(One, Sq(x));
        public U Sigmoid(U x) => Div(One, Add(One, Exp(Neg(x))));
        public U DSigmoid(U x) => Mul(x, Sub(One, x));
        public abstract U Round(U x, int d = 0);

        public abstract U Min(U x, U y);
        public abstract U Max(U x, U y);
        public abstract U Rand(U min, U max);

        public abstract double Eq(U x, U y);
        public abstract double Neq(U x, U y);

        public abstract double Gt(U x, U y);
        public abstract double Lt(U x, U y);

        public abstract double Gte(U x, U y);
        public abstract double Lte(U x, U y);

        public U Clamp(U x, double min, double max) => Max(Cast(min), Min(x, Cast(max)));
        public U Cast<V>(V x) => (U)Convert.ChangeType(x, typeof(U));
    }

    public class OpsInt : Operations<int>
    {
        public OpsInt() { One = 1; Zero = 0; Epsilon = 0; dtype = "np.int64"; Minvalue = int.MinValue; Maxvalue = int.MaxValue; }
        public override int Neg(int a) => -a;
        public override int Add(int a, int b) => a + b;
        public override int Sub(int a, int b) => a - b;
        public override int Mul(int a, int b) => a * b;
        public override int Div(int a, int b) => a / b;

        public override int Abs(int x) => Math.Abs(x);
        public override int Exp(int x) => throw new NotImplementedException();
        public override int Log(int x) => throw new NotImplementedException();
        public override int Sqrt(int x) => (int)Math.Sqrt(x);
        public override int Tanh(int x) => throw new NotImplementedException();
        public override int Round(int x, int d = 0) => x;

        public override int Min(int x, int y) => Math.Min(x, y);
        public override int Max(int x, int y) => Math.Max(x, y);
        public override int Rand(int min, int max) => Utils.GetRandom.Next(min, max);

        public override double Eq(int x, int y) => x == y ? 1 : 0;
        public override double Neq(int x, int y) => x != y ? 1 : 0;
        public override double Gt(int x, int y) => x > y ? 1 : 0;
        public override double Gte(int x, int y) => x >= y ? 1 : 0;
        public override double Lt(int x, int y) => x < y ? 1 : 0;
        public override double Lte(int x, int y) => x <= y ? 1 : 0;

    }

    public class OpsFloat : Operations<float>
    {
        public OpsFloat() { One = 1; Zero = 0; Epsilon = 1e-6f; dtype = "np.float32"; Minvalue = float.MinValue; Maxvalue = float.MaxValue; }
        public override float Neg(float a) => -a;
        public override float Add(float a, float b) => a + b;
        public override float Sub(float a, float b) => a - b;
        public override float Mul(float a, float b) => a * b;
        public override float Div(float a, float b) => a / b;

        public override float Abs(float x) => Math.Abs(x);
        public override float Exp(float x) => (float)Math.Exp(x);
        public override float Log(float x) => (float)Math.Log(x);
        public override float Sqrt(float x) => (float)Math.Sqrt(x);
        public override float Tanh(float x) => (float)Math.Tanh(x);
        public override float Round(float x, int d = 0) => (float)Math.Round(x, d);

        public override float Min(float x, float y) => Math.Min(x, y);
        public override float Max(float x, float y) => Math.Max(x, y);
        public override float Rand(float min, float max) => (float)(min + (max - min) * Utils.GetRandom.NextDouble());

        public override double Eq(float x, float y) => Math.Abs(x - y) <= Epsilon ? 1 : 0;
        public override double Neq(float x, float y) => Math.Abs(x - y) > Epsilon ? 1 : 0;
        public override double Gt(float x, float y) => x > y ? 1 : 0;
        public override double Gte(float x, float y) => x >= y ? 1 : 0;
        public override double Lt(float x, float y) => x < y ? 1 : 0;
        public override double Lte(float x, float y) => x <= y ? 1 : 0;
    }

    public class OpsDouble : Operations<double>
    {
        public OpsDouble() { One = 1; Zero = 0; Epsilon = 1e-6; dtype = "np.float64"; Minvalue = double.MinValue; Maxvalue = double.MaxValue; }
        public override double Neg(double a) => -a;
        public override double Add(double a, double b) => a + b;
        public override double Sub(double a, double b) => a - b;
        public override double Mul(double a, double b) => a * b;
        public override double Div(double a, double b) => a / b;

        public override double Abs(double x) => Math.Abs(x);
        public override double Exp(double x) => Math.Exp(x);
        public override double Log(double x) => Math.Log(x);
        public override double Sqrt(double x) => Math.Sqrt(x);
        public override double Tanh(double x) => Math.Tanh(x);
        public override double Round(double x, int d = 0) => Math.Round(x, d);

        public override double Min(double x, double y) => Math.Min(x, y);
        public override double Max(double x, double y) => Math.Max(x, y);
        public override double Rand(double min, double max) => min + (max - min) * Utils.GetRandom.NextDouble();

        public override double Eq(double x, double y) => Math.Abs(x - y) <= Epsilon ? 1.0 : 0.0;
        public override double Neq(double x, double y) => Math.Abs(x - y) > Epsilon ? 1.0 : 0.0;
        public override double Gt(double x, double y) => x > y ? 1.0 : 0.0;
        public override double Gte(double x, double y) => x >= y ? 1.0 : 0.0;
        public override double Lt(double x, double y) => x < y ? 1.0 : 0.0;
        public override double Lte(double x, double y) => x <= y ? 1.0 : 0.0;
    }
}
