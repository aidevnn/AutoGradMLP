using System;
namespace AutoGradMLP
{

    public static partial class ND
    {
        public static NDarray<U> Abs<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Abs);
        public static NDarray<U> Exp<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Exp);
        public static NDarray<U> Inv<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Inv);
        public static NDarray<U> Log<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Log);
        public static NDarray<U> Sq<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sq);
        public static NDarray<U> Sqrt<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sqrt);
        public static NDarray<U> Sigmoid<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Sigmoid);
        public static NDarray<U> Tanh<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Tanh);
        public static NDarray<U> DSigmoid<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.DSigmoid);
        public static NDarray<U> DTanh<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.DTanh);
        public static NDarray<U> Neg<U>(NDarray<U> x) => x.ApplyFunc(NDarray<U>.OpsT.Neg);
        public static NDarray<U> Round<U>(NDarray<U> x, int dec = 0) => x.ApplyFunc(a => NDarray<U>.OpsT.Round(a, dec));
        public static NDarray<U> Clamp<U>(NDarray<U> x, double min, double max) => x.ApplyFunc(a => NDarray<U>.OpsT.Clamp(a, min, max));

        public static NDarray<U> Add<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Add);
        public static NDarray<U> Sub<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Sub);
        public static NDarray<U> Mul<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Mul);
        public static NDarray<U> Div<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Div);
        public static NDarray<U> Min<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Min);
        public static NDarray<U> Max<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Max);

        public static NDarray<U> AddBCleft<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOpBCleft(a, b, NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero);
        public static NDarray<U> SubBCleft<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOpBCleft(a, Neg(b), NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero);
        public static NDarray<U> MulBCleft<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOpBCleft(a, b, NDarray<U>.OpsT.Mul, NDarray<U>.OpsT.One);
        public static NDarray<U> DivBCleft<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOpBCleft(a, Inv(b), NDarray<U>.OpsT.Mul, NDarray<U>.OpsT.One);

        public static NDarray<double> Eq<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Eq);
        public static NDarray<double> Neq<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Neq);
        public static NDarray<double> Lt<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Lt);
        public static NDarray<double> Lte<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Lte);
        public static NDarray<double> Gt<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Gt);
        public static NDarray<double> Gte<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp(a, b, NDarray<U>.OpsT.Gte);

        public static NDarray<U> Sum<U>(NDarray<U> a, int axis = -1, bool keepdims = false) => AxisOps(a, axis, keepdims, NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero);
        public static NDarray<U> Prod<U>(NDarray<U> a, int axis = -1, bool keepdims = false) => AxisOps(a, axis, keepdims, NDarray<U>.OpsT.Mul, NDarray<U>.OpsT.One);
        public static NDarray<U> Mean<U>(NDarray<U> a, int axis = -1, bool keepdims = false) => AxisOps(a, axis, keepdims, NDarray<U>.OpsT.Add, NDarray<U>.OpsT.Zero, true);

        public static NDarray<int> ArgMin<U>(NDarray<U> a, int axis) => ArgMinMax(a, axis, NDarray<U>.OpsT.Min, NDarray<U>.OpsT.Maxvalue);
        public static NDarray<int> ArgMax<U>(NDarray<U> a, int axis) => ArgMinMax(a, axis, NDarray<U>.OpsT.Max, NDarray<U>.OpsT.Minvalue);
    }
}
