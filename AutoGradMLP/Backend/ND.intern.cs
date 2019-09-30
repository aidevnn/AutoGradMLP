using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoGradMLP
{
    public static partial class ND
    {
        public static NDarray<U> Reshape<U>(NDarray<U> inArr, int[] shape)
        {
            var nshape = Utils.PrepareReshape(inArr.Count, shape);
            var outArr = new NDarray<U>(data: inArr.Data.ToArray(), shape: nshape);
            return outArr;
        }

        public static NDarray<U> Transpose<U>(NDarray<U> inArr, int[] table)
        {
            if (table == null || table.Length == 0)
                table = Utils.PrepareTranspose(inArr.Shape.Length);

            var nshape = Utils.DoTranspose(inArr.Shape, table);
            var nstrides = Utils.DoTranspose(inArr.Strides, table);
            var outArr = new NDarray<U>(shape: nshape);

            for (int idx = 0; idx < outArr.Count; ++idx)
            {
                int idx1 = Utils.Int2IntIndex(idx, nshape, nstrides);
                outArr.Data[idx] = inArr.Data[idx1];
            }

            return outArr;
        }

        public static NDarray<V> ElementWiseOp<U, V>(NDarray<U> left, NDarray<U> right, Func<U, U, V> func)
        {
            var nshape = Utils.BroadCastShapes(left.Shape, right.Shape);
            var outArr = new NDarray<V>(shape: nshape);

            for (int index = 0; index < outArr.Count; ++index)
            {
                Utils.Int2ArrayIndex(index, outArr.Shape, outArr.Indices);
                for (int k = outArr.Indices.Length - 1, i = left.Shape.Length - 1, j = right.Shape.Length - 1; k >= 0; --k, --i, --j)
                {
                    if (i >= 0) left.Indices[i] = outArr.Indices[k] % left.Shape[i];
                    if (j >= 0) right.Indices[j] = outArr.Indices[k] % right.Shape[j];
                }

                var v0 = left.Data[Utils.Array2IntIndex(left.Indices, left.Shape, left.Strides)];
                var v1 = right.Data[Utils.Array2IntIndex(right.Indices, right.Shape, right.Strides)];
                outArr.Data[index] = func(v0, v1);
            }

            return outArr;
        }

        public static NDarray<U> ElementWiseOpBCleft<U>(NDarray<U> left, NDarray<U> right, Func<U, U, U> func, U start)
        {
            var nshape = Utils.BroadCastShapes(left.Shape, right.Shape);
            var tmpRight = new NDarray<U>(right);

            for (int kl = left.Shape.Length - 1, kr = right.Shape.Length - 1, kn = nshape.Length - 1; kr >= 0; --kl, --kr, --kn)
            {
                var l0 = kl < 0 ? -1 : left.Shape[kl];
                var r0 = right.Shape[kr];
                var n0 = nshape[kn];
                if (l0 != n0 && r0 == n0)
                    tmpRight = AxisOps(tmpRight, kr, true, func, start);
            }

            return ElementWiseOp(left, tmpRight, func);
        }

        public static NDarray<U> Add2<U>(NDarray<U> a, NDarray<U> b) => ElementWiseOp2(a, b, NDarray<U>.OpsT.Add);
        public static NDarray<V> ElementWiseOp2<U, V>(NDarray<U> left, NDarray<U> right, Func<U, U, V> func)
        {
            var (nshape, lshape, rshape) = Utils.BroadCastShapes2(left.Shape, right.Shape);
            var outArr = new NDarray<V>(shape: nshape);

            var l0 = Utils.ArrMul(lshape) == 1 ? left : Tile2(left, lshape);
            var r0 = Utils.ArrMul(rshape) == 1 ? right : Tile2(right, rshape);

            for (int k = 0; k < outArr.Count; ++k)
                outArr.Data[k] = func(l0.Data[k], r0.Data[k]);

            return outArr;
        }

        public static NDarray<U> AxisOps<U>(NDarray<U> inArr, int axis, bool keepdims, Func<U, U, U> func, U start, bool mean = false)
        {
            var shape = Utils.PrepareAxisOps(inArr.Shape, axis, keepdims);
            var outArr = new NDarray<U>(shape: shape);

            if (axis == -1)
            {
                U res = start;
                U nb = mean ? NDarray<U>.OpsT.Cast(inArr.Count) : NDarray<U>.OpsT.One;
                for (int idx = 0; idx < inArr.Count; ++idx)
                    res = func(res, inArr.Data[idx]);

                res = NDarray<U>.OpsT.Div(res, nb);
                outArr.Data[0] = res;
            }
            else
            {
                var NShape = Utils.PrepareAxisOps(inArr.Shape, axis, true);
                var NIndices = new int[NShape.Length];
                U nb = mean ? NDarray<U>.OpsT.Cast(inArr.Shape[axis]) : NDarray<U>.OpsT.One;

                for (int idx0 = 0; idx0 < outArr.Count; ++idx0)
                {
                    U res = start;
                    Utils.Int2ArrayIndex(idx0, NShape, NIndices);

                    for (int k = 0; k < inArr.Shape[axis]; ++k)
                    {
                        NIndices[axis] = k;
                        int idx1 = Utils.Array2IntIndex(NIndices, inArr.Shape, inArr.Strides);
                        res = func(res, inArr.Data[idx1]);
                    }

                    outArr.Data[idx0] = NDarray<U>.OpsT.Div(res, nb);
                }
            }

            return outArr;
        }

        public static NDarray<int> ArgMinMax<U>(NDarray<U> inArr, int axis, Func<U, U, U> func, U tmp)
        {
            axis = (axis + inArr.Shape.Length) % inArr.Shape.Length;
            (int[] ishape, int[] nshape) = Utils.PrepareArgMinmax(inArr.Shape, axis);
            int[] indices = new int[ishape.Length];
            var outArr = new NDarray<int>(nshape);

            int nb = inArr.Shape[axis];
            for (int idx = 0; idx < outArr.Count; ++idx)
            {
                U valBest = tmp;
                int idxBest = 0;
                Utils.Int2ArrayIndex(idx, ishape, indices);
                for (int k = 0; k < nb; ++k)
                {
                    indices[axis] = k;
                    var v = inArr.Data[Utils.Array2IntIndex(indices, inArr.Shape, inArr.Strides)];
                    var v0 = func(v, valBest);
                    if (!valBest.Equals(v0))
                    {
                        idxBest = k;
                        valBest = v0;
                    }
                }

                outArr.Data[idx] = idxBest;
            }

            return outArr;
        }

        public static NDarray<U> Dot<U>(NDarray<U> left, NDarray<U> right)
        {
            (int[] lshape, int[] rshape, int[] shape, int[] idxInfos) = Utils.PrepareDot(left.Shape, right.Shape);
            var outArr = new NDarray<U>(shape: shape);

            var leftArr = left.Shape.Length == lshape.Length ? left : new NDarray<U>(data: left.Data, shape: lshape);
            var rightArr = right.Shape.Length == rshape.Length ? right : new NDarray<U>(data: right.Data, shape: rshape);

            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            int[] indices = new int[shape.Length];
            for (int idx = 0; idx < outArr.Count; ++idx)
            {
                U sum = NDarray<U>.OpsT.Zero;
                Utils.Int2ArrayIndex(idx, shape, indices);

                for (int k = 0; k < shape.Length; ++k)
                {
                    if (k < length0 - 1) leftArr.Indices[idxInfos[k]] = indices[k];
                    else rightArr.Indices[idxInfos[k]] = indices[k];
                }

                for (int i = 0; i < piv; ++i)
                {
                    leftArr.Indices[length0 - 1] = rightArr.Indices[length1 - 2] = i;

                    int idxl = Utils.Array2IntIndex(leftArr.Indices, leftArr.Shape, leftArr.Strides);
                    int idxr = Utils.Array2IntIndex(rightArr.Indices, rightArr.Shape, rightArr.Strides);
                    var prod = NDarray<U>.OpsT.Mul(leftArr.Data[idxl], rightArr.Data[idxr]);
                    sum = NDarray<U>.OpsT.Add(sum, prod);
                }

                outArr.Data[idx] = sum;
            }

            return outArr;
        }

        public static NDarray<U> Dot2<U>(NDarray<U> left, NDarray<U> right)
        {
            (int[] lshape, int[] rshape, int[] shape, int[] table) = Utils.PrepareDot2(left.Shape, right.Shape);

            var leftArr = left.Reshape(lshape);
            var rightArr = right.Transpose(table).Reshape(rshape);

            int m = leftArr.Shape[0];
            int p = leftArr.Shape[1];
            int n = rightArr.Shape[1];
            var outArr = new NDarray<U>(new int[] { m, n });

            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    var sum = NDarray<U>.OpsT.Zero;
                    for (int k = 0; k < p; ++k)
                        sum = NDarray<U>.OpsT.Add(sum, NDarray<U>.OpsT.Mul(leftArr.Data[i * p + k], rightArr.Data[k * n + j]));

                    outArr.Data[i * n + j] = sum;
                }
            }

            return outArr.Reshape(shape);
        }

        public static NDarray<U> Tile<U>(NDarray<U> nDarray, params int[] rep)
        {
            var nshape = Utils.PrepareTile(nDarray.Shape, rep);
            var nd0 = new NDarray<U>(shape: nshape);

            for (int idx = 0; idx < nd0.Count; ++idx)
            {
                Utils.Int2ArrayIndex(idx, nd0.Shape, nd0.Indices);
                for (int i = nDarray.Shape.Length - 1, j = nd0.Shape.Length - 1; i >= 0 && j >= 0; --i, --j)
                    nDarray.Indices[i] = nd0.Indices[j] % nDarray.Shape[i];

                int idx0 = Utils.Array2IntIndex(nDarray.Indices, nDarray.Shape, nDarray.Strides);
                nd0.Data[idx] = nDarray.Data[idx0];
            }

            return nd0;
        }

        public static NDarray<U> Tile2<U>(NDarray<U> nDarray, params int[] repeat)
        {
            var nshape = Utils.PrepareTile(nDarray.Shape, repeat);

            int rep = 0, stride = 1, k0 = 0, k1 = 0, ct = 0;
            U[] data = nDarray.Data;
            for (k0 = nDarray.Shape.Length - 1, k1 = repeat.Length - 1; k1 >= 0; --k0, --k1)
            {
                int shape0 = k0 < 0 ? 1 : nDarray.Shape[k0];
                rep = repeat[k1];
                stride *= shape0;
                U[] tmpData = new U[data.Length * rep];
                for (int idx0 = 0; idx0 < data.Length; ++idx0)
                {
                    int start = (idx0 / stride) * rep * stride + idx0 % stride;
                    for (int idx1 = 0; idx1 < rep; ++idx1)
                    {
                        ct++;
                        tmpData[start + idx1 * stride] = data[idx0];
                    }
                }
                data = tmpData;
                stride *= rep;
            }

            //Console.WriteLine($"Size:{Utils.ArrMul(nshape)} Counter:{ct}");
            return new NDarray<U>(data: data, shape: nshape);
        }
    }
}
