using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoGradMLP
{

    public static class Utils
    {
        public const int DbgNo = 0, DbgLvl1 = 0b1, DbgLvl2 = 0b10, DbgLvlAll = 0b11;
        public static int DebugNumpy = DbgNo;
        public static bool IsDebugNo => DebugNumpy == DbgNo;
        public static bool IsDebugLvl1 => (DebugNumpy & DbgLvl1) == DbgLvl1;
        public static bool IsDebugLvl2 => (DebugNumpy & DbgLvl2) == DbgLvl2;

        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string fmt = "{0}") =>
            string.Join(sep, ts.Select(a => string.Format(fmt, a)));

        public static Random random = new Random(123);
        //public static Random random = new Random((int)DateTime.Now.Ticks);
        public static Random GetRandom => random;

        public static int ArrMul(int[] shape, int start = 0) => shape.Skip(start).Aggregate(1, (a, i) => a * i);
        public static int[] Shape2Strides(int[] shape) => Enumerable.Range(0, shape.Length).Select(i => ArrMul(shape, i + 1)).ToArray();

        public static int Array2IntIndex(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                idx += v * strides[k];
            }

            return idx;
        }

        public static int Array2IntBCIndex(int[] args, int[] shape, int[] strides)
        {
            int idx = 0;
            for (int k = shape.Length - 1, k0 = args.Length - 1; k >= 0; --k, --k0)
            {
                var v = args[k0] % shape[k];
                idx += v * strides[k];
            }

            return idx;
        }

        public static void Int2ArrayIndex(int idx, int[] shape, int[] indices)
        {

            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                indices[k] = idx % sk;
                idx = idx / sk;
            }
        }

        public static int Int2IntIndex(int idx0, int[] shape, int[] strides)
        {
            int idx1 = 0;
            for (int k = shape.Length - 1; k >= 0; --k)
            {
                var sk = shape[k];
                idx1 += strides[k] * (idx0 % sk);
                idx0 = idx0 / sk;
            }

            return idx1;
        }

        public static int[] PrepareReshape(int[] baseShape, int[] shape) => PrepareReshape(ArrMul(baseShape), shape);

        public static int[] PrepareReshape(int dim0, int[] shape)
        {
            int mone = shape.Count(i => i == -1);
            if (mone > 1)
                throw new ArgumentException("Can only specify one unknown dimension");

            if (mone == 1)
            {
                int idx = shape.ToList().FindIndex(i => i == -1);
                shape[idx] = 1;
                var dim2 = ArrMul(shape);
                shape[idx] = dim0 / dim2;
            }

            var dim1 = ArrMul(shape);

            if (dim0 != dim1)
                throw new ArgumentException($"cannot reshape array of size {dim0} into shape ({shape.Glue()})");

            return shape;
        }

        public static int[] PrepareTranspose(int rank) => Enumerable.Range(0, rank).Reverse().ToArray();
        public static int[] DoTranspose(int[] arr, int[] table) => Enumerable.Range(0, arr.Length).Select(i => arr[table[i]]).ToArray();

        public static int[] BroadCastShapes(int[] shape0, int[] shape1)
        {
            int sLength0 = shape0.Length;
            int sLength1 = shape1.Length;
            int mLength = Math.Max(sLength0, sLength1);

            int[] nshape = new int[mLength];
            for (int k = mLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : shape0[i];
                int idx1 = j < 0 ? 1 : shape1[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast ({shape0.Glue()}) with ({shape1.Glue()})");

                nshape[k] = Math.Max(idx0, idx1);
            }

            return nshape;
        }

        public static (int[], int[], int[]) BroadCastShapes2(int[] shape0, int[] shape1)
        {
            var nshape = BroadCastShapes(shape0, shape1);
            int[] l = nshape.ToArray();
            int[] r = nshape.ToArray();

            for (int k0 = shape0.Length - 1, k1 = shape1.Length - 1, k2 = nshape.Length - 1; k0 >= 0 || k1 >= 0 || k2 >= 0; --k0, --k1, --k2)
            {
                if (k0 >= 0)
                    l[k2] = nshape[k2] / shape0[k0];

                if (k1 >= 0)
                    r[k2] = nshape[k2] / shape1[k1];
            }

            return (nshape, l, r);
        }

        public static int[] PrepareAxisOps(int[] shape, int axis, bool keepdims)
        {
            List<int> nshape = new List<int>(shape);

            if (axis == -1)
                nshape = Enumerable.Repeat(1, shape.Length).ToList();
            else
                nshape[axis] = 1;

            if (!keepdims)
            {
                if (axis == -1)
                    nshape = new List<int>() { 1 };
                else
                    nshape.RemoveAt(axis);
            }

            return nshape.ToArray();
        }

        public static (int[], int[]) PrepareArgMinmax(int[] shape, int axis)
        {
            if (axis < 0 || axis >= shape.Length)
                throw new ArgumentException("Bad axis for ArgMinMax");

            var ishape = shape.ToArray();
            ishape[axis] = 1;
            var nshape = shape.Select((v, i) => (v, i)).Where(t => t.i != axis).Select(t => t.v).ToArray();
            return (ishape, nshape);
        }

        public static (int[], int[], int[], int[]) PrepareDot(int[] shape0, int[] shape1)
        {
            bool head = false, tail = false;
            int[] nshape;
            int[] lshape, rshape, idxInfos;

            if (head = shape0.Length == 1)
                lshape = new int[] { 1, shape0[0] };
            else
                lshape = shape0.ToArray();

            if (tail = shape1.Length == 1)
                rshape = new int[] { shape1[0], 1 };
            else
                rshape = shape1.ToArray();


            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            if (piv != rshape[length1 - 2])
                throw new ArgumentException($"Cannot multiply ({shape0.Glue()}) and ({shape1.Glue()})");

            nshape = new int[length0 + length1 - 2];
            idxInfos = new int[length0 + length1 - 2];

            for (int k = 0, k0 = 0; k < length0 + length1; ++k)
            {
                if (k == length0 - 1 || k == length0 + length1 - 2) continue;
                if (k < length0 - 1) nshape[k] = lshape[idxInfos[k] = k];
                else nshape[k0] = rshape[idxInfos[k0] = k - length0];
                ++k0;
            }

            return (lshape, rshape, nshape, idxInfos);
        }

        public static (int[], int[], int[], int[]) PrepareDot2(int[] shape0, int[] shape1)
        {
            int[] nshape;
            int[] lshape, rshape;

            if (shape0.Length == 1)
                lshape = new int[] { 1, shape0[0] };
            else
                lshape = shape0.ToArray();

            if (shape1.Length == 1)
                rshape = new int[] { shape1[0], 1 };
            else
                rshape = shape1.ToArray();

            int length0 = lshape.Length;
            int length1 = rshape.Length;
            int piv = lshape.Last();

            var lshape0 = PrepareReshape(lshape, new int[] { -1, piv });
            var rshape0 = PrepareReshape(rshape, new int[] { piv, -1 });

            int[] table = Enumerable.Range(-1, length1).ToArray();
            table[0] = length1 - 2;
            table[length1 - 1] = length1 - 1;

            if (piv != rshape[length1 - 2])
                throw new ArgumentException($"Cannot multiply ({shape0.Glue()}) and ({shape1.Glue()})");

            nshape = new int[length0 + length1 - 2];

            for (int k = 0, k0 = 0; k < length0 + length1; ++k)
            {
                if (k == length0 - 1 || k == length0 + length1 - 2) continue;
                if (k < length0 - 1) nshape[k] = lshape[k];
                else nshape[k0] = rshape[k - length0];
                ++k0;
            }

            return (lshape0, rshape0, nshape, table);
        }

        public static int[] PrepareTile(int[] shape, int[] rep)
        {
            if (rep.Any(i => i <= 0))
                throw new ArgumentException("Repetition must be greater than 0");

            var nshape = shape.Length >= rep.Length ? shape.ToArray() : rep.ToArray();
            for (int i = rep.Length - 1, j = shape.Length - 1; i >= 0 && j >= 0; --i, --j)
                nshape[Math.Max(i, j)] = shape[j] * rep[i];

            return nshape;
        }

    }
}
