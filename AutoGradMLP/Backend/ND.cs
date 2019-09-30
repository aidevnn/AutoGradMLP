using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoGradMLP
{

    public static partial class ND
    {
        public static NDarray<U> Zeros<U>(params int[] shape) => new NDarray<U>(shape);

        public static NDarray<U> Ones<U>(params int[] shape)
        {
            int count = Utils.ArrMul(shape);
            U[] data = Enumerable.Range(0, count).Select(i => NDarray<U>.OpsT.One).ToArray();
            return new NDarray<U>(data: data, shape: shape);
        }

        public static NDarray<U> CreateNDarray<U>(U[] data, params int[] shape)
            => new NDarray<U>(data: data.ToArray(), shape: Utils.PrepareReshape(data.Length, shape));

        public static NDarray<U> CreateNDarray<U>(U[,] data)
        {
            int dim0 = data.GetLength(0);
            int dim1 = data.GetLength(1);

            U[] data0 = new U[dim0 * dim1];
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j)
                    data0[i * dim1 + j] = data[i, j];

            return CreateNDarray(data0, new int[] { dim0, dim1 });
        }

        public static NDarray<U> Uniform<U>(U min, U max, params int[] shape)
        {
            int count = Utils.ArrMul(shape);
            U[] data = Enumerable.Range(0, count).Select(i => NDarray<U>.OpsT.Rand(min, max)).ToArray();
            return new NDarray<U>(data: data, shape: shape);
        }

        public static NDarray<int> Arange(int start, int length, int step = 1)
        {
            int[] data = Enumerable.Range(0, length).Select(i => start + i * step).ToArray();
            return new NDarray<int>(data: data, shape: new int[] { length });
        }

        public static NDarray<int> Arange(int length) => Arange(0, length, 1);

    }
}
