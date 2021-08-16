# Day 1

## *Tensorflow*



**TensorFlow** is an open-source end-to-end platform for creating Machine Learning applications. It is a symbolic math library that uses dataflow and differentiable programming to perform various tasks focused on training and inference of deep neural networks. It allows developers to create machine learning applications using various tools, libraries, and community resources.



[what is a tensor](https://www.youtube.com/watch?v=f5liqUk0ZTw)


1. importing tensorflow

```python
import tensorflow as tf
```

2. checking version 

```python
print(tf.__version__)
```
3. creating **constant** tensors

```python
scalar = tf.constant(7)  # rank 0
scalar.ndim
```

```python
vector = tf.constant([2,3])  # rank 1
vector.ndim
```

```python
matrix = tf.constant([[3,1],[7,8]])  # rank 2
matrix.ndim
```

```python
tensor = tf.constant([[[1,2,3],   # rank 3
                       [4,5,6]],
                       [[7,8,9],
                        [10,11,12]],
                        [[13,14,15],
                         [16,17,18]]])
tensor
```

- Rank - 0 
  - Scalar
  - We need no index to find it
- Rank - 1
  - Vector
  - We need only one index to find the element
- Rank - 2
  - Matrix
  - We need to indices to find an element
- So on.....


4.  Creating tensor and giving own data type

```python
mat = tf.constant([[10.,5.],[7.,8.]],dtype = tf.float16)
mat
```

> <tf.Tensor: shape=(2, 2), dtype=float16, numpy=array([[10.,  5.],
>        																				[ 7.,  8.]], dtype=float16)>

5. Creating **changeable** tensor

```python
changeable_tensor = tf.Variable([10,7])
changeable_tensor
```

```pytho
changeable_tensor[0].assign(2)
changeable_tensor
```

