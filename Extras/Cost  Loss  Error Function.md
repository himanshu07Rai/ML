# Cost / Loss / Error Function

- Minimize this cost
- Finding where the derivative ( slope ) is zero, is equivalent to finding a min / max. 
- ![image](C:\Users\ACER\AppData\Roaming\Typora\typora-user-images\image-20210710145753578.png)
- The multidimensional equivalent of derivative is the **gradient**
- To solve for a model parameters ( weights ), we find the gradient and set it to zero.
-  Tensorflow uses automatic differentiation 
  - It automatically finds the gradients of loss wrt all the model weights.
  - Uses the gradients to train model.