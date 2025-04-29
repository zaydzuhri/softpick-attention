# softpick-attention

From the paper:
Softpick: No Attention Sinks, No Massive Activations with Rectified Softmax

In this repository are implementations of attention with the softpick function. Both a naive implementation and a FlashAttention-2 kernel modification are included. We do NOT recommend using the triton kernels here directly, since they are taken from the flash-linear-attention repository and are untested outside of that context. The code here is meant as a reference for those who want to implement softpick in their own kernels.

For the training code that we used in the paper, see:
https://github.com/zaydzuhri/flame/tree/softpick-attention