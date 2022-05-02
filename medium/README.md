# Implementing SegFormer in PyTorch
*An fast, efficient and lightweight model for image segmentation*


Hello There!! Today we'll see how to implement SegFormer in PyTorch proposed in [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203).

Code is here, an interactive version of this article can be downloaded from here.

Let's get started!

The paper proposes a new transformer-based model to tackle image segmentation. Even if "transformer" is nowadays a buzzword, and the model itself only has the basic attention mechanism. This model has two main advantages, first *SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale
features*. Then, *It does not need positional encoding, thereby avoiding the interpolation of
positional codes which leads to decreased performance when the testing resolution differs from training*. 

Funny enough, we are going backward in research, these two advantages are present in convnets since the beginning and we'll see that SegFormer, in the end, is just a convnet + attention.

The following picture shows SegFormer's performance against the different models/sizes on ADE20K dataset, they have **sota**.

<img src="/images/results.png" width="500px"></img>

It's better than the old good FCN-R50 and it's 2x faster. Since it has 24 fewer FLOPS I am wondering why it's only double as fast.

## Architecture

The model is a classic encoder-decoder/backbone-neck. A head is attached to predict the final segmentation mask. 

<img src="/images/architecture.png"></img>

We are going to implement it in a bottom-up approach, starting from the lowest module inside the decoder. 

**The image in the paper is wrong** 🤦, I don't understand why any reviewers pointed that out, maybe I am wrong. In the official [implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py) there isn't a first patch embedding. The overlapping patch merging block (the purple one) should be before the Self-Efficient Attention block.

This is how it should look like:

<img src="/images/architecture_fixed.png"></img>

With a little photoshop

<img src="/images/BlockCorrect.png"></img>


See the code [here](https://github.com/NVlabs/SegFormer/blob/9454025f0e74acbbc19c65cbbdf3ff8224997fe3/mmseg/models/backbones/mix_transformer.py#L318)

### Decoder

The Decoder used is called `MixVisionTransformer` (`MiT`), another `ViT` with some random stuff in the middle, we will call it `SegFormerDecoder`. Let's start with the first individual component of the block itself, `OverlapPatchMerging`.

#### OverlapPatchMerging

<img src="/images/OverlapPatchMerging.png"></img>

The `OverlapPatchMerging` block can be implemented with a convolution layer with a `stride` less than the `kernel_size`, so it overlaps different patches. It's the same thing proposed years ago when you use a `stride` greater than one to reduce the input's spatial dimension. In `SegFormer`, the conv layer is followed by a layer norm.

Since `nn.LayerNorm` in PyTorch works for tensors of shape `batch, ...., channels`, we can create a `LayerNorm2d` that first swaps the `channels` axis with the last one, then applies layer norm, and swaps it back. I'll use [`einops`](https://github.com/arogozhnikov/einops) to make the code more readable




https://gist.github.com/47d5d213a7eb92c090ba8ca77037021c

Then our `OverlapPatchMerging` is just a conv layer followed by our layer norm


https://gist.github.com/9811c1f8721f59b44f464d3f2e5718dc

#### Efficient Self Attention

<img src="/images/EfficientSelfAttention.png"></img>

We all know attention has a square complexity `O(N^2)` where `N=H*W` in our case. We can reduce `N` by a factor of `R`, the complexity becomes `O(N^2/R)`. One easy way is to flat the spatial dimension and use a linear layer. 


https://gist.github.com/0c9830f134b79383bc595a35b0faa754

We have reduced the spatial size by `r=4`, so by `2` on each dimension (`height` and `width`). If you think about it, you can use a convolution layer with a `kernel_size=r` and a `stride=r` to achieve the same effect. 

Since the attention is equal to `softmax((QK^T/scale)V)`, we need to compute `K` and `V` using the reduced tensor otherwise shape won't match. `Q \in NxC, K \in (N/R)xC, V \in (N/R)xC`, we can use PyTorch's `MultiheadAttention` to compute the attention.


https://gist.github.com/c5ef6d68704cab5930c1dd6d4b5ee0b1




    torch.Size([1, 8, 64, 64])



#### MixMLP
<img src="/images/Mix-FFN.png"></img>

The careful reader may have noticed we are not using positional encoding. SegFormer uses a `3x3` depth-wise conv. Quoting from the paper *We argue that positional encoding is not necessary for semantic segmentation. Instead, we introduce Mix-FFN which considers the effect of zero padding to leak location information*. I have no idea what it means, so we will take it for granted.

I am pretty sure it's called **Mix** because it mixes information using the `3x3` conv.

The layer is composed by a `dense layer` -> `3x3 depth-wise conv` -> `GELU` -> `dense layer`. Like in ViT, this is an inverse bottleneck layer, the information is expanded in the middle layer.


https://gist.github.com/9de911b5e189daf80dadb30feff69f3f

### Encoder (Transformer) Block

<img src="/images/BlockCorrect.png"></img>

Let's put everything together and create our Encoder Block. We will follow a better (imho) naming convention, we call `SegFormerEncoderBlock` the part with the self attention and the mix-fpn and `SegFormerEncoderStage` the whole overlap patch merging + N x `SegFormerEncoderBlock`


Very similar to `ViT`, we have skip connections and normalization layers + Stochastic Depth, also known as Drop Path, (I have an [article](https://towardsdatascience.com/implementing-stochastic-depth-drop-path-in-pytorch-291498c4a974) about it).


https://gist.github.com/1da3e9f7a85aea42fa870b0689a71116




    torch.Size([1, 8, 64, 64])



Okay, let's create a stage. I don't know why, they apply layer norm at the end, so we'll do the same :) 


https://gist.github.com/69c00f3143bdbd9f5b66cb37bcdee6b2

The final `SegFormerEncoder` is composed by multiple stages.


https://gist.github.com/40ecb6719eb5a83990409d1d3ddc0c90

I've added the function `chunks` to keep the code clean. It works like this


https://gist.github.com/5f9a488ba640dca5e6bf64ac420f70cb




    [[1, 2], [3, 4, 5]]



It is handy since `drop_probs` is a list containing the drop path probabilities for each stage's block and we need to pass a list with the correct values to each stage. 

From the encoder, we return a list of inner features, one from each stage.

## Decoder / Neck

Luckily, the decoder/neck's picture matches the original code. They called the decoder part `MLP Layer`.

<img src="/images/DecoderBlock.png" width="400px"></img>

What it does is very simple, it takes `F` features with sizes `batch, channels_i, height_i, width_i` and outputs `F'` features of the same spatial and channel size. The spatial size is fixed to `first_features_spatial_size / 4`. In our case, since our input is a `224x224` image, the output will be a `56x56` mask.

So a single `SegFormerDecoderBlock` contains one upsample layer (for the spatial dimension) and one conv layer (for the channels). The `scale_factor` parameter is needed to tell it how much we want to upsample the feature.


https://gist.github.com/8bd38d15ee7f5f6b774aaddd3146b829

In this case, we don't have stages, so our `SegFormerDecoder` is just a list of blocks. It takes a list of features and returns a list of new features with the same spatial size and channels.


https://gist.github.com/840d1771bb51b2be9d884f777931fe24

## SegFormer Head

<img src="/images/Head.png" width="400px"></img>


We are almost there! The decoder's features are concatenated (remember they all have the same channels and spatial dimensions) on the channel axis. Then, they are passed to a segmentation head to reduce them from `channels * number of features` to `channels`. Finally, a dense layer outputs the final segmentation.


https://gist.github.com/85759bfe5d7ac36ec05d5f6d06eae172

## SegFormer

Well, our final model is just `encoder + decoder + head`. Easy peasy


https://gist.github.com/27f2e3415cfa46d75fa7584af80102e1

Let's try it!


https://gist.github.com/310a7634436224432993c6e291d7cea5




    torch.Size([1, 100, 56, 56])



The output is correct, we expect a mask of spatial shape `image_size // 4` and `224 // 4 = 56`.

We did it! 🎉🎉🎉

## Conclusions


In this article we have seen, step by step, how to create SegFormer; a fast and efficient model for image segmentation.

Thank you for reading it!

Francesco
