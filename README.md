# CNNMRF
This is the torch implementation for paper "[Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis](http://arxiv.org/abs/1601.04589)"

This algorithm is for
* un-guided image synthesis (for example, classical texture synthesis)
* guided image synthesis (for example, transfer the style between different images)

# Hardware
* For CUDA backend: choose 'speed' if your have at least 4GB graphic memory, and 'memory' otherwise. There is also an opencl backend (thanks to Dionýz Lazar). See "run_trans.lua" and "run_syn.lua" for our reference tests with Titan X, GT750M 2G and Sapphire Radeon R9 280 3G.


# Examples
* guided image synthesis

<p><a href="/data/examples/content.jpg" target="_blank"><img src="/data/examples/content.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/style.jpg" target="_blank"><img src="/data/examples/style.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/3_balanced.png" target="_blank"><img src="/data/examples/Interpolation/3_balanced.png" height="320px" style="max-width:100%;"></a>
</p>

<p>A photo (left) is transfered into a painting (right) using Picasso's self portrait 1907 (middle) as the reference style. Notice important facial features, such as eyes and nose, are faithfully kept as those in the Picasso's painting.</p>

<p><a href="/data/content/1.jpg" target="_blank"><img src="/data/content/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/style/1.jpg" target="_blank"><img src="/data/style/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/0_to_0.png" target="_blank"><img src="/data/examples/1_to_1.png" height="320px" style="max-width:100%;"></a></p>
<p>In this example, we first transfer a cartoon into a photo.</p>
<p><a href="/data/content/1.jpg" target="_blank"><img src="/data/style/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/style/1.jpg" target="_blank"><img src="/data/content/1.jpg" height="320px" style="max-width:100%;"></a>
<a href="/data/examples/1_to_1.png" target="_blank"><img src="/data/examples/0_to_0.png" height="320px" style="max-width:100%;"></a></p>
<p>We then swap the two inputs and transfer the photo into the cartoon.</p>

<p><a href="/data/examples/content.jpg" target="_blank"><img src="/data/examples/content.jpg" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/2_morecontent.png" target="_blank"><img src="/data/examples/Interpolation/2_morecontent.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/4_morestyle.png" target="_blank"><img src="/data/examples/Interpolation/4_morestyle.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/style.jpg" target="_blank"><img src="/data/examples/style.jpg" height="256px" style="max-width:100%;"></a></p>
<p><a href="/data/examples/content2.jpg" target="_blank"><img src="/data/examples/content2.jpg" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/2_morecontent2.png" target="_blank"><img src="/data/examples/Interpolation/2_morecontent2.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/Interpolation/4_morestyle2.png" target="_blank"><img src="/data/examples/Interpolation/4_morestyle2.png" height="256px" style="max-width:100%;"></a>
<a href="/data/examples/style2.jpg" target="_blank"><img src="/data/examples/style2.jpg" height="200px" style="max-width:100%;"></a></p>
<p>It is possible to balance the amount of content and the style in the result: pictures in the second coloumn take more content, and pictures in the third column take more style.</p>

# Setup
This code is based on Torch. It has only been tested on Mac and Ubuntu.

Dependencies:
* [Torch](https://github.com/torch/torch7)
* [loadcaffe](https://github.com/szagoruyko/loadcaffe)

For CUDA backend:
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cudnn](https://developer.nvidia.com/cudnn)

For OpenCL backend:
* [cltorch](https://github.com/hughperkins/cltorch)
* [clnn](https://github.com/hughperkins/clnn)

Pre-trained network:
We use the the original VGG-19 model. You can find the download script at [Neural Style](https://github.com/jcjohnson/neural-style). The downloaded model and prototxt file MUST be saved in the folder "data/models"

# Un-guided Synthesis
* Run `qlua cnnmrf.lua` in a terminal. Most important parameters are '-style_image' for specifying style input image and '-max_size' for resulting image size.
* The content/style images are located in the folders "data/content" and "data/style" respectively. Notice by default the content image is the same as the style image; and the content image is only used for initalization (optional). 
* Results are located in the folder "data/result/freesyn/MRF"
* All parameters are explained in "qlua cnnmrf.lua --help".

# Guided Synthesis
* Run `qlua run_trans.lua` in a terminal. Most important parameters are '-style_image' for specifying style input image, '-content_image' for specifying content input image and '-max_size' for resulting image size.
* The content/style images are located in the folders "data/content" and "data/style" respectively. 
* Results are located in the folder "data/result/trans/MRF"
* Parameters are defined & explained in "run_trans.lua".

# Acknowledgement
* This work is inspired and closely related to the paper: [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. The key difference between their method and our method is the different "style" constraints: While Gatys et al used a global constraint for non-photorealistic synthesis, we use a local constraint which works for both non-photorealistic and photorealistic synthesis. See our paper for more details.
* Our implementation is based on Justin Johnson's implementation of [Neural Style](https://github.com/jcjohnson/neural-style).   
# Other resources (added)
1. [中文论文笔记](https://blog.csdn.net/wangxc_123/article/details/76096082)
2. [其他解读](https://zhuanlan.zhihu.com/p/36256766)
把原来loss function中定义在Gram矩阵上的style loss改成了基于马尔科夫场的一项误差。

3. [基于MRF和CNN的图像生成](https://blog.csdn.net/stdcoutzyx/article/details/54173846)
MRF算法，全程马尔科夫随机场，所谓的马尔科夫特性，就是当前变量的值只和距离最近的变量有关系。
论文将MRF和CNN结合起来，把[2]的Gram矩阵用MRF Loss进行了替代。即能利用CNN抽象特征的提取能力，又能利用MRF的空间布局限制，提高了生成图像的质量。
- 实现上：
MRF Loss是由各个patch的损失值加起来的。
而每个patch的损失值则是该patch A与在风格图像中的最接近A的patch的欧式距离，该损失是基于feature_map的

- basic设置
19层的VGG Net
x随机初始化
MRF Loss使用relu3_1和relu4_1
内容损失使用relu4_2


4. [马尔科夫随机场MRF](https://blog.csdn.net/qiudw/article/details/8087636)
通过MAP inference来确定图中每个节点的label。（最大似然估计（MLE）、最大后验概率估计（MAP））
MRF相比其他方法的优势是：
1）提供了一种principled method来对Prior knowledge建模；
2）MRF可以很容易用定量的方法描述contextual information。
因此，相比其它pixel-based，或local-based 方法，它可以考虑到环境知识的影响，如果建立的图模型得当，进而可能获得全局最优解释，这样正是向human vision更靠近了一步。

[2](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) Gram Matrix: 用来计算feature correlation的。
G_i_j is the inner product betweenthe vectorised feature maps i and j in layer l. 
This is done by using gradient descent from a white noise image to minimise the **mean-squared distance** between the entries of the **Gram matrices** from the *original image* and the Gram matrices of the *image to be generated* [10, 25])
