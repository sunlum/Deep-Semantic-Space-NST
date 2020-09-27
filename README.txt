Baseline：Neural Style Transfer(CVPR 2015) AND Neural Image Analogies

The prior space segmentation and illumination information are used to guide the correct style migration.In addition, this method is introduced as an additional Loss Function and implemented in an optimized way.

====================algorithm procedure=====================
? Initialize the target image X as white noise or content image
? Given a content image, execute:
         a.Obtain spatial segmentation images
         b.Obtain illumination estimation images
         c.Fusion the result of a and b
         d.Calculate Feature Maps of Conv2/3/4_1 layer of c by VGG-19, Obtain "B_sem".
? Given a style image, execute:
         a.Obtain spatial segmentation images
         b.Obtain illumination estimation images
         c.Fusion the result of a and b
         d.Calculate Feature Maps of Conv2/3/4_1 layer of c by VGG-19, Obtain "A_sem".
? Calculate Feature Maps of Conv2/3/4_1 layer of style image by VGG-19, Obtain "A".
? Calculate Feature Maps of Conv2/3/4_1 layer of content image by VGG-19, Obtain "B".
? Use "B" to calculate content loss
? Use "A" to calculate style loss
? Use PatchMatch algorithm to establish the matching of "B_sem" and "A_sem" to obtain NNF
? Deform "A" using NNF index to obtain "A_aligned"
? Calculate DSS loss using "A_aligned"
? Three loss functions are superimposed according to certain weight
? Optimize LossT(x) to get the target image x

=================== Directions ======================
MacOSX, CentOS7,Matlab, Pycharm（或terminal）, Photoshop,Python2.7
Python Tools ：

[
        'h5py>=2.5.0',
        'Keras>=1.0.0',
        'numpy>=1.10.4',
        'Pillow>=3.1.1',
        'PyYAML>=3.11',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.0',
        'six>=1.10.0',
        'Theano>=0.8.2',
]

virtualenv： https://virtualenv.readthedocs.org/en/latest/installation.html

Step 1: space segmentation

（1) Photoshop，image
example: DSS-basedStyleTransfer/DSSImage_matlab/segmentation-taking-examples/***.psd
Save the PS file as JPG/PNG.(The more pixels the image has, the slower the algorithm will be）

（2）DSS-basedStyleTransfer/examples/images/inputs-sem     sunlight_sem.png


Step 2：Superposition of light

（1）Matlab， DSS-basedStyleTransfer/DSSImage_matlab/masktaking    main.m

（2）input image：DSS-basedStyleTransfer/examples/images/inputs，segmentation:DSS-basedStyleTransfer/examples/images/inputs-sem /sunlight.jpg  and  sunlight_sem.png

（3）output image： DSS-basedStyleTransfer/examples/images/inputs-semlight / sunlight_semlight.png

3）Step 3：composite image

（1）enter scripts，activate python-venv

make_image_analogy.py img-style_semlight img-style img-content_semlight img-content output-img-name

eg:make_image_analogy.py  ../examples/images/inputssemlight/oil_semlight.png ../examples/images/inputs/oil.jpg../examples/images/inputs/semlight/sunlight_semlight.png ../examples/images/inputs/sunlight.jpg ../my_outputs/oil/oil2sunlight

（2）If you want to debug the intermediate parameters:：Then choose the compiler as the virtual Python in the project directory in Pycharm, and then execute the file make_image_analogy.

（3）other parameters：

 * --width  默认值为content图像宽，希望output图像的宽度
 * --height 默认值为content图像高，希望output图像的高度
 * --scales 默认值为3，分为3种尺度执行3次风格迁移
 * --iters  默认值为7，每个尺度图象迭代的次数
 * --min-scale   最低尺度大小，默认值为content图像大小的1/3*1/3

 * --output-full 输出中间迭代图像时，是否按照全图像大小输出

 * --analogy-w  深度语义损失函数整体的权重，默认值为5.0
 * --analogy-layers  语义semlight图像在CNN中相互匹配的层数，默认值为['conv2_1', 'conv3_1','conv4_1']

 * --b-content-w   内容损失函数整体的权重，默认值为0.05
 * --content-layers  内容损失生效的层数(深层较好)，默认值为['conv3_1','conv4_1']

 * --nstyle-w 内容损失函数整体的权重，默认值为10.0
 * --nstyle-layers 内容损失生效的层数(深层较好)，默认值为['conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

 * --tv-w   目标输出图像像素内在不变形损失total variation的权重
 * --vgg-weights    VGG模型所在的位置和名字，已下载和设定好

（4）output image： DSS-basedStyleTransfer/my_outputs 
（5）Neural Image Analogies https://raw.githubusercontent.com/awentzonline/image-analogies 相同。

