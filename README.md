<h2>TensorFlow-FlexUNet-Tiled-Image-Segmentation-KITTI-Stereo-2015 (2025/12/05)</h2>
<h3>Revisiting Semantic Segmentation: KITTI-Stereo-2015 </h3>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Tiled Image Segmentation for <b>KITTI-Stereo-2015</b>  based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512  pixels 
<a href="https://drive.google.com/file/d/1BjrZpx1_S77LL7aXbuqvJ1HKqICBtmHp/view?usp=sharing">
<b>Augmented-Tiled-KITTI-Stereo-2015-ImageMask-Dataset.zip </b></a>
which was derived by us from 
<br>
<br>
<a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip">
Download label for semantic and instance segmentation (314 MB)
</a>
<br>
in <a href="https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015">Semantic Segmentation Evaluation
</a>
<br>
<br>
<b>Divide-and-Conquer Strategy</b><br>
The pixel size of images and masks in the <b>KITTI</b> dataset is 1242x375 pixels, which seems to be slightly small to use for our 
segmentation model.<br>
Therefore, we first generated an <b>Upscaled dataset</b> of slightly larger 1536x512 pixels, which are multiple of 512 repectively, 
than the original one, 
and then generated our Augmented-Tiled-KITTI-Stereo-2015 dataset from the <b>Upscaled one</b>.
<br><br>
Please see also our experiment 
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image">
TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image</a>
<br>
<br>
<b>1. Tiled Image Mask Dataset</b><br>
We generated a 512 x 512 pixels tiledly-split dataset from the 1536 x 512 pixels <b>Upscaled dastaset</b> 
by using our offline augmentation tool <a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator</a>
<br>
<br>
<b>2. Train Model by Tiled ImageMask Dataset</b><br>
We trained and validated the TensorFlowFlexUNet model for the KITTI-Stereo by using the 
Tiled-KITTI dataset.
<br><br>
<b>3. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict mask regions for the mini_test images 
with the resolution of resized 1536 x 512 pixels.
<br><br>
<hr>
<b>Actual Image Segmentation for KITTI-Stereo Images of 1536 x 512 pixels</b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<a href="#color_map">KITTI color-class-mapping-table</a><br><br>
<table cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled_inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20002.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20010.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20047.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20015.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20035.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20035.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20046.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20046.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20046.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
1 Dataset Citation
</h3>
The dataset used here was derived from <br>
<br>
<a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip">
Download label for semantic and instance segmentation (314 MB)
</a>
<br>
in <a href="https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015">Semantic Segmentation Evaluation
</a>
<br>
<br>
This is the KITTI semantic segmentation benchmark. It consists of 200 semantically annotated train as well as 200 test images corresponding to the KITTI Stereo and Flow Benchmark 2015. <br>
The data format and metrics are conform with <a href="https://www.cityscapes-dataset.com/">The Cityscapes Dataset.</a><br>

<b>Citation</b><br>
<pre>
Citation
When using this dataset in your research, we will be happy if you cite us:
@article{Alhaija2018IJCV,
  author = { AlhaijaandHassan and MustikovelaandSiva and MeschederandLars and GeigerandAndreas and RotherandCarsten},
  title = {Augmented Reality Meets Computer Vision: Efficient Data Generation for Urban Driving Scenes},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2018}
}
</pre>

<b>Copyright</b><br>

Creative Commons LicenseAll datasets and benchmarks on this page are copyright by us and published under the <br>

<b>Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.</b> <br>
This means that you must attribute the work in the manner specified by the authors, 
you may not use this work for commercial purposes and if you alter, transform, or build upon this work, 
you may distribute the resulting work only under the same license.
<br>
<br>
<h3>
2 Tiled KITTI ImageMask Dataset
</h3>
<h4>2.1 Download KITTI Tiled Dataset</h4>
 If you would like to train this KITTI-Stereo Segmentation model by yourself,
 please download <a href="https://drive.google.com/file/d/1BjrZpx1_S77LL7aXbuqvJ1HKqICBtmHp/view?usp=sharing">
<b>Augmented-Tiled-KITTI-Stereo-2015-ImageMask-Dataset.zip </b></a>
on the google drive, expand the downloaded, and put it under dataset folder to be:
<pre>
./dataset
└─KITTI
    ├─test
    │  ├─images
    │  └─masks
    ├─train
    │  ├─images
    │  └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Tiled-KITTI Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/KITTI/KITTI_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large to use for a training set of our segmentation model.
<br><br> 

<h4>2.2 KITTI Tiled Dataset Derivation</h4>
The original folder structure of <b>data_semantics</b> of <b>KITTI-Stereo-2015</b> is the following:
<pre>
./data_semantics
├─testing
│  └─image_2
└─training
    ├─image_2
    ├─instance
    ├─semantic
    └─semantic_rgb
</pre>
We used the following 2 Python scripts to derive our Tiled dataset from PNG files under <b>training/images_2</b> and <b>training/semantic_rgb</b> folders.
<br>
<ul>
<li>
<a href="./generator/TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a>
</li>
<li>
<a href="./generator/split_master.py">split_master.py</a>
</li>
</ul>
The first Generator script generates our augmented tiledly-split dataset from the Upscaled 200 images and their corresponding masks,
by splitting them into a lot of 512x512 pixels tiles, and augmenting those tiles.<br>
We also used the following KITTI color and class mapping table in <a href="./projects/TensorFlowFlexUNet/KITTI/train_eval_infer.config"> 
<b>train_eval_infer.config</b></a> file 
to define a rgb_map for our mask format between indexed colors and rgb colors.
For more details on Cityscapes labels, please refer to <a href='https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py'>labels.py</a>
<br><br>
<a id="color_map">KITTI color-class-mapping-table</a><br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<caption>KITTI semantic segmentation 19 classes</caption>
<tr><th>Indexed Color</th><th>ID</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td>7</td><td with='80' height='auto'><img src='./color_class_mapping/road.png' widith='40' height='25'></td><td>(128, 64, 128)</td><td>road</td></tr>
<tr><td>2</td><td>8</td><td with='80' height='auto'><img src='./color_class_mapping/sidewalk.png' widith='40' height='25'></td><td>(244, 35, 232)</td><td>sidewalk</td></tr>
<tr><td>3</td><td>11</td><td with='80' height='auto'><img src='./color_class_mapping/building.png' widith='40' height='25'></td><td>(70, 70, 70)</td><td>building</td></tr>
<tr><td>4</td><td>12</td><td with='80' height='auto'><img src='./color_class_mapping/wall.png' widith='40' height='25'></td><td>(102, 102, 156)</td><td>wall</td></tr>
<tr><td>5</td><td>13</td><td with='80' height='auto'><img src='./color_class_mapping/fence.png' widith='40' height='25'></td><td>(190, 153, 153)</td><td>fence</td></tr>
<tr><td>6</td><td>17</td><td with='80' height='auto'><img src='./color_class_mapping/pole.png' widith='40' height='25'></td><td>(153, 153, 153)</td><td>pole</td></tr>
<tr><td>7</td><td>19</td><td with='80' height='auto'><img src='./color_class_mapping/traffic light.png' widith='40' height='25'></td><td>(250, 170, 30)</td><td>traffic light</td></tr>
<tr><td>8</td><td>20</td><td with='80' height='auto'><img src='./color_class_mapping/traffic sign.png' widith='40' height='25'></td><td>(220, 220, 0)</td><td>traffic sign</td></tr>
<tr><td>9</td><td>21</td><td with='80' height='auto'><img src='./color_class_mapping/vegetation.png' widith='40' height='25'></td><td>(107, 142, 35)</td><td>vegetation</td></tr>
<tr><td>10</td><td>22</td><td with='80' height='auto'><img src='./color_class_mapping/terrain.png' widith='40' height='25'></td><td>(152, 251, 152)</td><td>terrain</td></tr>
<tr><td>11</td><td>23</td><td with='80' height='auto'><img src='./color_class_mapping/sky.png' widith='40' height='25'></td><td>(70, 130, 180)</td><td>sky</td></tr>
<tr><td>12</td><td>24</td><td with='80' height='auto'><img src='./color_class_mapping/person.png' widith='40' height='25'></td><td>(220, 20, 60)</td><td>person</td></tr>
<tr><td>13</td><td>25</td><td with='80' height='auto'><img src='./color_class_mapping/rider.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>rider</td></tr>
<tr><td>14</td><td>26</td><td with='80' height='auto'><img src='./color_class_mapping/car.png' widith='40' height='25'></td><td>(0, 0, 142)</td><td>car</td></tr>
<tr><td>15</td><td>27</td><td with='80' height='auto'><img src='./color_class_mapping/truck.png' widith='40' height='25'></td><td>(0, 0, 70)</td><td>truck</td></tr>
<tr><td>16</td><td>28</td><td with='80' height='auto'><img src='./color_class_mapping/bus.png' widith='40' height='25'></td><td>(0, 60, 100)</td><td>bus</td></tr>
<tr><td>17</td><td>31</td><td with='80' height='auto'><img src='./color_class_mapping/train.png' widith='40' height='25'></td><td>(0, 80, 100)</td><td>train</td></tr>
<tr><td>18</td><td>32</td><td with='80' height='auto'><img src='./color_class_mapping/motorcycle.png' widith='40' height='25'></td><td>(0, 0, 230)</td><td>motorcycle</td></tr>
<tr><td>19</td><td>33</td><td with='80' height='auto'><img src='./color_class_mapping/bicycle.png' widith='40' height='25'></td><td>(119, 11, 32)</td><td>bicycle</td></tr>
</table>

<br><br>
<h4>2.3 Tiled Image and Mask Samples</h4>

<b>Tiled Train Images Sample</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Tiled Train Masks Sample</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNetModel
</h3>
 We trained KITTI TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/KITTI/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/KITTI, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and a large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalization=True

num_classes    = 20
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for KITTI 1+19 classes.<br>
For more detail, please refer to <a href="#color_map">KITTI color-class-mapping-table</a><br>
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;         
rgb_map={(0,0,0):0,(128,64,128):1,(244,35,232):2,(70,70,70):3,(102,102,156):4,(190,153,153):5,(153,153,153):6,(250,170,30):7,(220,220,0):8,(107,142,35):9,(152,251,152):10,(70,130,180):11,(220,20,60):12,(255,0,0):13,(0,0,142):14,(0,0,70):15,(0,60,100):16,(0,80,100):17,(0,0,230):18,(119,11,32):19,}
</pre>

<b>Tiled inference parameters</b><br>
<pre>
[tiledinfer] 
overlapping = 128
images_dir    = "./mini_test/images/"
output_dir    = "./mini_test_output_tiled/"
</pre>

<b>Epoch change tiled inference callback</b><br>
Enabled <a href="./src/EpochChangeTiledInferencer.py">epoch_change_tiled_infer callback (EpochChangeTiledInferencer.py)</a></b>.<br>
<pre>
[train]
poch_change_infer     = False
epoch_change_infer_dir =  "./epoch_change_infer"
epoch_change_tiled_infer     = True
epoch_change_tiled_infer_dir =  "./epoch_change_tiled_infer"
</pre>

By using this callback, on every epoch_change, the tiled inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (epoch 1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<b>Epoch_change_inference output at middlepoint (epoch 44,45,46,47)</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<b>Epoch_change_inference output at ending (epoch 89,90,91,92)</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
In this experiment, the training process was stopped at epoch 92 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/train_console_output_at_epoch92.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/KITTI/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/eval/train_metrics.png" width="520" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/KITTI/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/KITTI</b> folder,
and run the following bat file to evaluate TensorFlowFlexUNet model for KITTI.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/evaluate_console_output_at_epoch92.png" width="880" height="auto">
<br><br>Image-Segmentation-Aerial-Imagery

<a href="./projects/TensorFlowFlexUNet/KITTI/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this KITTI/test was not low, and dice_coef_multiclass not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1697
dice_coef_multiclass,0.9356
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/KITTI</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for KITTI.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled_inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/KITTI/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for KITTI Images of 1536 x 512 pixels </b><br>
As shown below, the tiled_inferred masks predicted by our segmentation model trained on the 
Tiled dataset appear similar to the ground truth masks, but they lack precision in certain areas.<br>
<a href="#color_map">KITTI color-class-mapping-table</a><br><br>
<table cellpadding='10'>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20001.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20004.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20004.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20004.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20014.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20014.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20024.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20024.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20024.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20037.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20037.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20037.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20047.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20047.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20032.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20032.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20032.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20020.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20030.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20030.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20030.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/images/20048.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test/masks/20048.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KITTI/mini_test_output_tiled/20048.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. City-Scale Road Extraction from Satellite Imagery v2:
Road Speeds and Travel Times</b><br>
Adam Van Etten<br>
<a href="https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf">
https://openaccess.thecvf.com/content_WACV_2020/papers/Van_Etten_City-Scale_Road_Extraction_from_Satellite_Imagery_v2_Road_Speeds_and_WACV_2020_paper.pdf
</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Aerial-Imagery
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image">
https://github.com/atlan-antillia/TensorFlow-FlexUNet-Tiled-Image-Segmentation-Drone-Image
</a>
<br>
<br>

