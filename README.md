# Fast_RCNN_tensorflow
Implementation of Faster RCNN by Tensorflow (In development)  

・Complete  
Load Images of KiTTI Object Detection Datasets  
Preprocessing for Network Input  
RPN(Region Proposal Network)  
Proposal Layer(Convert rpn to rois)  

・ToDO  
Trainer for RCNN  

```
# Prepare KiTTI Datasets
http://www.cvlibs.net/datasets/kitti/eval_object.php

# Compile Cython File
cd cython_util
./setup.sh

# Training RPN  
cd rpn
python rpn.py
```

# ROI Pooling
ROI Pooling layer was implemented by this repository  
https://github.com/deepsense-io/roi-pooling
