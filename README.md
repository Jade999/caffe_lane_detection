# caffe_lane_detection
caffe, lane detection, 300fps
ECCV2020, "Ultra Fast Structure-aware Deep Lane Detection".

this work is base on https://github.com/cfzd/Ultra-Fast-Lane-Detection, thans for the author's gread work!
I just convert the pytorch model to caffe, in order to use this model in tensorrt, mnn,ncnn, nnie and other frameworks.
Speed of the pytorch model is about 263fps, but it been coverted to caffe, it is about 130 fps [1080ti], I don't know why!
But if you thinks this caffe model will help for your work, please feel feel to take it. and thanks the author again.[https://github.com/cfzd/Ultra-Fast-Lane-Detection]
