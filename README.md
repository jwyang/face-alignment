Face alignment in 3000 FPS
==========================

##Introduction

This project is aimed to reproducing (partially) the face alignment algorithm in the CVPR 2014 paper: 

  Face Alignment at 3000 FPS via Regressing Local Binary Features. Shaoqing Ren, Xudong Cao, Yichen Wei, Jian Sun; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 1685-1692 

##How to run the codes?

* First of all, we need prepare datasets, such as afw, lfpw, helen, ibug, etc. All these can be downloaded freely from http://ibug.doc.ic.ac.uk/resources/facial-point-annotations.  Then get the filelist file Path_Images.txt for each dataset (please refer to the Q&A).

* For training, initialize variable dbnames as {'Dataset_a', 'Dataset_b', ..., }, then run train_model in command line window.

* For testing, run test_model in command line window after having obtained trained model. Please remember to initialize dbnames to be the names of dataset you would like to test on.
    
##Dependencies

* liblinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/.

##Learned Model

Off-the-shelf model can be downloaded here: http://pan.baidu.com/s/1i325Rbn, whose configure file can be found in folder "models". 
Its performance is analogy to the lbf_fast model evaluated in the original paper. 

##Q&A

* How to get the file Path_Images.txt?

      It can be obtained by run bat file in the root folder of a dataset, the code is simply "dir /b/s/p/w *.jpg>Path_Images.txt".

* What is Ts_bbox.mat?

      This problem is solved in recent version. Ts_bbox is a transformation matrix to adapt bounding boxes obtained from face detector to the boxes suitable for the face alignment algorithm.

* How to define the variable dbnames in train_model and test_model functions?

      It is formed as a cell array {'dbname_1' 'dbname_2' ... 'dbname_N'}. For example, if we use the images in afw for trainig, we then define it as {'afw'}.

* Why does an error occur when initializing parallel computing?

      It may be caused by Matlab version. For Matlab 2014, it will be okay. For earlier version, please use matlabpool alternatively.

* Some function correspondences from Matlab 2014 to older version

      fitgeotrans -> cp2tform, transformPointsForward -> tformfwd

For those Tecent QQ users, we can discuss more on face algorithms in the group face hacker: 180634020.