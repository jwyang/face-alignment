Face alignment in 3000 FPS
==========================

This project is built by reproducing (at least partially) the face alignment algorithm in the CVPR 2014 paper: 

  Face Alignment at 3000 FPS via Regressing Local Binary Features. Shaoqing Ren, Xudong Cao, Yichen Wei, Jian Sun; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014, pp. 1685-1692 

How to run the codes?

(1) First of all, we need prepare datasets, such as afw, lfpw, helen, ibug, etc. All these can be downloaded freely from http://ibug.doc.ic.ac.uk/resources/facial-point-annotations. 

(2) For training, run train_model.m with appropriate dataset name.

    usage: lbfmodel = train_model({'afw' 'lfpw'});

(3) For testing, run test_model.m with dataset name and pre-trained model as input.

    usage: load lbfmodel from disk, and then test_model({'ibug'}, lbfmodel);
    
Dependencies

(1) liblinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/.

Q&A

Q_1: How to get the file Path_Images.txt?

A_1: It can be obtained by run bat file in the root folder of a dataset, the code is simply "dir /b/s/p/w *.jpg>Path_Images.txt".

Q_2: What is Ts_bbox.mat?

A_2: This problem is solved in recent version. Ts_bbox is a transformation matrix to adapt bounding boxes obtained from face detector to the boxes suitable for the face alignment algorithm.

Q_3: How to define the input variable dbnames in train_model and test_model functions?

A_3: It is formed as a cell array {'dbname_1' 'dbname_2' ... 'dbname_N'}. For example, if we use the images in afw for trainig, we then define it as {'afw'}.

Q_4: Why does an error occur when initializing parallel computing?

A_4: It may be caused by Matlab version. For Matlab 2014, it will be okay. For earlier version, please use the following commands:


if params.isparallel ＜/br＞
   if matlabpool('size') <= 0 ＜/br＞
       matlabpool('open','local',4); ＜/br＞
   else ＜/br＞
       disp('Already initialized'); ＜/br＞
   end ＜/br＞
end ＜/br＞

At last, for those who are from china, I am glad to discuss with you in the Tecent QQ group: 180634020


