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

