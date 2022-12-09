# Image-Deblurring
**This is a Course Project of Advance AI(2022-Fall), Konkuk Univ. :fire:**

deblurring 성능이 좋지 않아, 학기 종료 이후 CNN 네트워크를 개선해보는 작업을 진행할 예정입니다.

Image deblurring is a very interesting technique in the field of computer vision, which restores the sharp information from the blurry image. This problem is quite under constrained, thus previous approaches in literature attempted to resolve this problem via the large amount of training pairs (i.e., blurry-sharp pairs), which are acquired based on the special hardware setting. In this project, applying the convolutional neural network to resolve the problem of image deblurring in an automatic manner.

![image](https://user-images.githubusercontent.com/96612168/202887398-23121db1-19d3-4001-b01b-b0019cd72d58.png)
< Top : blurry input images / Bottom : results of image deblurring >

* Dataset : "GoPro Dataset [1]", which is provided by SNU.
* Performance Evaluation : PSNR metric 
