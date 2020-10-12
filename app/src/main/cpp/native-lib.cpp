
/*
#include <jni.h>

#include "com_example_android_opencvdemo_OpenCvMaker.h"


extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeGray
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toGray((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toGray(Mat img, Mat& gray) {
    cv::cvtColor(img, gray,CV_RGBA2GRAY);
    if(gray.rows == img.rows && gray.cols == img.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeCanny
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toCanny((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toCanny(Mat input, Mat& output) {
    Canny(input,output,75,150,3,false);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeDilate
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toDilate((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toDilate(Mat input, Mat& output){
    dilate(input,output,Mat(), Point(-1,-1),2,1,1);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_example_myshite_OpenCvMaker_makeErode
        (JNIEnv *, jclass, jlong addrInput, jlong addrOutput){
    return (jint) toErode((*(Mat*) addrInput),(*(Mat*) addrOutput));
}

int toErode(Mat input, Mat& output){
    erode(input,output,Mat(), Point(-1,-1),2,1,1);
    if(output.rows == input.rows && output.cols == input.cols)
        return 1;
    return 0;
}
*/
#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sstream>
#include <android/log.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

//#include "JHashMap.h"
#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
using namespace std;
using namespace cv;
using namespace cv::ml;

#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_Kmeans_kmeans(JNIEnv *env, jclass clazz, jlong img1) {



    Mat &captured = *(Mat *)  img1;

    // Mat &target = *(Mat *) img2;
    BFMatcher matcher(NORM_L2);

    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;
    cv::Mat descriptorsTarget;
//cv::Mat captured;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    //Mat &MatchesImage = *(Mat *) image3;

    medianBlur(captured, captured, 5);
/*
    cv::Mat reshaped_image = captured.reshape(1, captured.cols * captured.rows);
    cv::Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);
//////////if(descriptors_1.type()!=CV_32F) {
//////////////////descriptors_1.convertTo(descriptors_1, CV_32F); }

 /*
 Mat labels = new Mat(contours.size(), 2, CvType.CV_32SC1);
    TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 100, 1.0);
    Mat centers = new Mat();
    Core.kmeans(samples32final, 5, labels, criteria, 10, Core.KMEANS_PP_CENTERS, centers);
    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
*/
    // medianBlur(target, target, 5);

    orb->detectAndCompute(captured, noArray(), keypointsCaptured, descriptorsCaptured);
    if(descriptorsCaptured.type()!=CV_32F) {
        descriptorsCaptured.convertTo(descriptorsCaptured, CV_32F); }
    // drawKeypoints(captured, keypointsCaptured, target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    FileStorage fs;
    fs.open("/sdcard/descriptors.xml", FileStorage::WRITE);
    fs << "Des" << descriptorsCaptured;
    fs.release();


    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(descriptorsCaptured, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);


    LOGI("Done writing descriptors.\n");
    // orb->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints2 size = %d", keypointsTarget.size());
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

//Match images based on k nearest neighbour
    // std::vector<std::vector<cv::DMatch> > matches1;
    //   matcher.knnMatch(descriptorsCaptured , descriptorsTarget,
    // matches1, 2);
//__android_log_print(ANDROID_LOG_INFO, "sometag", "Matches1 = %d",     matches1.size());
    std::vector<std::vector<cv::DMatch> > matches2;
    matcher.knnMatch(descriptorsTarget , descriptorsCaptured,
                     matches2, 2);
    //Mat &MatchesImage = *(Mat *) image3;
    LOGD("run hist");

    //  drawMatches(captured,keypointsCaptured, target, keypointsTarget, matches, MatchesImage, cv::Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    /* Mat_<float> trainingData(10, 2);
     trainingData <<  0, 0,
             0.5, 0,
             0.5, 0.25,
             1, 1.25,
             1, 1.5,
             1, 1,
             0.5, 1.5,
             0.25, 1,
             2, 1.5,
             2, 2.5;

     Mat_<int> trainingLabels(1, 10);
     trainingLabels << 0, 0, 0, 1, 1, 1, 0, 1, 1, 1;

     Ptr<SVM> svm = SVM::create();
     svm->setType(SVM::C_SVC);
     svm->setKernel(SVM::LINEAR);
     svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

     svm->train(trainingData, ROW_SAMPLE, trainingLabels);

     Mat_<float> testFeatures(1, 2);
     testFeatures << 2.5, 2.5;

     Mat res;
     svm->predict(testFeatures, res);*/



}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_orbMatchTry_compareImages(JNIEnv *env, jclass clazz, jlong img1,
                                                   jlong img2, jlong image3) {



    Mat &captured = *(Mat *)  img1;

    BFMatcher matcher(NORM_L2);
    Mat &target = *(Mat *) img2;
    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;
    cv::Mat descriptorsTarget;
//cv::Mat captured;
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    Mat &MatchesImage = *(Mat *) image3;

    medianBlur(captured, captured, 5);
/*
    cv::Mat reshaped_image = captured.reshape(1, captured.cols * captured.rows);
    cv::Mat reshaped_image32f;
    reshaped_image.convertTo(reshaped_image32f, CV_32FC1, 1.0 / 255.0);

    cv::Mat labels;
    int cluster_number = 5;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT, 100, 1);
    cv::Mat centers;
    cv::kmeans(reshaped_image32f, cluster_number, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
*/
    medianBlur(target, target, 5);

    orb->detectAndCompute(captured, noArray(), keypointsCaptured, descriptorsCaptured);
    orb->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints2 size = %d", keypointsTarget.size());
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

//Match images based on k nearest neighbour
    std::vector<std::vector<cv::DMatch> > matches1;
    matcher.knnMatch(descriptorsCaptured , descriptorsTarget,
                     matches1, 2);

    //Mat &MatchesImage = *(Mat *) image3;
    LOGD("run hist");
    /*
    double dist_th = 64;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < dist_th )
        { good_matches.push_back( matches[i]); }
    }*/
//cluster the feature vectors

    /*
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)captured.cols, 0 );
    obj_corners[2] = Point2f( (float)captured.cols, (float)captured.rows );
    obj_corners[3] = Point2f( 0, (float)captured.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( target, scene_corners[0] + Point2f((float)captured.cols, 0),
          scene_corners[1] + Point2f((float)captured.cols, 0), Scalar(0, 255, 0), 4 );
    line( target, scene_corners[1] + Point2f((float)captured.cols, 0),
          scene_corners[2] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    line( target, scene_corners[2] + Point2f((float)captured.cols, 0),
          scene_corners[3] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    line( target, scene_corners[3] + Point2f((float)captured.cols, 0),
          scene_corners[0] + Point2f((float)captured.cols, 0), Scalar( 0, 255, 0), 4 );
    std::vector< DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypointsCaptured[matches[i].queryIdx].pt);
        scene.push_back(keypointsTarget[matches[i].trainIdx].pt);
    }
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0);
    obj_corners[1] = Point(captured.cols, 0);
    obj_corners[2] = Point(captured.cols,captured.rows);
    obj_corners[3] = Point(0, captured.rows);
    std::vector<Point2f> scene_corners(4);

    Mat H = findHomography(obj, scene, RANSAC);
    perspectiveTransform(obj_corners, scene_corners, H);
    std::vector<Point2f> scene_corners_ ;
    scene_corners_ = scene_corners;

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(MatchesImage,
         scene_corners[0] + Point2f((float)captured.cols, 0), scene_corners[1] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[1] + Point2f((float)captured.cols, 0), scene_corners[2] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[2] + Point2f((float)captured.cols, 0), scene_corners[3] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);
    line(MatchesImage,
         scene_corners[3] + Point2f((float)captured.cols, 0), scene_corners[0] + Point2f((float)captured.cols, 0),
         Scalar(0, 255, 0), 2, LINE_AA);*/

}
////////////////////////////////////////////////

int current_radius1 = 3;
Mat Expand_LBP_demo1(Mat gray_src)
{
    int offset = current_radius1 * 2;
    Mat elbpImg = Mat::zeros(gray_src.rows-offset, gray_src.cols-offset, CV_8UC1);
    int numNeighbor = 8;
    for (int n = 0; n < numNeighbor; n++) {
        float x = current_radius1 * cos((2 * CV_PI * n) / numNeighbor);
        float y = current_radius1 * (-sin((2 * CV_PI * n) / numNeighbor));

        int fx = static_cast<int>(floor(x)); //向下取整，它返回的是小于或等于函数参数,并且与之最接近的整数
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x)); //向上取整，它返回的是大于或等于函数参数,并且与之最接近的整数
        int cy = static_cast<int>(ceil(y));

        float ty = y - fy;
        float tx = x = fx;

        float w1 = (1 - tx) * (1 - ty);
        float w2 = (tx) * (1 - ty);
        float w3 = (1 - tx) * (ty);
        float w4 = (tx) * (ty);

#pragma omp parallel for
        {
            for (int row = current_radius1; row < (gray_src.rows - current_radius1); row++) {
                for (int col = current_radius1; col < (gray_src.cols - current_radius1); col++) {
                    float t = w1 * gray_src.at<uchar>(row + fy, col + fx) +
                              w2 * gray_src.at<uchar>(row + fy, col + cx) +
                              w3 * gray_src.at<uchar>(row + cy, col + fx) +
                              w4 * gray_src.at<uchar>(row + cy, col + cx);
                    elbpImg.at<uchar>(row - current_radius1, col - current_radius1) +=
                            ((t > gray_src.at<uchar>(row, col)) &&
                             (abs(t - gray_src.at<uchar>(row, col)) >
                              std::numeric_limits<float>::epsilon())) << n;

                }
            }

        }
    }

    return elbpImg;

}
class WatershedSegmenter1{
private:
    cv::Mat markers;
public:
    void setMarkers(cv::Mat& markerImage)
    {
        markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
        cv::watershed(image, markers);
        markers.convertTo(markers,CV_8U);
        return markers;
    }
};
Mat sobell(Mat gray){
    Mat edges;

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat edges_x, edges_y;
    Mat abs_edges_x, abs_edges_y;
    Sobel(gray, edges_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs( edges_x, abs_edges_x );
    Sobel(gray, edges_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(edges_y, abs_edges_y);
    addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, edges);

    return edges;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_ChooseimageActivity_excract1(JNIEnv *env, jobject thiz,
        jlong input_image, jlong output_image) {
// TODO: implement excract1()


Mat &img1= *(Mat *)  input_image;
Mat img3=img1;  Mat gray,img5;
// resize(img1, img1, Size(), 0.25, 0.25);

cv::cvtColor(img1 , img5 , CV_RGBA2RGB);

//  GaussianBlur(img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT);

Mat &img2 = *(Mat *) output_image;

const int channels[] = {0, 1, 2};
const int histSize[] = {32, 32, 32};
const float rgbRange[] = {0, 256};
const float* ranges[] = {rgbRange, rgbRange, rgbRange};

Mat hist;
Mat im32fc3, backpr32f, backpr8u, backprBw, kernel;

int64 t0 = cv::getTickCount();


img5.convertTo(im32fc3, CV_32FC3);
calcHist(&im32fc3, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
calcBackProject(&im32fc3, 1, channels, hist, backpr32f, ranges);
cv::Mat img4,edges;
double minval, maxval;
minMaxIdx(backpr32f, &minval, &maxval);

threshold(backpr32f, backpr32f, maxval/32, 255, THRESH_TOZERO);

backpr32f.convertTo(backpr8u, CV_8U, 255.0/maxval);



threshold(backpr8u, backprBw, 10, 255, THRESH_BINARY);
//bitwise_or(backprBw,edges,backprBw);
//  img2=backprBw;
kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

dilate(backprBw, backprBw, kernel);
morphologyEx(backprBw, backprBw, MORPH_CLOSE, kernel, Point(-1, -1), 2);

backprBw = 255 - backprBw;

morphologyEx(backprBw, backprBw, MORPH_OPEN, kernel, Point(-1, -1), 2);

erode(backprBw, backprBw, kernel);

Mat mask(backpr8u.rows, backpr8u.cols, CV_8U);

mask.setTo(GC_PR_BGD);
mask.setTo(GC_PR_FGD, backprBw);

Mat bgdModel, fgdModel;
grabCut(img5, mask, Rect(), bgdModel, fgdModel,GC_INIT_WITH_MASK);

Mat fg = mask == GC_PR_FGD;
int largestArea = 0;
int largestContourIndex = 0;
Rect boundingRectangle;
Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;

findContours(fg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

for(int i=0; i<contours.size(); i++)
{
double a = contourArea(contours[i], false);
if(a > largestArea)
{
largestArea = a;
largestContourIndex = i;
boundingRectangle = boundingRect(contours[i]);
}
}

Scalar color(255, 255, 255);
drawContours(largestContour, contours, largestContourIndex, color, CV_FILLED, 8, hierarchy); //Draw the largest contour using previously stored index.
// rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);

img5.copyTo(img3,largestContour);
cvtColor(img3,img3,COLOR_BGRA2BGR);
cv::Mat blank(img3.size(),CV_8U,cv::Scalar(0xFF));
cv::Mat dest;


// Create markers image
cv::Mat markers(img3.size(),CV_8U,cv::Scalar(-1));
//Rect(topleftcornerX, topleftcornerY, width, height);
//top rectangle
markers(Rect(0,0,img3.cols, 5)) = Scalar::all(1);
//bottom rectangle
markers(Rect(0,img3.rows-5,img3.cols, 5)) = Scalar::all(1);
//left rectangle
markers(Rect(0,0,5,img3.rows)) = Scalar::all(1);
//right rectangle
markers(Rect(img3.cols-5,0,5,img3.rows)) = Scalar::all(1);
//centre rectangle
int centreW = img3.cols/4;
int centreH = img3.rows/4;
markers(Rect((img3.cols/2)-(centreW/2),(img1.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
markers.convertTo(markers,CV_BGR2GRAY);


//Create watershed segmentation object
WatershedSegmenter1 segmenter;
segmenter.setMarkers(markers);
cv::Mat wshedMask = segmenter.process(img3);
// cv::Mat mask;
convertScaleAbs(wshedMask, mask, 1, 0);
double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
bitwise_and(img3, img3, img2, mask);
int64 t1 = cv::getTickCount();
double secs = (t1-t0)/cv::getTickFrequency();
__android_log_print(ANDROID_LOG_INFO, "sometag", "time  = %f", secs);
img2.convertTo(dest,CV_8U);

}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_ChooseimageActivity_elbbp1(JNIEnv *env, jobject thiz,
        jlong input_image, jlong output_image,
        jlong hist) {
// TODO: implement elbbp1()


Mat &img1 = *(Mat *) input_image;
Mat &img2 = *(Mat *) output_image;
Mat &histt = *(Mat *) hist;
cvtColor(img1,img1,CV_RGB2GRAY);
Mat img3;
string s="";
img3=Expand_LBP_demo1(img1);
Mat spat;
Mat lbp_hist, lbp1_hist;
int histSize[] = {256};
float s_ranges[] = { 0, 256 };
const float* ranges[] = { s_ranges };

// Use the o-th and 1-st channels
int channels[] = { 0 };

calcHist( &img3, 1, channels, Mat(), lbp_hist, 1, histSize, ranges, true, false );

histt=lbp_hist;
FileStorage fs;
fs.open("/sdcard/lbp/file.xml", FileStorage::WRITE);
fs << "Des" <<lbp_hist;
fs.release();
img2=img3;
/* jintArray arr = env->NewIntArray( feature_vector.size() );
 env->SetIntArrayRegion( arr, 0, feature_vector.size(), ( jint * ) &feature_vector[0] );
 return arr;*/
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_ChooseimageActivity_histths1(JNIEnv *env, jobject thiz,
        jlong input_image, jlong output_image) {
// TODO: implement histths1()
Mat &img1 = *(Mat *) input_image;
Mat &img2 = *(Mat *) output_image;
//variables preparing
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
int hbins = 30, sbins = 32;
int channels[] = {0};
int histSize[] = {hbins};
float hranges[] = { 0, 180 };
float sranges[] = { 0, 255 };
const float* ranges[] = { hranges};

UMat b;

Mat patch_HSV;
//MatND HistA, HistB;
Mat HistA;
//cal histogram & normalization
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvtColor(img1, patch_HSV, CV_BGR2HSV);
calcHist( &patch_HSV, 1, channels,  Mat(), // do not use mask
HistA, 1, histSize, ranges,
true, // the histogram is uniform
false );

normalize(HistA, HistA,  0, 255, CV_MINMAX);

img2=HistA;

}
extern "C"
JNIEXPORT jdoubleArray JNICALL
        Java_com_example_myshite_App_ChooseimageActivity_humom1(JNIEnv *env, jobject thiz,
jlong input_image, jlong output_image) {
// TODO: implement humom1()
Mat &img1 = *(Mat *) input_image;
Mat &img2 = *(Mat *) output_image;
Mat im ;
cvtColor(img1,im, CV_RGB2GRAY);



Mat edges =sobell(im);


threshold(edges, img2, 0, 255, THRESH_OTSU);
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
//findContours(img2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

// Moments moments = cv::moments(contours[0], false);
Moments moments = cv::moments(img2, false);

// Calculate Hu Moments
double huMoments[7];
HuMoments(moments, huMoments);


for(int i = 0; i < 7; i++)

{

huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));

}
jdoubleArray arr = env->NewDoubleArray( 7 );
env->SetDoubleArrayRegion( arr, 0,7, ( jdouble * ) &huMoments[0] );
return arr;

}