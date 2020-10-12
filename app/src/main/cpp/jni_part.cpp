#include <jni.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/types.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/base.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <iterator>



#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <string>
#include <android/log.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>


#define LOG_TAG "Tracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
#define BLOCK_SIZE 24



#define CELL_SIZE   32
#define FRAME_SIZE 164
#define DEFAULT_LBP_R 2
#define DEFAULT_LBP_P 8

using namespace cv;
using namespace std;
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
}
double getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

vector<int> convertToBinary(int x) {
    vector<int> result(8, 0);

    int idx = 0;
    while(x != 0) {
        result[idx] = x % 2;
        ++idx;
        x /= 2;
    }

    reverse(result.begin(), result.end());
    return result;
}

int countTransitions(vector<int> x) {
    int result = 0;
    for(int i = 0; i < 8; ++i)
        result += (x[i] != x[(i+1) % 8]);
    return result;
}

Mat uniformPatternHistogram(const Mat& src, int numPatterns) {
    Mat hist;
    hist = Mat::zeros(1, (numPatterns+1), CV_32SC1);

    for (int i = 0; i < numPatterns; ++i) {
        if (countTransitions(convertToBinary(i)) > 2)
            hist.at<int>(0, i) = -1;
    }

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            int bin = src.at<uchar>(i, j);
            if (hist.at<int>(0, bin) == -1)
                hist.at<int>(0, numPatterns) += 1;
            else
                hist.at<int>(0, bin) += 1;
        }
    }
    return hist;
}

void uniformPatternSpatialHistogram(const Mat& src, Mat& hist, int numPatterns,
                                    int gridX, int gridY, int overlap) {

    int width = src.cols;
    int height = src.rows;
    vector<Mat> histograms;

    Size window = Size(static_cast<int>(floor(src.cols/gridX)),
                       static_cast<int>(floor(src.rows/gridY)));

    for (int x = 0; x <= (width - window.width); x+= (window.width - overlap)) {
        for (int y = 0; y <= (height - window.height); y+= (window.height - overlap)) {
            Mat cell = Mat(src, Rect(x, y, window.width, window.height));
            histograms.push_back(uniformPatternHistogram(cell, numPatterns));
        }
    }

    hist.create(1, histograms.size()*(numPatterns+1), CV_32SC1);
    for (int histIdx = 0; histIdx < histograms.size(); ++histIdx) {
        for (int valIdx = 0; valIdx < (numPatterns+1); ++valIdx) {
            int y = (histIdx * (numPatterns+1)) + valIdx;
            hist.at<int>(0, y) = histograms[histIdx].at<int>(valIdx);
        }
    }
}

vector<int> getFeatureVector(Mat spatial_hist) {
    vector<int> feature_vector;
    for(int j = 0; j < spatial_hist.cols; ++j) {
        if(spatial_hist.at<int>(0, j) != -1)
            feature_vector.push_back(spatial_hist.at<int>(0, j));
    }
    return feature_vector;
}

void LBP(const Mat& src, Mat& dst) {


    const int iRows = src.rows;
    const int iCols = src.cols;


    for(int y = 1;y<iRows-1; y++)
    {
        for(int x = 1;x<iCols-1; x++)
        {
            //定义8邻域
            uchar cNeighbor[8] = {0};
            cNeighbor[0] = src.at<uchar>(y-1,x-1);
            cNeighbor[1] = src.at<uchar>(y-1,x);
            cNeighbor[2] = src.at<uchar>(y-1,x+1);
            cNeighbor[3] = src.at<uchar>(y,  x+1);
            cNeighbor[4] = src.at<uchar>(y+1,x+1);
            cNeighbor[5] = src.at<uchar>(y+1,x);
            cNeighbor[6] = src.at<uchar>(y+1,x-1);
            cNeighbor[7] = src.at<uchar>(y  ,x-1);
            //当前图像的中心像素点
            uchar cCenter = src.at<uchar>(y,x);
            uchar cTemp   = 0;
            //计算LBP的值
            for(int k =0;k<8;k++)
            {
                cTemp += (cNeighbor[k]>=cCenter)*(1<<k);           //将1的二进制数按位左移k位
            }
            dst.at<uchar>(y,x) = cTemp;
        }//for x
    }
}



extern "C"

JNIEXPORT  jstring JNICALL
Java_com_example_myshite_MainActivity_imageprocessing(JNIEnv *env,
                                                      jobject instance,
                                                      jlong inputImage,
                                                      jlong outputImage,

                                                      jint th1,

                                                      jint th2 ,
                                                      jstring path) {
    // TODO: implement imageprocessing()
    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) inputImage;

    Mat &img_output = *(Mat *) outputImage;

    Mat &dst = *(Mat *) outputImage;

    cvtColor( img_input, dst, COLOR_RGB2GRAY);


    blur( dst, dst, Size(5,5) );
    /*
    vector<KeyPoint> v;
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(dst, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(img_input, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }*/
    jstring result;

    //double otsu_thresh_val = cv::threshold(img_input,img_input, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // th1=otsu_thresh_val;
    //th2 = otsu_thresh_val * 0.5;

    // bilateralFilter(img_output, img_output,-1,sigmaColor,sigmaSpace);
    Canny( dst, dst, th1, th2);
//RETR_EXTERNAL
    findContours(dst, contours, hierarchy,CV_FILLED , CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    /*
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(dst, mc[i], 4, color, -1, 8, 0);
    }*/
    /// Get the moments
    vector<Moments> mu(contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mu[i] = moments( contours[i], false ); }

    ///  Get the mass centers:
    vector<Point2f> mc( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

    /// Draw contours
    // Mat drawing = Mat::zeros( dst.size(), CV_8UC3 );
    //for( int i = 0; i< contours.size(); i++ )
    //  {
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( dst, contours, -1, color, 2, 8, hierarchy, 0, Point() );
    //    circle( drawing, mc[i], 4, color, -1, 8, 0 );
    //}

/*
    /// Get the moments
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }

    ///  Get the mass centers:
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

    /// Draw contours
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(dst, mc[i], 4, color, -1, 8, 0);
    }

    const int iRows = dst.rows;
    const int iCols = dst.cols;

    //  cv::Mat resultMat(img_output.size(),img_output.type());

    //变量图像，生成LBP特征

    for(int y = 1;y<iRows-1; y++)
    {
        for(int x = 1;x<iCols-1; x++)
        {
            //定义8邻域
            uchar cNeighbor[8] = {0};
            cNeighbor[0] = img_output.at<uchar>(y-1,x-1);
            cNeighbor[1] = img_output.at<uchar>(y-1,x);
            cNeighbor[2] = img_output.at<uchar>(y-1,x+1);
            cNeighbor[3] = img_output.at<uchar>(y,  x+1);
            cNeighbor[4] = img_output.at<uchar>(y+1,x+1);
            cNeighbor[5] = img_output.at<uchar>(y+1,x);
            cNeighbor[6] = img_output.at<uchar>(y+1,x-1);
            cNeighbor[7] = img_output.at<uchar>(y  ,x-1);
            //当前图像的中心像素点
            uchar cCenter = img_output.at<uchar>(y,x);
            uchar cTemp   = 0;
            //计算LBP的值
            for(int k =0;k<8;k++)
            {
                cTemp += (cNeighbor[k]>=cCenter)*(1<<k);           //将1的二进制数按位左移k位
            }
            img_output.at<uchar>(y,x) = cTemp;
        }//for x
    }*/
    //String &path1=*( String*)path; frommm herree bicha
    LBP(dst, img_output);
    Mat spatial_histogram;
    uniformPatternSpatialHistogram(img_output, spatial_histogram, 256, 3, 3, 0);

    vector<int> feature_vector = getFeatureVector(spatial_histogram);

    LOGD("r hist");

    /*
        FileStorage f;
        // code to write a cv::Mat to file
        f.open("hello.txt", FileStorage::WRITE);
        f << "m" <<  _histograms;
            f.release();


        FILE* file = fopen("/sdcard/hello.txt","w+");

        if (file != NULL)
        {
            fputs("hello", file);
            fflush(file);
            fclose(file);
        }
            __android_log_print(ANDROID_LOG_DEBUG,"path","%s",(path1 + "/info.txt").c_str());
            ofstream file;
            file.open((path1 + "/info.txt").c_str());
            for (int i = 0; i < _histograms.size(); i ++)
            {
                file.write(reinterpret_cast<char*>(&_histograms[i]), sizeof(float) * 100 * 1024);
            }
            file.close();
        FILE* file = NULL;
        file.open();
            for (int i = 0; i < _histograms.size(); i ++)
            {
                file.write(reinterpret_cast<char*>(&_histograms[i]), sizeof(float) * 100 * 1024);
            }

         file.close(); """""""""heeree comback"""""""""""""""
  */
    const char* file_path = (*env).GetStringUTFChars(path,NULL);
    if(file_path != NULL){
        LOGD("From c file_path %s", file_path);
    }

    //打开文件
    FILE* file = fopen(file_path, "w+");
    if(file != NULL){
        LOGD("From c open file success");
    }

    //写入文件
    char data[BLOCK_SIZE]= "I am a child";

    int count = fwrite(data, BLOCK_SIZE , 1, file);
    if(count > 0){
        LOGD("Frome c write file success");
    }

    //关闭文件
    if(file != NULL){
        fclose(file);
    }

    (*env).ReleaseStringUTFChars(path, file_path);



    stringstream ss;
    copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
    string s1 = ss.str();
    s1 = s1.substr(0, s1.length()-1);

    char cstr[s1.size() + 1];

    std::copy(s1.begin(), s1.end(), cstr);
    cstr[s1.size()] = '\0';
    LOGD("finish");

    return env->NewStringUTF("finish");

}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myshite_App_StartActivity_LbpCal(JNIEnv *env, jclass StartActivity, jlong input_image,
                                                  jlong output_image ,jstring path) {
    // TODO: implement LbpCal()
    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) input_image;

    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;


    cvtColor( img_input, dst, COLOR_RGB2GRAY);

    //cvErode (dst,0,0,1);

    // GaussianBlur( dst, dst, Size(5,5),0,0 );
    medianBlur(dst, dst, 5);
    cv::adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

    Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);

    findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    vector<Moments> mu(contours.size());

    int largest_area=0;
    int largest_contour_index=-1;

    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        int area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;
        }
    }

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    drawContours( dst, contours,largest_contour_index, Scalar( 0, 255, 0 ), 2 );
    LBP(dst, img_output);
    Mat spatial_histogram;
    uniformPatternSpatialHistogram(img_output, spatial_histogram, 256, 3, 3, 0);

    vector<int> feature_vector = getFeatureVector(spatial_histogram);

    LOGD("run hist");
    const char* file_path = (*env).GetStringUTFChars(path,NULL);
    if(file_path != NULL){
        LOGD("From c file_path %s", file_path);
    }

    //打开文件
    FILE* file = fopen(file_path, "w+");
    if(file != NULL){
        LOGD("From c open file success");
    }

    //写入文件
    char data[BLOCK_SIZE]= "I am a child";

    int count = fwrite(data, BLOCK_SIZE , 1, file);
    if(count > 0){
        LOGD("Frome c write file success");
    }

    //关闭文件
    if(file != NULL){
        fclose(file);
    }

    (*env).ReleaseStringUTFChars(path, file_path);



    stringstream ss;
    copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
    string s1 = ss.str();
    s1 = s1.substr(0, s1.length()-1);

    char cstr[s1.size() + 1];

    std::copy(s1.begin(), s1.end(), cstr);
    cstr[s1.size()] = '\0';
    LOGD("finish");

    return env->NewStringUTF("finish");
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_Orbcal(JNIEnv *env, jobject thiz, jlong input_image,
                                                  jlong output_image) {
    // TODO: implement Orbcal()
    Mat &captured = *(Mat *)  input_image;

    Mat &target = *(Mat *) output_image;
    // Mat &target = *(Mat *) img2;
    BFMatcher matcher(NORM_HAMMING);

    Ptr<ORB> orb = ORB::create();
    std::vector<cv::KeyPoint> keypointsCaptured;
    //std::vector<cv::KeyPoint> keypointsTarget;

    cv::Mat descriptorsCaptured;

    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> symMatches;
    orb = ORB::create();

//Pre-process
    //Mat &MatchesImage = *(Mat *) image3;

    // medianBlur(captured, target, 5);
    Mat &dst = *(Mat *) output_image;

    cvtColor( captured, dst, COLOR_RGB2GRAY);

    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;





    //GaussianBlur( dst, dst, Size(5,5),0,0 );
    medianBlur(dst, dst, 5);
    cv::adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

    Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);

    findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    cv::Moments mom = cv::moments(contours[0]);
    double hu[7];
    cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
    //Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
    vector<Point2f> mc(contours.size());
    vector<Moments> mu(contours.size());

    int largest_area=0;
    int largest_contour_index=-1;

    for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
    {
        int area = contourArea( contours[i] );  //  Find the area of contour

        if( area > largest_area )
        {
            largest_area = area;
            largest_contour_index = i;
        }
    }

    Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    drawContours( dst, contours,largest_contour_index, Scalar( 0, 255, 0 ), 2 );
    orb->detectAndCompute(dst, noArray(), keypointsCaptured, target);
    if(descriptorsCaptured.type()!=CV_32F) {
        descriptorsCaptured.convertTo(descriptorsCaptured, CV_32F); }
    drawKeypoints(captured, keypointsCaptured, captured, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    FileStorage fs;
    fs.open("/sdcard/descriptors.xml", FileStorage::WRITE);
    fs << "Des" <<target;
    fs.release();
    /*
    cv::Size size = descriptorsCaptured.size();

    int total = size.width * size.height* descriptorsCaptured.channels();


    std::vector<uchar> data(descriptorsCaptured.ptr(), descriptorsCaptured.ptr() + total);
    string s(data.begin(), data.end());
    LOGI("Done writing descriptors.\n");

return s;*/
    __android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_DetectEdge(JNIEnv *env, jobject thiz, jlong input_image,
                                                  jlong output_image) {
    // TODO: implement DetectEdge()
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    vector<vector<Point>>();
    RNG rng(12345);
    vector<Vec4i>();
    Mat &img_input = *(Mat *) input_image;

    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;
    //resize(img_input, img_input, Size(), 0.25, 0.25);
    cvtColor(img_input, dst, COLOR_RGB2GRAY);

    //cvErode (dst,0,0,1);

    // GaussianBlur( dst, dst, Size(5,5),0,0 );
    medianBlur(dst, dst, 5);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(dst, dst, kernel);
    //   morphologyEx(backprBw, backprBw, MORPH_CLOSE, kernel, Point(-1, -1), 2);



//    morphologyEx(backprBw, backprBw, MORPH_OPEN, kernel, Point(-1, -1), 2);

    erode(dst, dst, kernel);
    //  cv::adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    //Laplacian( dst, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    //dilate()
    /* minee*********************************************************
    vector<KeyPoint> v;
    Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(dst, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const KeyPoint& kp = v[i];
        circle(img_input, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    ***************************************************}*/


    //double otsu_thresh_val = cv::threshold(img_input,img_input, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    // th1=otsu_thresh_val;
    //th2 = otsu_thresh_val * 0.5;

    // bilateralFilter(img_output, img_output,-1,sigmaColor,sigmaSpace);

    // Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);
    Mat edges;
    Mat dst1 = dst;
    int scale1 = 1;
    int delta1 = 0;
    int ddepth1 = CV_16S;
    Mat edges_x, edges_y;
    Mat abs_edges_x, abs_edges_y;
    Sobel(dst1, edges_x, ddepth1, 1, 0, 3, scale1, delta1, BORDER_DEFAULT);
    convertScaleAbs(edges_x, abs_edges_x);
    Sobel(dst1, edges_y, ddepth1, 0, 1, 3, scale1, delta1, BORDER_DEFAULT);
    convertScaleAbs(edges_y, abs_edges_y);
    addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, dst);

    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img_input.rows, img_input.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(dst, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        double a = contourArea(contours[i], false);
        if (a > largestArea) {
            largestArea = a;
            largestContourIndex = i;
            boundingRectangle = boundingRect(contours[i]);
        }
    }

    Scalar color(255, 255, 255);
    drawContours(largestContour, contours, largestContourIndex, color, CV_FILLED, 8, hierarchy);
    dst = largestContour;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_lbpfunc(JNIEnv *env, jobject thiz, jlong input_image,
                                               jlong output_image ) {
    // TODO: implement LbpCal()
    vector<vector<Point> > contours;
    RNG rng(12345);
    vector<Vec4i> hierarchy;
    Mat &img_input = *(Mat *) input_image;

    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;


    cvtColor( img_input, dst, COLOR_RGB2GRAY);

    //cvErode (dst,0,0,1);

    // GaussianBlur( dst, dst, Size(5,5),0,0 );
    medianBlur(dst, dst, 5);
    /*  cv::adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

      Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);

      findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
      cv::Moments mom = cv::moments(contours[0]);
      double hu[7];
      cv::HuMoments(mom, hu); // now in hu are your 7 Hu-Moments
      Mat drawing = Mat::zeros(img_output.size(), CV_8UC3);
      vector<Point2f> mc(contours.size());
      vector<Moments> mu(contours.size());

      int largest_area=0;
      int largest_contour_index=-1;

      for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
      {
          int area = contourArea( contours[i] );  //  Find the area of contour

          if( area > largest_area )
          {
              largest_area = area;
              largest_contour_index = i;
          }
      }

      Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      // drawContours(dst, contours, largest_contour_index, color);
      drawContours( dst, contours,largest_contour_index, Scalar( 0, 255, 0 ), 2 );*/
    LBP(dst, img_output);
    Mat spatial_histogram;
    uniformPatternSpatialHistogram(img_output, spatial_histogram, 256, 3, 3, 0);

    vector<int> feature_vector = getFeatureVector(spatial_histogram);

    LOGD("run hist");

}

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