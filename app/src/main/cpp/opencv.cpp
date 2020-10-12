#include <jni.h>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <android/log.h>
#include <queue>
#include <iterator>
#include "lbp.hpp"
#include <omp.h>
#include <pthread.h>

#define LOG_TAG "FaceDetection/DetectionBasedTracker"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))
using namespace std;
using namespace cv;

enum COLOR{ RED, BLUE };

int current_radius = 3;
int max_count = 20;
//扩展LBP demo

class Parallel_process : public cv::ParallelLoopBody
{

private:
    cv::Mat img,b;
    cv::Mat& retVal;

    int size;
    int diff;

public:
    Parallel_process(cv::Mat inputImgage, cv::Mat& outImage,cv::Mat& mage,
                     int sizeVal, int diffVal)
            : img(inputImgage), retVal(outImage),b(mage),
              size(sizeVal), diff(diffVal){}

    virtual void operator()(const cv::Range& range) const
    {
        for(int i = range.start; i < range.end; i++)
        {
            /* divide image in 'diff' number
            of parts and process simultaneously */

            cv::Mat in(img, cv::Rect(0, (img.rows/diff)*i,
                                     img.cols, img.rows/diff));
            cv::Mat out(retVal, cv::Rect(0, (retVal.rows/diff)*i,retVal.cols, retVal.rows/diff));


            retVal.setTo(GC_PR_BGD);
            retVal.setTo(GC_PR_FGD, b);

            Mat bgdModel, fgdModel;
            //cv::GaussianBlur(in, out, cv::Size(size, size), 0);
            grabCut(in, out, Rect(), bgdModel, fgdModel,GC_INIT_WITH_MASK);
        }
    }
};

void elbhistogram(const Mat& src, Mat& hist, int numPatterns) {
    hist = Mat::zeros(1, numPatterns, CV_8UC1);
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            int bin = src.at<uchar>(i,j);
            hist.at<int>(0,bin) += 1;
        }
    }

}

Mat Expand_LBP_demo(Mat gray_src)
{
    int offset = current_radius * 2;
    Mat elbpImg = Mat::zeros(gray_src.rows-offset, gray_src.cols-offset, CV_8UC1);
    int numNeighbor = 8;
    for (int n = 0; n < numNeighbor; n++) {
        float x = current_radius * cos((2 * CV_PI * n) / numNeighbor);
        float y = current_radius * (-sin((2 * CV_PI * n) / numNeighbor));

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
            for (int row = current_radius; row < (gray_src.rows - current_radius); row++) {
                for (int col = current_radius; col < (gray_src.cols - current_radius); col++) {
                    float t = w1 * gray_src.at<uchar>(row + fy, col + fx) +
                              w2 * gray_src.at<uchar>(row + fy, col + cx) +
                              w3 * gray_src.at<uchar>(row + cy, col + fx) +
                              w4 * gray_src.at<uchar>(row + cy, col + cx);
                    elbpImg.at<uchar>(row - current_radius, col - current_radius) +=
                            ((t > gray_src.at<uchar>(row, col)) &&
                             (abs(t - gray_src.at<uchar>(row, col)) >
                              std::numeric_limits<float>::epsilon())) << n;

                }
            }

        }
    }

    return elbpImg;

}

void LBPsht(const Mat& src, Mat& dst) {

    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i = 1; i < (src.rows - 1); i++) {
        for(int j = 1; j < (src.cols-1); j++) {
            uchar center = src.at<uchar>(i,j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i-1,j-1) > center) << 7;
            code |= (src.at<uchar>(i-1,j) > center) << 6;
            code |= (src.at<uchar>(i-1,j+1) > center) << 5;
            code |= (src.at<uchar>(i,j+1) > center) << 4;
            code |= (src.at<uchar>(i+1,j+1) > center) << 3;
            code |= (src.at<uchar>(i+1,j) > center) << 2;
            code |= (src.at<uchar>(i+1,j-1) > center) << 1;
            code |= (src.at<uchar>(i,j-1) > center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}
Mat ycbr(Mat img1){



    cvtColor(img1, img1, CV_BGRA2BGR);
    Mat hsvImg;


    cvtColor(img1, hsvImg, CV_BGR2HSV);

    pyrMeanShiftFiltering( img1,img1 , 10, 35, 3);
    medianBlur(img1, img1,7);
    cvtColor(img1, img1, CV_BGR2YCrCb);


    Mat tmp;
    Mat();
    Mat Y, Cr, Cb;
    vector<Mat> channels;


    split(img1, channels);
    Y = channels.at(0);
    Cr = channels.at(1);
    Cb = channels.at(2);

    Mat result(img1.rows, img1.cols, CV_8UC1);


    for (int j = 1; j < Y.rows - 1; j++)
    {
        uchar* currentCr = Cr.ptr< uchar>(j);
        uchar* currentCb = Cb.ptr< uchar>(j);
        uchar* current = result.ptr< uchar>(j);
        for (int i = 1; i < Y.cols - 1; i++)
        {
            if ((currentCr[i] > 150) && (currentCr[i] < 175) && (currentCb[i] > 80) && (currentCb[i] < 130))
                current[i] = 255;
            else
                current[i] = 0;
        }}
    return result;
}

/////////
void showImgContours(Mat &threshedimg, Mat &result, COLOR color)

{



    vector<vector<Point>> contours;

    vector<Vec4i> hierarchy;

    int largest_area = 0;

    int largest_contour_index = 0;

    findContours(threshedimg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));



    vector<vector<Point> > contours_poly(contours.size());

    vector<Rect> boundRect(contours.size());


//contours 0: smallest, contours max: biggest

    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2)

    {

        return contourArea(c1, false) < contourArea(c2, false);

    });



    vector<vector<Point>> top5_contours;

    for (int i = 1; i<6; i++){

        if (contours.size()>i)

        {

            top5_contours.push_back(contours[contours.size() - i]);

        }

    }




    for (int i = 0; i <top5_contours.size(); i++)

    {

        approxPolyDP(Mat(top5_contours[i]), contours_poly[i], 3, true);

        boundRect[i] = boundingRect(Mat(contours_poly[i]));

    }





/// Draw polygonal contour + bonding rects + circles

    for (int i = 0; i < contours.size(); i++)

    {

        Scalar color = CV_RGB(255, 0, 0);

        rectangle(result, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

    }



/*

for (int i = 0; i< contours.size(); i++) // iterate through each contour.

{

double a = contourArea(contours[i], false);  //  Find the area of contour

if (a>largest_area)

{

largest_area = a;

largest_contour_index = i;                //Store the index of largest contour

}

}

*/

    if (contours.size() > 0)

    {

        switch (color) {

            case RED:

                drawContours(result, contours, -1, CV_RGB(255, 0, 0), 2, 8, hierarchy);

                break;

            case BLUE:

                drawContours(result, contours, -1, CV_RGB(0, 0, 255), 2, 8, hierarchy);

                break;

        }

    }


}
//////////////
static Scalar randomColor(RNG& rng)
{
    int icolor = (unsigned)rng;
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

//background removal
Mat removeLight(Mat img, Mat pattern, int method)
{
    Mat aux;
    // if method is normalization
    if (method == 1)
    {
        // Require change our image to 32 float for division
        Mat img32, pattern32;
        img.convertTo(img32, CV_32F);
        pattern.convertTo(pattern32, CV_32F);
        // Divide the image by the pattern
        aux = 1 - (img32 / pattern32);
        // Scale it to convert to 8bit format
        aux = aux * 255;
        // Convert 8 bits format
        aux.convertTo(aux, CV_8U);
    }
    else{
        aux = pattern - img;
    }
    return aux;
}

//creates this light pattern or background
Mat calculateLightPattern(Mat img)
{
    Mat pattern;
    // Basic and effective way to calculate the light pattern from one image
    blur(img, pattern, Size(img.cols / 3, img.cols / 3));
    return pattern;
}

void ConnectedComponents(Mat img)
{
    // Use connected components to divide our possibles parts of images
    Mat labels;
    int num_objects = connectedComponents(img, labels);
    // Check the number of objects detected
    if (num_objects < 2){
        cout << "No objects detected" << endl;
        return;
    }
    else{
        cout << "Number of objects detected: " << num_objects - 1 << endl;
    }
    // Create output image coloring the objects
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (int i = 1; i<num_objects; i++){
        Mat mask = labels == i;
        output.setTo(randomColor(rng), mask);
    }

}

void ConnectedComponentsStats(Mat img)
{
    // Use connected components with stats
    Mat labels, stats, centroids;
    int threshold = 1000;//检测物体大小阈值
    /*
    connectedComponentsWithStats(image, labels, stats, centroids, connectivity=8, ltype=CV_32S)
    返回整型的检测到的分类个数，label 0 表示背景
    */
    int num_objects = connectedComponentsWithStats(img, labels, stats, centroids);
    // Check the number of objects detected
    if (num_objects < 2){
        cout << "No objects detected" << endl;
        return;
    }
    else{
        cout << "Number of objects detected: " << num_objects - 1 << endl;
    }
    // Create output image coloring the objects and show area
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    RNG rng(0xFFFFFFFF);
    for (int i = 1; i<num_objects; i++){
        if (stats.at<int>(i, CC_STAT_AREA) > threshold)
        {
            cout << "Object " << i << " with pos: " << centroids.at<Point2d>(i)
                 << " with area " << stats.at<int>(i, CC_STAT_AREA) << endl;
            Mat mask = labels == i;
            output.setTo(randomColor(rng), mask);
            // draw text with area
            stringstream ss;
            ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
            putText(output, ss.str(), centroids.at<Point2d>(i), FONT_HERSHEY_SIMPLEX, 0.4,Scalar(255, 255, 255));
        }
    }

}

void FindContoursBasic(Mat img)
{
    vector<vector<Point> > contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat output = Mat::zeros(img.rows, img.cols, CV_8UC3);
    // Check the number of objects detected
    if (contours.size() == 0){
        cout << "No objects detected" << endl;
        return;
    }
    else{
        cout << "Number of objects detected: " << contours.size() << endl;
    }
    RNG rng(0xFFFFFFFF);
    for (int i = 0; i < contours.size(); i++)
        drawContours(output, contours, i, randomColor(rng));

}

///////////////////////////////////////////////
void performSUACE(Mat & src, Mat & dst, int distance=20, double sigma=7);
void performSUACE(Mat & src, Mat & dst, int distance, double sigma) {

    CV_Assert(src.type() == CV_8UC1);
    if (!(distance > 0 && sigma > 0)) {
        CV_Error(CV_StsBadArg, "distance and sigma must be greater 0");
    }
    dst = Mat(src.size(), CV_8UC1);
    Mat smoothed;
    int val;
    int a, b;
    int adjuster;
    int half_distance = distance / 2;
    double distance_d = distance;

    GaussianBlur(src, smoothed, cv::Size(0, 0), sigma);

    for (int x = 0;x<src.cols;x++)
        for (int y = 0;y < src.rows;y++) {
            val = src.at<uchar>(y, x);
            adjuster = smoothed.at<uchar>(y, x);
            if ((val - adjuster) > distance_d)adjuster += (val - adjuster)*0.5;
            adjuster = adjuster < half_distance ? half_distance : adjuster;
            b = adjuster + half_distance;
            b = b > 255 ? 255 : b;
            a = b - distance;
            a = a < 0 ? 0 : a;

            if (val >= a && val <= b)
            {
                dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
            }
            else if (val < a) {
                dst.at<uchar>(y, x) = 0;
            }
            else if (val > b) {
                dst.at<uchar>(y, x) = 255;
            }
        }
}
class WatershedSegmenter{
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

vector<Point> FindBigestContour(Mat src){
    int imax = 0;
    int imaxcontour = -1;
    std::vector<std::vector<Point> >contours;
    findContours(src,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int i=0;i<contours.size();i++){
        int itmp =  contourArea(contours[i]);
        if (imaxcontour < itmp ){
            imax = i;
            imaxcontour = itmp;
        }
    }
    return contours[imax];
}
Mat exc(Mat img1){
    Mat img3=img1;
    cv::cvtColor(img1 , img1 , CV_RGBA2RGB);
    Mat img2;
    //  GaussianBlur(img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT);



    const int channels[] = {0, 1, 2};
    const int histSize[] = {32, 32, 32};
    const float rgbRange[] = {0, 256};
    const float* ranges[] = {rgbRange, rgbRange, rgbRange};

    Mat hist;
    Mat im32fc3, backpr32f, backpr8u, backprBw, kernel;


    img1.convertTo(im32fc3, CV_32FC3);
    calcHist(&im32fc3, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    calcBackProject(&im32fc3, 1, channels, hist, backpr32f, ranges);
    Mat img4,edges;
    double minval, maxval;
    minMaxIdx(backpr32f, &minval, &maxval);
    /*  cv::adaptiveThreshold(gray, gray ,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 6);
      double otsu_thresh_val = cv::threshold(
              gray ,img4, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
      double high_thresh_val  = otsu_thresh_val,
              lower_thresh_val = otsu_thresh_val * 2;
      cv::Canny( gray,edges, lower_thresh_val,high_thresh_val );
    Mat floodFilled = cv::Mat::zeros(edges.rows+2, edges.cols+2, CV_8U);
    floodFill(edges, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, edges.cols-2, edges.rows-2)).copyTo(temp);
    floodFilled = temp;
    Mat kernel1 = Mat(11,11, CV_8UC1, cv::Scalar(1));
      Mat ed1,img,diff;
      morphologyEx(temp,temp, CV_MOP_CLOSE, kernel,Point(-1,-1),1);*/
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
    grabCut(img1, mask, Rect(), bgdModel, fgdModel,GC_INIT_WITH_MASK);

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

    img1.copyTo(img3,largestContour);
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
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img3);
    // cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(img3, img3, img2, mask);
    img2.convertTo(dest,CV_8U);
    return  img2;
}

Mat canny(Mat src)
{
    Mat detected_edges;

    int edgeThresh = 1;
    int lowThreshold = 250;
    int highThreshold = 750;
    int kernel_size = 5;
    Canny(src, detected_edges, lowThreshold, highThreshold, kernel_size);

    return detected_edges;
}
Mat sobel(Mat gray){
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
void fillEdgeImage(cv::Mat edgesIn, cv::Mat& filledEdgesOut)
{
    cv::Mat edgesNeg = edgesIn.clone();

    cv::floodFill(edgesNeg, cv::Point(0,0), CV_RGB(255,255,255));
    bitwise_not(edgesNeg, edgesNeg);
    filledEdgesOut = (edgesNeg | edgesIn);

    return;
}
vector<Point> contoursConvexHull( vector<vector<Point> > contours )
{
    vector<Point> result;
    vector<Point> pts;
    for ( size_t i = 0; i< contours.size(); i++)
        for ( size_t j = 0; j< contours[i].size(); j++)
            pts.push_back(contours[i][j]);
    convexHull( pts, result );
    return result;
}
//remove Light difference by using top hat
Mat moveLightDiff(Mat src,int radius){
    Mat dst;
    Mat srcclone = src.clone();
    Mat mask = Mat::zeros(radius*2,radius*2,CV_8U);
    circle(mask,Point(radius,radius),radius,Scalar(255),-1);
    //top hat
    erode(srcclone,srcclone,mask);
    dilate(srcclone,srcclone,mask);
    dst =  src - srcclone;
    return dst;
}

/////////////////removerbackgroundalgo/////////////

void firstremover(Mat img1 , Mat img2)
{  Mat src_hsv;
    Mat bin;
    Mat src_h;

    cvtColor(img1,src_hsv,COLOR_BGR2HSV);
    vector<Mat> rgb_planes;
    split(src_hsv, rgb_planes );
    src_h = rgb_planes[0]; // h channel is useful

    src_h = moveLightDiff(src_h,40);
    medianBlur( src_h,  src_h, 5);
    //cvtColor(img1,src_gray,COLOR_BGR2HSV);
    //find and draw the biggest contour
    cv::adaptiveThreshold(src_h, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    // threshold(src_h,bin,100,255,THRESH_OTSU);
    // cv::bitwise_or( src_gray,bin, res );
    //  Mat holes ;
    // fillEdgeImage(bin, holes);
    Canny( bin,img2, THRESH_OTSU, THRESH_OTSU*2);
}
///////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////
void secondRemover (Mat src,Mat img1,Mat img2)
{
    Mat small, bordered;

    resize(src,small, Size(), .25, .25);
    // add a zero border
    int b = 20;
    copyMakeBorder(small, bordered, b, b, b, b, BORDER_CONSTANT, Scalar(0));
    // close
    for (int i = 1; i < 15; i++)
    {
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(2*i+1, 2*i+1));
        morphologyEx(bordered, bordered, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    }
    // remove border
    Mat mask = bordered(Rect(b, b, small.cols, small.rows));
    // resize the mask
    Mat largeMask;
    resize(mask, largeMask, Size(img1.cols, img1.rows));
    // the foreground
    Mat fg;
    img1.copyTo(img2, largeMask);

}
////////////////////////////////////////////////////////////////////

void thirdRemover(Mat img1, Mat img2){

//1. Remove Shadows
//Convert to HSV
    Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

//
//2. Convert to gray and normalize
    Mat gray(rgbImg.rows, img1.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    //avoid the influence of  high frequency noise and very low noise .And on the other hand ,it make image data satisfy  nomal distribution
    //is not only to remove noise but at the same time to bring the image into a range of intensity values that is 'normal'...(meaning statistically it follows a normal distribution as far as possible),physically less stressful to our (visual) sense. The mean value will depend on the actual intensity distribution in the image...but the aim will be to ultimately state this mean with high confidence level
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);


//3. Edge detector
    GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    Mat edges;
    bool useCanny = false;

//edges = canny(gray);

//Use Sobel filter and thresholding.
    edges = sobel(gray);
//Automatic thresholding
    threshold(edges, edges, 0, 255, cv::THRESH_OTSU);





//4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);

//5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

//7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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
    rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);



    img1.copyTo(img2, largestContour);




}
/////////////////////////////////////////////////

void foruthremover(Mat img1,Mat img2){
    cvtColor(img1,img1,COLOR_BGRA2BGR);
    cv::Mat blank(img1.size(),CV_8U,cv::Scalar(0xFF));
    cv::Mat dest;


    // Create markers image
    cv::Mat markers(img1.size(),CV_8U,cv::Scalar(-1));
    //Rect(topleftcornerX, topleftcornerY, width, height);
    //top rectangle
    markers(Rect(0,0,img1.cols, 5)) = Scalar::all(1);
    //bottom rectangle
    markers(Rect(0,img1.rows-5,img1.cols, 5)) = Scalar::all(1);
    //left rectangle
    markers(Rect(0,0,5,img1.rows)) = Scalar::all(1);
    //right rectangle
    markers(Rect(img1.cols-5,0,5,img1.rows)) = Scalar::all(1);
    //centre rectangle
    int centreW = img1.cols/4;
    int centreH = img1.rows/4;
    markers(Rect((img1.cols/2)-(centreW/2),(img1.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
    markers.convertTo(markers,CV_BGR2GRAY);


    //Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img1);
    cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(img1, img1, img2, mask);
    img2.convertTo(dest,CV_8U);


}

////
void joinEdges(Mat input) {
    int width = input.cols;
    int height = input.rows;
    const int dx[] = {1, 1, 0, -1, -1, -1, 0, +1, 1};
    const int dy[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
    const int d2x[] = {2, 2, 2, 1, 0, -1, -2, -2, -2, -2, -2, -1, 0, 1, 2, 2};
    const int d2y[] = {0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2, -2, -1};
    const int d3x[] = {3, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -3, -3, -3, -3, -2, -1, 0, 1, 2, 3,
                       3, 3};
    const int d3y[] = {0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -3, -3, -3, -3,
                       -2, -1};

    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++) {

            uchar one_step, two_step, three_step;
            vector<int> connection;
            if (input.at<uchar>(i, j) != 0) {
                for (int dir = 0; dir < 8; ++dir) {
                    int tmp_x = j + dx[dir];
                    int tmp_y = i + dy[dir];
                    if (tmp_x < 0 || tmp_x >= width || tmp_y < 0 || tmp_y >= height)
                        one_step = 0;//treat outside image as 0
                    else one_step = input.at<uchar>(tmp_y, tmp_x);

                    if (one_step != 0) connection.push_back(dir);
                }

                if (connection.size() < 3 && connection.size() != 0) {
                    int direction1 = connection.front();
                    int direction2 = connection.back();
                    if (direction2 - direction1 <= 1 || direction2 - direction1 == 7) {
                        int start_direction = (direction2 + 2) * 3;
                        int end_direction = (direction1 + 6) * 3;
                        for (int dir2 = start_direction;
                             dir2 < end_direction; dir2++) {//dir2 is between 6~39
                            int tmp_1x = j + dx[dir2 % 24 / 3];
                            int tmp_1y = i + dy[dir2 % 24 / 3];
                            int tmp_2x = j + d2x[dir2 % 24 * 2 / 3];
                            int tmp_2y = i + d2y[dir2 % 24 * 2 / 3];
                            int tmp_3x = j + d3x[dir2 % 24];
                            int tmp_3y = i + d3y[dir2 % 24];
                            if (tmp_2x < 0 || tmp_2x >= width || tmp_2y < 0 ||
                                tmp_2y >= height)
                                two_step = 0;//treat outside image as 0
                            else two_step = input.at<uchar>(tmp_2y, tmp_2x);
                            if (tmp_3x < 0 || tmp_3x >= width || tmp_3y < 0 ||
                                tmp_3y >= height)
                                three_step = 0;//treat outside image as 0
                            else three_step = input.at<uchar>(tmp_3y, tmp_3x);
                            if (two_step != 0) {
                                input.at<uchar>(tmp_1y, tmp_1x) = 255;
                                break;
                            } else if (three_step != 0) {
                                input.at<uchar>(tmp_1y, tmp_1x) = 255;
                                input.at<uchar>(tmp_2y, tmp_2x) = 255;
                                break;

                            }
                        }
                    }
                }
            }
        }
}
/////////////////////////////////////////////
void glcm(Mat &img)
{
    float energy=0,contrast=0,homogenity=0,IDM=0,entropy=0,mean1=0;
    int row=img.rows,col=img.cols;
    Mat gl=Mat::zeros(256,256,CV_32FC1);

    //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
    for(int i=0;i<row;i++)
        for(int j=0;j<col-1;j++)
            gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))=gl.at<float>(img.at<uchar>(i,j),img.at<uchar>(i,j+1))+1;

    // normalizing glcm matrix for parameter determination
    gl=gl+gl.t();
    gl=gl/sum(gl)[0];


    for(int i=0;i<256;i++)
        for(int j=0;j<256;j++)
        {
            energy=energy+gl.at<float>(i,j)*gl.at<float>(i,j);            //finding parameters
            contrast=contrast+(i-j)*(i-j)*gl.at<float>(i,j);
            homogenity=homogenity+gl.at<float>(i,j)/(1+abs(i-j));
            if(i!=j)
                IDM=IDM+gl.at<float>(i,j)/((i-j)*(i-j));                      //Taking k=2;
            if(gl.at<float>(i,j)!=0)
                entropy=entropy-gl.at<float>(i,j)*log10(gl.at<float>(i,j));
            mean1=mean1+0.5*(i*gl.at<float>(i,j)+j*gl.at<float>(i,j));
        }
    cout<<"energy="<<energy<<endl;
    cout<<"contrast="<<contrast<<endl;
    cout<<"homogenity="<<homogenity<<endl;
    cout<<"IDM="<<IDM<<endl;
    cout<<"entropy="<<entropy<<endl;
    cout<<"mean="<<mean1<<endl;
}
const static int VIEW_MODE_ORB = 2;

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_myshite_App_CompareUtil_nativeComparePSNR(JNIEnv *env, jobject /* this */,
                                                           jlong addr1, jlong addr2) {
    Mat &I1 = *(Mat *) addr1;
    Mat &I2 = *(Mat *) addr2;

    cvtColor(I1, I1, CV_BGR2GRAY);
    cvtColor(I2, I2, CV_BGR2GRAY);

    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);        // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else {
        double mse = sse / (double) (I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_myshite_App_CompareUtil_nativeCompareSSIM(JNIEnv *env, jobject /* this */,
                                                           jlong addr1, jlong addr2) {
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat &i1 = *(Mat *) addr1;
    Mat &i2 = *(Mat *) addr2;

    cvtColor(i1, i1, CV_BGR2GRAY);
    cvtColor(i2, i2, CV_BGR2GRAY);

    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);        // I2^2
    Mat I1_2 = I1.mul(I1);        // I1^2
    Mat I1_I2 = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map

    return mssim.val[0] + mssim.val[1] + mssim.val[2];
}
cv::Mat findBiggestBlob( cv::Mat &inputImage ){

    cv::Mat biggestBlob = inputImage.clone();

    int largest_area = 0;
    int largest_contour_index=0;

    std::vector< std::vector<cv::Point> > contours; // Vector for storing contour
    std::vector<cv::Vec4i> hierarchy;

    // Find the contours in the image
    cv::findContours( biggestBlob, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

    for( int i = 0; i< (int)contours.size(); i++ ) {

        //Find the area of the contour
        double a = cv::contourArea( contours[i],false);
        //Store the index of largest contour:
        if( a > largest_area ){
            largest_area = a;
            largest_contour_index = i;
        }

    }

    //Once you get the biggest blob, paint it black:
    cv::Mat tempMat = biggestBlob.clone();
    cv::drawContours( tempMat, contours, largest_contour_index, cv::Scalar(0),
                      CV_FILLED, 8, hierarchy );

    //Erase the smaller blobs:
    biggestBlob = biggestBlob - tempMat;
    tempMat.release();
    return biggestBlob;
}
jlong calHammingDistance(Mat srcMat) {
    Mat dstMat;
    resize(srcMat, dstMat, Size(8, 8), 0, 0, INTER_CUBIC);
    cvtColor(dstMat, dstMat, CV_BGR2GRAY);

    int iAvg = 0;
    int arr[64];
    for (int i = 0; i < 8; i++) {
        uchar *data1 = dstMat.ptr<uchar>(i);
        int tmp = i * 8;
        for (int j = 0; j < 8; j++) {
            int tmp1 = tmp + j;
            arr[tmp1] = data1[j] / 4 * 4;
            iAvg += arr[tmp1];
        }
    }
    iAvg /= 64;

    int p = 1;
    jlong value = 0;
    for (int i = 0; i < 64; i++) {
        p *= 2;
        if (arr[i] >= iAvg) {
            value += p;
        }
    }
    return value;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_myshite_App_CompareUtil_nativeComparePH(JNIEnv *env, jclass, jlong addr1,
                                                         jlong addr2) {
    Mat &mat1 = *(Mat *) addr1;
    Mat &mat2 = *(Mat *) addr2;
    if (!mat1.data || !mat2.data) {
        return 0;
    }

    jlong distance1 = calHammingDistance(mat1);
    jlong distance2 = calHammingDistance(mat2);
    return distance1 - distance2;
}
/*extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_GLCM(JNIEnv *env, jobject thiz, jlong input_image) {
// TODO: implement GLCM()


    Mat &img = *(Mat *)input_image;
    cvtColor(img, img, COLOR_RGB2GRAY);
    glcm(img);

    Mat img_higher_contrast;
    img.convertTo(img_higher_contrast, -1, 2, 0); //increase the contrast (double)

    Mat img_lower_contrast;
    img.convertTo(img_lower_contrast, -1, 0.5, 0); //decrease the contrast (halve)

    Mat img_higher_brightness;
    img.convertTo(img_higher_brightness, -1, 1, 20); //increase the brightness by 20 for each pixel

    Mat img_lower_brightness;
    img.convertTo(img_lower_brightness, -1, 1, -20); //decrease the brightness by 20 for each pixel



}
*/

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_matchshape(JNIEnv *env, jobject thiz, jlong input_image,
                                                      jlong output_image) {
    // TODO: implement matchshape()
    Mat &img1= *(Mat *)  input_image;

    Mat &img2 = *(Mat *) output_image;
    Mat src_hsv;
    Mat bin;
    Mat src_h;

    cvtColor(img1,src_hsv,COLOR_BGR2HSV);
    vector<Mat> rgb_planes;
    split(src_hsv, rgb_planes );
    src_h = rgb_planes[0]; // h channel is useful

    src_h = moveLightDiff(src_h,40);
    GaussianBlur(src_h, src_h, Size(5,5), 0, 0, BORDER_DEFAULT);
    //cvtColor(img1,src_gray,COLOR_BGR2HSV);
    //find and draw the biggest contour
    cv::adaptiveThreshold(src_h, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    // threshold(src_h,bin,100,255,THRESH_OTSU);
    // cv::bitwise_or( src_gray,bin, res );
    //  Mat holes ;
    // fillEdgeImage(bin, holes);

    bin=sobel(bin);
    threshold(bin, bin, 0, 255, cv::THRESH_BINARY+THRESH_OTSU);
    // Canny( bin, img2, THRESH_OTSU, THRESH_OTSU*2);
    //4. Dilate
    Mat dilateGrad = bin;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( bin, contours, hierarchy,
                  CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    for( ; idx >= 0; idx = hierarchy[idx][0] )
    {
        Scalar color( rand()&255, rand()&255, rand()&255 );
        drawContours( largestContour, contours, idx, color, CV_FILLED, 8, hierarchy );

    }

    img1.copyTo(img2, largestContour);
    //  dilate(bin, dilateGrad, elementDilate);
    //fillEdgeImage(bin,img2);
    // Combine the two images to get the foreground.

    // img2 = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    //floodFill(dilateGrad, img2, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    /*  floodFilled = cv::Scalar::all(255) - floodFilled;
      Mat temp;
      floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
      floodFilled = temp;*/


//6. Erode
    /*  int erosionType = MORPH_ELLIPSE;
      int erosionSize = 4;
      Mat erosionElement = getStructuringElement(erosionType,
                                                 Size(2*erosionSize+1, 2*erosionSize+1),
                                                 Point(erosionSize, erosionSize));
      erode(floodFilled, img2, erosionElement);*/

//7. Find largest contour
/*
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
*/
////////////mine
    //   vector<vector<Point> > contours;
    // RNG rng(12345);
    //  vector<Vec4i> hierarchy;
    //  findContours(floodFilled, contours, hierarchy,  RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    /*   for(int i=0; i<contours.size(); i++)
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
       rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);

   */
//////////////////
    /* cv::Moments mom = cv::moments(contours[0]);

     Mat drawing = Mat::zeros(img2.size(), CV_8UC3);
     vector<Point2f> mc(contours.size());
     vector<Moments> mu(contours.size());



     for (int i = 0; i < contours.size(); i++)
     {
         mu[i] = moments(contours[i], false);
     }

     ///  Get the mass centers:


     for (int i = 0; i < contours.size(); i++)
     {
         mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
     }


     Rect bounding_rect;
     int largest_area=0;
     int largest_contour_index=-1;
     Scalar color = Scalar(167,151,0);
     for( size_t i = 0; i< contours.size(); i++ ) // iterate through each contour.
     {
         int area = contourArea( contours[i] );  //  Find the area of contour

         if( area > largest_area )
         {
             largest_area = area;
             largest_contour_index = i;               //Store the index of largest contour
             //bounding_rect = boundingRect( contours[i] ); // Find the bounding rectangle for biggest contour
             //circle( dst, mc[largest_contour_index], 4, color, -1, 8, 0 );

         }

     }*/
    ///////////
    // bounding_rect = boundingRect(contours[largest_contour_index]);
    // Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    // drawContours(dst, contours, largest_contour_index, color);
    /*
    drawContours( dst, contours, largest_contour_index, color, 2 );
    circle(dst, mc[largest_contour_index], 4, color, -1, 8, 0);*/
    /*  for(int i = 0; i < contours.size(); i++) // Iterate through each contour
      {
          if (i != largest_contour_index) {
            //  approxPolyDP( Mat(contours[i]), contours[i], 3, true );
             //boundRect[i] = boundingRect( Mat(contours[i]) );
            //  drawContours(dst, contours, i, color, CV_FILLED, 8, hierarchy);
          }
      }*/

    ////////
    /*  Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
      Scalar color2(255,0,0);
      drawContours(largestContour, contours,-1, color2, CV_FILLED); // Draw the largest contour using previously stored index.
      rectangle(img1, bounding_rect, Scalar(0,255,0), 1, 8, 0);


      img1.copyTo(img2, largestContour);*/
    ///////////////
/*
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    img2=temp;

    Mat edgesNeg=dilateGrad.clone();
    floodFill(edgesNeg,Point(0,0),CV_RGB(255,255,255));


    bitwise_not(edgesNeg,edgesNeg);


    img2=(edgesNeg|dilateGrad);*/
/*
    cvtColor( img1, img1, COLOR_RGB2GRAY);

    medianBlur(img1, img1, 5);
   cv::adaptiveThreshold(img1 ,img2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

// Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform

    Moments moments = cv::moments(img2, false);

// Calculate Hu Moments
    double huMoments[7];
    HuMoments(moments, huMoments);
    for(int i = 0; i < 7; i++)
    {
        huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));

    }

*/

    /*  Mat src_hsv;
      Mat bin,res;
      Mat src_h,src_gray ;
     // medianBlur(dst, dst, 5);
      cvtColor(img1,src_hsv,COLOR_BGR2HSV);

      vector<Mat> rgb_planes;
      split(src_hsv, rgb_planes );
      src_h = rgb_planes[0]; // h channel is useful

      src_h = moveLightDiff(src_h,40);

      //cvtColor(img1,src_gray,COLOR_BGR2HSV);
      //find and draw the biggest contour
    cv::adaptiveThreshold(src_h, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
     // threshold(src_h,bin,100,255,THRESH_OTSU);
     // cv::bitwise_or( src_gray,bin, res );
    //  Mat holes ;
     // fillEdgeImage(bin, holes);
     Canny( bin,res, THRESH_OTSU, THRESH_OTSU*2);

      vector<Point> bigestcontrour =  FindBigestContour(res);

     /*  vector<vector<Point> > controus;
       controus.push_back(bigestcontrour);
       cv::drawContours(bin,controus,0,Scalar(0,0,255),3);
      Mat small, bordered;

       resize(res, small, Size(), .25, .25);
       // add a zero border
       int b = 20;
       copyMakeBorder(small, bordered, b, b, b, b, BORDER_CONSTANT, Scalar(0));
       // close
       for (int i = 1; i < 15; i++)
       {
           Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(2*i+1, 2*i+1));
           morphologyEx(bordered, bordered, MORPH_CLOSE, kernel, Point(-1, -1), 1);
       }
       // remove border
       Mat mask = bordered(Rect(b, b, small.cols, small.rows));
       // resize the mask
       Mat largeMask;
       resize(mask, largeMask, Size(img1.cols, img1.rows));
       // the foreground
       Mat fg;
       img1.copyTo(img2, largeMask);

  //1. Remove Shadows
   //Convert to HSV
      Mat hsvImg;
      cvtColor(img1, hsvImg, CV_BGR2HSV);
     Mat channel[3];
      split(hsvImg, channel);
      channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
      //Merge channels
      merge(channel, 3, hsvImg);
      Mat rgbImg;
      cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

      //2. Convert to gray and normalize
      Mat gray( rgbImg.rows, img1.cols, CV_8UC1);
     cvtColor(rgbImg, gray, CV_BGR2GRAY);
      normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);

      //3. Edge detector
      GaussianBlur(gray, gray, Size(5,5), 0, 0, BORDER_DEFAULT);
      Mat edges;
      int edgeThresh = 1;
      int lowThreshold = 250;
      int highThreshold = 750;
      int kernel_size = 5;

  //    Canny(hsvImg, edges, lowThreshold, highThreshold, kernel_size);

      edges = sobel(gray);
      //Automatic thresholding
      //
      //Manual thresholding
   //  threshold(edges, edges, 45, 255, THRESH_OTSU);


      //4. Dilate
      Mat dilateGrad = edges;
      int dilateType = MORPH_ELLIPSE;
      int dilateSize = 3;
      Mat elementDilate = getStructuringElement(dilateType,
                                                Size(2*dilateSize + 1, 2*dilateSize+1),
                                                Point(dilateSize, dilateSize));
      dilate(edges, dilateGrad, elementDilate);

      // Combine the two images to get the foreground.

      //6. Erode
      int erosionType = MORPH_ELLIPSE;
      int erosionSize = 4;
      Mat erosionElement = getStructuringElement(erosionType,
                                                 Size(2*erosionSize+1, 2*erosionSize+1),
                                                 Point(erosionSize, erosionSize));
      erode(dilateGrad,dilateGrad, erosionElement);*/
    /*img2=dilateGrad.clone();

    vector<vector<Point>> contours;

    findContours(img2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    drawContours(img2, contours, -1, Scalar(255), CV_FILLED);


    Mat thresh;
    threshold(dilateGrad,thresh,220,255,THRESH_BINARY+THRESH_OTSU);

    vector<Point> bigestcontrour =  FindBigestContour(thresh);

    Mat edgesNeg=thresh.clone();
    floodFill(edgesNeg,Point(0,0),CV_RGB(255,255,255));


    bitwise_not(edgesNeg,edgesNeg);
Mat res ;

    img2=(edgesNeg|thresh);*/
    //img1.copyTo(img2, dilateGrad);
/*
    cv::Mat holes=img2.clone();
    cv::floodFill(holes,cv::Point2i(0,0),cv::Scalar(1));
    for(int i=0;i<img2.rows*img2.cols;i++)
    {
        if(holes.data[i]==0)
            img2.data[i]=1;
    }*/

/*
    //5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
  img2 = temp;

   // fillEdgeImage(dilateGrad, img2);


    //6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);


    //7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = -1;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

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
    rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);


    //8. Mask original image
    Mat maskedSrc;
    img1.copyTo(img2, dilateGrad);
/*new part
cvtColor(img1,img1,COLOR_BGRA2BGR);
    cv::Mat blank(img1.size(),CV_8U,cv::Scalar(0xFF));
    cv::Mat dest;


    // Create markers image
    cv::Mat markers(img1.size(),CV_8U,cv::Scalar(-1));
    //Rect(topleftcornerX, topleftcornerY, width, height);
    //top rectangle
    markers(Rect(0,0,img1.cols, 5)) = Scalar::all(1);
    //bottom rectangle
    markers(Rect(0,img1.rows-5,img1.cols, 5)) = Scalar::all(1);
    //left rectangle
    markers(Rect(0,0,5,img1.rows)) = Scalar::all(1);
    //right rectangle
    markers(Rect(img1.cols-5,0,5,img1.rows)) = Scalar::all(1);
    //centre rectangle
    int centreW = img1.cols/4;
    int centreH = img1.rows/4;
    markers(Rect((img1.cols/2)-(centreW/2),(img1.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
    markers.convertTo(markers,CV_BGR2GRAY);


    //Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img1);
    cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(img1, img1, img2, mask);
   img2.convertTo(dest,CV_8U);*/



}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_segmentation(JNIEnv *env, jobject thiz,
                                                        jlong input_image, jlong output_image) {
    // TODO: implement segmentation()
    Mat &img1 = *(Mat *) input_image;

    Mat &img2 = *(Mat *) output_image;
    // cvtColor(src,src,COLOR_BGRA2BGR);
    /* vector<vector<Point>>();
     RNG rng(12345);
     vector<Vec4i> hierarchy;
     //1. Remove Shadows
     Mat hsvImg;
     cvtColor(img1, hsvImg, CV_BGR2HSV);
     Mat channel[3];
     split(hsvImg, channel);
     channel[0] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
     hsvImg = moveLightDiff(hsvImg,40);
     //Merge channels
     merge(channel, 3, hsvImg);
     Mat rgbImg;
     Mat bin;
     cvtColor(hsvImg, bin, CV_HSV2BGR);
    bin=sobel(bin);

     cv::adaptiveThreshold(bin,bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
     vector<vector<Point>> contours;

     findContours(bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);*/

    //////////////////secondidea/////////////////

    /*  int distance=20;double sigma=7;
      cvtColor(src,src,CV_BGR2GRAY);
      CV_Assert(src.type() == CV_8UC1);
      if (!(distance > 0 && sigma > 0)) {
          CV_Error(CV_StsBadArg, "distance and sigma must be greater 0");
      }
     Mat  dst = Mat(src.size(), CV_8UC1);
      Mat smoothed;
      int val;
      int a, b;
      int adjuster;
      int half_distance = distance / 2;
      double distance_d = distance;

      GaussianBlur(src, smoothed, cv::Size(0, 0), sigma);

      for (int x = 0;x<src.cols;x++)
          for (int y = 0;y < src.rows;y++) {
              val = src.at<uchar>(y, x);
              adjuster = smoothed.at<uchar>(y, x);
              if ((val - adjuster) > distance_d)adjuster += (val - adjuster)*0.5;
              adjuster = adjuster < half_distance ? half_distance : adjuster;
              b = adjuster + half_distance;
              b = b > 255 ? 255 : b;
              a = b - distance;
              a = a < 0 ? 0 : a;

              if (val >= a && val <= b)
              {
                  dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
              }
              else if (val < a) {
                  dst.at<uchar>(y, x) = 0;
              }
              else if (val > b) {
                  dst.at<uchar>(y, x) = 255;
              }
          }
     */
    //////////////////////thirdideaaa 20%
    /*
    Mat img1;
     cvtColor(src,img1,CV_BGR2GRAY);
     medianBlur(img1, img1, 3);

     //计算光图像或背景
     Mat pattern = calculateLightPattern(img1);


     //去除背景Removing the background using the light pattern for segmentation
     Mat removedPattern0 = removeLight(img1, pattern, 0);
     Mat removedPattern1 = removeLight(img1, pattern, 1);

     // 全局二值化The thresholding operation
     int th = 30;//阈值

     threshold(removedPattern0, removedPattern0, th, 255, CV_THRESH_BINARY);

     Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
     //分割图像Segmenting our input image
     //ConnectedComponentsStats(removedPattern0);
     //ConnectedComponentsStats(removedPattern1);
     FindContoursBasic(removedPattern0);//找出物体轮廓
     //ConnectedComponents(removedPattern0);

     src.copyTo(img2,removedPattern0);*/
///////////////////////////failed try
/*
    Mat gray, thresh;
    Mat img = imread("coins.jpg");
    cvtColor(src,src, COLOR_BGRA2BGR);
    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV+CV_THRESH_OTSU);

    Mat opening; Mat sure_bg;
    Mat sure_fg; Mat unknow;
    Mat dist_transform;
    double maxValue;
// noise removal
    Mat kernel = Mat::ones(3, 3, CV_8U);
    morphologyEx(thresh, opening, MORPH_OPEN, kernel);

// sure background area
    dilate(opening, sure_bg, kernel, Point(-1, -1), 3);

// Finding sure foreground area
    distanceTransform(opening, dist_transform, DIST_L2, 5);
    minMaxLoc(dist_transform, 0, &maxValue, 0, 0);
    threshold(dist_transform, sure_fg, 0.7*maxValue, 255, 0);

// Finding unknown region
    sure_fg.convertTo(sure_fg, CV_8U);
    subtract(sure_bg, sure_fg, unknow);

    Mat markers;
    connectedComponents(sure_fg, markers);

// Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1;

// Now, mark the region of unknown with zero
    markers.setTo(0, unknow);
    Mat marker;
    Mat mask;
    markers.convertTo(markers,CV_32SC1);
    watershed(src, markers);
    compare(markers, -1, img2, CMP_EQ);
    //src.setTo(Scalar(0, 0, 255), mask);*/
//////////////////////////////////////
    /*   cv::Mat blank(src.size(),CV_8U,cv::Scalar(0xFF));
       cv::Mat dest;


       // Create markers image
       cv::Mat markers(src.size(),CV_8U,cv::Scalar(-1));
       //Rect(topleftcornerX, topleftcornerY, width, height);
       //top rectangle
       markers(Rect(0,0,src.cols, 5)) = Scalar::all(1);
       //bottom rectangle
       markers(Rect(0,src.rows-5,src.cols, 5)) = Scalar::all(1);
       //left rectangle
       markers(Rect(0,0,5,src.rows)) = Scalar::all(1);
       //right rectangle
       markers(Rect(src.cols-5,0,5,src.rows)) = Scalar::all(1);
       //centre rectangle
       int centreW = src.cols/4;
       int centreH = src.rows/4;
       markers(Rect((src.cols/2)-(centreW/2),(src.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
       markers.convertTo(markers,CV_BGR2GRAY);


       //Create watershed segmentation object
       WatershedSegmenter segmenter;
       segmenter.setMarkers(markers);
       cv::Mat wshedMask = segmenter.process(src);
       cv::Mat mask;
       convertScaleAbs(wshedMask, mask, 1, 0);
       double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
       bitwise_and(src, src, img2, mask);
      img2.convertTo(img2,CV_8U);*/
    ////////////////////////////////////////////////////////////////////////////////////////////
    /*Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
    //Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

    //2. Convert to gray and normalize
    Mat gray( rgbImg.rows, img1.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);

    //3. Edge detector
    GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    Mat edges;
    int edgeThresh = 1;
    int lowThreshold = 250;
    int highThreshold = 750;
    int kernel_size = 5;

//    Canny(hsvImg, edges, lowThreshold, highThreshold, kernel_size);

    edges = sobel(gray);
    //Automatic thresholding
    //
    //Manual thresholding
    threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
    //4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);


    //5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    img2 = temp;*/

    ///////////////////////////////////////////////essai1  07/06/2020 15.12
    Mat hsvImg;
    cvtColor(img1, img1, COLOR_BGRA2BGR);

    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

    Mat hsv;
    pyrMeanShiftFiltering(rgbImg, rgbImg, 10, 35, 3);
    medianBlur(rgbImg, rgbImg, 5);
//2. Convert to gray and normalize
    Mat gray(rgbImg.rows, rgbImg.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat dilate_img;

    Mat m = Mat(3, 3, CV_8UC1, cv::Scalar(1));
    // dilate(gray, dilate_img, m, Point(-1, -1), 2, 1, 1);
    // erode(dilate_img, dilate_img, getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat img3;

    cv::adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 5);
    double otsu_thresh_val = cv::threshold(
            dilate_img, img3, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
    );
    double high_thresh_val = otsu_thresh_val,
            lower_thresh_val = otsu_thresh_val * 0.25;
    cv::Canny(gray, dilate_img, lower_thresh_val, high_thresh_val);
    Mat kernel = Mat(3, 3, CV_8UC1, cv::Scalar(1));
    Mat ed1;
    morphologyEx(dilate_img, img2, CV_MOP_CLOSE, kernel, Point(-1, -1), 10);
    /* Mat floodFilled = cv::Mat::zeros(ed1.rows+2, ed1.cols+2, CV_8U);
     floodFill(ed1, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
     floodFilled = cv::Scalar::all(255) - floodFilled;
     Mat temp;
     floodFilled(Rect(1, 1, ed1.cols-2, ed1.rows-2)).copyTo(temp);
     img2= temp;

     Mat edges,image_Laplacian;
     //  edges=sobel(dilate_img);
     /* cv::Mat image_X;

      cv::Sobel(dilate_img, image_X, CV_8UC1, 1, 0);

      cv::Mat image_Y;
      // this is how we can create a vertical edge detector.
      cv::Sobel(dilate_img, image_Y, CV_8UC1, 0, 1);
      cv::Mat sobel = image_X + image_Y;
      double sobmin, sobmax;
      cv::minMaxLoc(sobel, &sobmin, &sobmax);

      cv::Mat sobelImage;
      sobel.convertTo(sobelImage, CV_8UC1, -255./sobmax, 255);



      cv::Mat image_Sobel_thresholded;
      double max_value, min_value;
      cv::minMaxLoc(sobelImage, &min_value, &max_value);
      //image_Laplacian = image_Laplacian / max_value * 255;


      cv::threshold(sobelImage, image_Sobel_thresholded, 20, 255, cv::THRESH_BINARY);
     // cv::Mat image_Laplacian;
      // here we will apply low pass filtering in order to better detect edges
      // try to uncomment this line and the result will be much poorer.
      cv::GaussianBlur(dilate_img, dilate_img, Size(5,5), 1);

      cv::Laplacian(dilate_img, image_Laplacian, CV_8UC1);


      cv::Mat image_Laplacian_thresholded;
      double max_value1, min_value1;
      cv::minMaxLoc(image_Laplacian, &min_value1, &max_value1);



      cv::threshold(sobel, image_Laplacian_thresholded, 70, 220, cv::THRESH_BINARY);*/
    // cv::Canny( dilate_img,edges, lower_thresh_val, high_thresh_val );

    /*   Mat closing,bg,dst;
       dilate(edges,closing, m, Point(-1, -1), 10, 1, 1);
       Mat kernel = Mat(3, 3, CV_8UC1, cv::Scalar(1));
       morphologyEx(closing, img2, CV_MOP_CLOSE, kernel);


       /*
      Mat floodFilled = cv::Mat::zeros(bg.rows+2, bg.cols+2, CV_8U);
       floodFill(bg, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
       floodFilled = cv::Scalar::all(255) - floodFilled;
       Mat temp;
       floodFilled(Rect(1, 1, bg.cols-2, bg.rows-2)).copyTo(temp);
       dilate(temp, img2, m, Point(-1, -1), 10, 1, 1);*/


//joinEdges(img2);
/*

//5. Floodfill
Mat floodFilled;
   fillEdgeImage(dilateGrad,floodFilled);


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

//7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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
    rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);



    img1.copyTo(img2, largestContour);

          ////////////-0
  /*  vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( closing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  Mat  drawing = Mat::zeros( closing.size(), CV_8UC3 );

    vector<Point> approxShape;
    for(size_t i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], apepproxShape, arcLength(Mat(contours[i]), true)*0.04, true);
        drawContours(drawing, contours, i, Scalar(255, 0, 0), CV_FILLED);   // fill BLUE
    }
    img2=drawing;*/
    /////////1-
    /*  vector<vector<Point>> contours;
      vector< Vec4i > hierarchy;
      findContours(closing, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

      Mat tmp = Mat::zeros(closing.size(), CV_8U);
      Mat tmp2 = Mat::zeros(closing.size(), CV_8U);

      for (size_t i = 0; i < contours.size(); i++)
          if (hierarchy[i][3]<0)
              drawContours(tmp, contours, i, Scalar(255, 255, 255), -1);

      for (size_t i = 0; i < contours.size(); i++)
          if (hierarchy[i][2]<0 && hierarchy[i][3]>-1)
              drawContours(tmp2, contours, i, Scalar(255, 255, 255), -1);
     img2= tmp ;*/
    //////////
    /////////2-
    /*  int dilateType = MORPH_ELLIPSE;
      int dilateSize = 3;
      Mat elementDilate = getStructuringElement(dilateType,
                                                Size(2*dilateSize + 1, 2*dilateSize+1),
                                                Point(dilateSize, dilateSize));
      dilate(closing, closing, elementDilate);

  fillEdgeImage(closing ,img2);
    /*vector<vector<Point> > contours;

    findContours(closing, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

    Mat drawing = Mat::zeros(closing.size(), CV_8UC3);

    for (int i = 0; i< contours.size(); i++)
    {
        Scalar color = Scalar( 255,255,255);
        drawContours( drawing, contours, i, color, 2 );
    }

    vector<Point> ConvexHullPoints =  contoursConvexHull(contours);

    polylines( drawing, ConvexHullPoints, true, Scalar(0,0,255), 2 );
    Mat dst1;
    fillEdgeImage(drawing,dst1);
    img1.copyTo(img2, drawing);*/

    /*

//3. Edge detector
   // GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    medianBlur(gray, gray, 5);
   // Mat gray;

    Mat dilateGrad = gray;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(gray, dilateGrad, elementDilate);

    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(dilateGrad, dilateGrad, erosionElement);

  //  medianBlur(dilateGrad, dilateGrad, 3);
    Mat edges;
    bool useCanny = false;

//edges = canny(gray);
    Mat img3;
//Use Sobel filter and thresholding.
    double otsu_thresh_val = cv::threshold(
            dilateGrad, img3, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
    );
    double high_thresh_val  = otsu_thresh_val,
            lower_thresh_val = otsu_thresh_val * 0.5;
    cv::Canny( dilateGrad,img2, lower_thresh_val, high_thresh_val );*/


//Use Sobel filter and thresholding.

    //////////////////////

/*


//4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);

//5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(floodFilled, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);

    double const MIN_CONTOUR_AREA(1000.0);
    for (int i(0); i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);

        if (area >= MIN_CONTOUR_AREA) {
            cv::drawContours(mask, contours, i, cv::Scalar(255, 255, 255), CV_FILLED);
        }
    }


    cv::Mat masked_object;
    cv::bitwise_and(img1, img1, img2, mask);

*/
    //////////////////////////////////////////////////////////////////////
    //resize(srcGray, srcGray, Size(), 0.25, 0.25);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_newseg(JNIEnv *env, jobject thiz, jlong input_image,
                                                  jlong output_image) {
    // TODO: implement newseg()


    Mat &img1= *(Mat *)  input_image;
    Mat img3;
    Mat &img2 = *(Mat *) output_image;
    cvtColor(img1, img1, CV_BGRA2BGR);
    Mat hsvImg;


    cvtColor(img1, hsvImg, CV_BGR2HSV);

    pyrMeanShiftFiltering( img1,img1 , 10, 35, 3);
    medianBlur(img1, img1,7);
    cvtColor(img1, img1, CV_BGR2YCrCb);


    Mat tmp;
    Mat();
    Mat Y, Cr, Cb;
    vector<Mat> channels;


    split(img1, channels);
    Y = channels.at(0);
    Cr = channels.at(1);
    Cb = channels.at(2);

    Mat result(img1.rows, img1.cols, CV_8UC1);


    for (int j = 1; j < Y.rows - 1; j++)
    {
        uchar* currentCr = Cr.ptr< uchar>(j);
        uchar* currentCb = Cb.ptr< uchar>(j);
        uchar* current = result.ptr< uchar>(j);
        for (int i = 1; i < Y.cols - 1; i++)
        {
            if ((currentCr[i] > 150) && (currentCr[i] < 175) && (currentCb[i] > 80) && (currentCb[i] < 130))
                current[i] = 255;
            else
                current[i] = 0;
        }}
    img2=result;
    ////////////


/*
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);
    Mat();
    Mat hsv;
    pyrMeanShiftFiltering( rgbImg,rgbImg , 10, 35, 3);
    medianBlur(rgbImg, rgbImg,5);
    Mat gray(rgbImg.rows, rgbImg.cols, CV_8UC1);

//2. Convert to gray and normalize

    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat dilate_img;

    Mat m = Mat(3, 3, CV_8UC1, cv::Scalar(1));
    // dilate(gray, dilate_img, m, Point(-1, -1), 2, 1, 1);
    // erode(dilate_img, dilate_img, getStructuringElement(MORPH_RECT, Size(3, 3)));

    Mat gray1;
  //  cv::adaptiveThreshold(gray, gray ,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 6);
    dilate(gray,gray,Mat(11,11,CV_8UC1));
    erode(gray,gray,Mat(11,11,CV_8UC1));
    double otsu_thresh_val = cv::threshold(
            gray ,img3, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
    double high_thresh_val  = otsu_thresh_val,
            lower_thresh_val = otsu_thresh_val * 2;
    cv::Canny( gray,dilate_img, lower_thresh_val,high_thresh_val );
    Mat kernel = Mat(11,11, CV_8UC1, cv::Scalar(1));
    Mat ed1,img,diff;
    morphologyEx(dilate_img,ed1, CV_MOP_CLOSE, kernel,Point(-1,-1),1);
//bitwise_or(ed1,result,img);
    //subtract(result, img,img2,result, -1);
  //  absdiff(img, result, diff);
    img2=ed1;
///////////////addWeighted(ed1, 0.5,result,1.0,0.0,img2,-1);
          //////////
/* int   Y_MIN  = 0;
  int  Y_MAX  = 255;
  int  Cr_MIN = 133;
   int Cr_MAX = 173;
    int Cb_MIN = 77;
   int  Cb_MAX = 127;
    Mat skin;*/

    //  cv::inRange(img1,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);

    /* Mat hsvImg;
     cvtColor(img1,img1,COLOR_BGRA2BGR);

     cvtColor(img1, hsvImg, CV_BGR2HSV);
     Mat channel[3];
     split(hsvImg, channel);
     channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
 //Merge channels
     merge(channel, 3, hsvImg);
     Mat rgbImg;
     cvtColor(hsvImg, rgbImg, CV_HSV2BGR);
     Mat();
     Mat hsv;
     pyrMeanShiftFiltering( rgbImg,rgbImg , 10, 35, 3);
     medianBlur(rgbImg, rgbImg,5);
     Mat gray(rgbImg.rows, rgbImg.cols, CV_8UC1);

 //2. Convert to gray and normalize

   cvtColor(rgbImg, gray, CV_BGR2GRAY);
     normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);

     Mat dilate_img;

     Mat m = Mat(3, 3, CV_8UC1, cv::Scalar(1));
     // dilate(gray, dilate_img, m, Point(-1, -1), 2, 1, 1);
     // erode(dilate_img, dilate_img, getStructuringElement(MORPH_RECT, Size(3, 3)));
     Mat img3;
  Mat gray1;
     cv::adaptiveThreshold(gray, gray ,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 6);
     dilate(gray,gray,Mat(11,11,CV_8UC1));
     erode(gray,gray,Mat(11,11,CV_8UC1));
     double otsu_thresh_val = cv::threshold(
             gray ,img3, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
     double high_thresh_val  = otsu_thresh_val,
             lower_thresh_val = otsu_thresh_val * 0.05;
     cv::Canny( gray,dilate_img, lower_thresh_val,high_thresh_val );
     Mat kernel = Mat(11,11, CV_8UC1, cv::Scalar(1));
     Mat ed1;
     morphologyEx(dilate_img,ed1, CV_MOP_CLOSE, kernel,Point(-1,-1),1);
     //fillEdgeImage(ed1,img2);
     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;
     findContours(ed1, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
     /// 计算矩
     vector<Moments> mu(contours.size() );
     for( int i = 0; i < contours.size(); i++ )
         mu[i] = moments( contours[i], false );
     ///  计算中心矩:
     Mat img(img1.rows, img1.cols, CV_8UC3, Scalar::all(0));
     vector<Point2f> mc( contours.size() );
     for( int i = 0; i < contours.size(); i++ )
         mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
     //connect all contours into ONE
     for (int i = 0; i < contours.size(); ++i)
     {
         Scalar color = Scalar( (0, 255), (0,255), (0,255) );
         drawContours( img, contours, i, color, 2, 8, hierarchy, 0, Point() );
         circle( img, mc[i], 4, color, -1, 8, 0 );
         //connect
         if (i+1 <contours.size())
             line(ed1,mc[i],mc[i+1],Scalar(255,255,255));
     }

     img2=img;/*
     contours.clear();
     hierarchy.clear();
     //寻找结果
     findContours(ed1, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
     for (int i = 0;i<contours.size();i++)
     {
         RotatedRect minRect = minAreaRect( Mat(contours[i]) );
         Point2f rect_points[4];
         minRect.points( rect_points );
         for( int j = 0; j < 4; j++ )
             line( img, rect_points[j], rect_points[(j+1)%4],Scalar(255,255,0),2);
         float fshort = std::min(minRect.size.width,minRect.size.height); //short
         float flong = std::max(minRect.size.width,minRect.size.height);  //long
     }
     img2=img;
   /*  vector<vector<Point>> contours; // Vector for storing contour
     vector<Vec4i> hierarchy;
     int largest_area = 0;
     int largest_contour_index = 0;
     Rect bounding_rect;


     Mat dst(img1.rows, img1.cols, CV_8UC3, Scalar::all(0));
     // Convert to gray

     findContours(ed1, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

     Scalar color(0,0,255);

     for(int i = 0; i < contours.size(); i++) // Iterate through each contour
     {
         double a = contourArea(contours[i], false); // Find the area of contour
         if(a > largest_area){
             largest_area = a;
             largest_contour_index = i; // Store the index of largest contour
             bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
         }

     }
     for(int i = 0; i < contours.size(); i++) // Iterate through each contour
     {
         if (i != largest_contour_index) {
             drawContours(dst, contours, i, color, CV_FILLED, 8, hierarchy);
         }
     }

     Scalar color2(255,0,0);
     drawContours(dst, contours,largest_contour_index, color2, 5, 8, hierarchy);*/

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_lastshit(JNIEnv *env, jobject thiz, jlong input_image,
                                                    jlong output_image) {
    // TODO: implement lastshit()
    Mat &img1= *(Mat *)  input_image;
    Mat img3;
    Mat &img2 = *(Mat *) output_image;
//1. Remove Shadows
//Convert to HSV
    Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);


//2. Convert to gray and normalize
    Mat gray(rgbImg.rows, rgbImg.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);


//3. Edge detector
    GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    Mat edges;
    bool useCanny = false;

//edges = canny(gray);

//Use Sobel filter and thresholding.
    // edges = sobel(gray);
//Automatic thresholding
    //  threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
    Mat img4;
    cv::adaptiveThreshold(gray, gray ,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 6);
    double otsu_thresh_val = cv::threshold(
            gray ,img4, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
    double high_thresh_val  = otsu_thresh_val,
            lower_thresh_val = otsu_thresh_val * 2;
    cv::Canny( gray,edges, lower_thresh_val,high_thresh_val );
    Mat kernel = Mat(11,11, CV_8UC1, cv::Scalar(1));
    Mat ed1,img,diff;
    morphologyEx(edges,edges, CV_MOP_CLOSE, kernel,Point(-1,-1),1);



//4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);

//5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

//7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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
    rectangle(img1, boundingRectangle, Scalar(0, 255, 0), 1, 8, 0);
    img1.copyTo(img3,largestContour);

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
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img3);
    cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(img3, img3, img2, mask);
    img2.convertTo(dest,CV_8U);




}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_excract(JNIEnv *env, jobject thiz,
                                               jlong input_image, jlong output_image) {


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
//
// some lengthy op.
//

    img5.convertTo(im32fc3, CV_32FC3);
    calcHist(&im32fc3, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    calcBackProject(&im32fc3, 1, channels, hist, backpr32f, ranges);
    Mat img4,edges;
    double minval, maxval;
    minMaxIdx(backpr32f, &minval, &maxval);
    /*  cv::adaptiveThreshold(gray, gray ,255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 45, 6);
      double otsu_thresh_val = cv::threshold(
              gray ,img4, 0, 255, CV_THRESH_BINARY+CV_THRESH_OTSU);
      double high_thresh_val  = otsu_thresh_val,
              lower_thresh_val = otsu_thresh_val * 2;
      cv::Canny( gray,edges, lower_thresh_val,high_thresh_val );
    Mat floodFilled = cv::Mat::zeros(edges.rows+2, edges.cols+2, CV_8U);
    floodFill(edges, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, edges.cols-2, edges.rows-2)).copyTo(temp);
    floodFilled = temp;
    Mat kernel1 = Mat(11,11, CV_8UC1, cv::Scalar(1));
      Mat ed1,img,diff;
      morphologyEx(temp,temp, CV_MOP_CLOSE, kernel,Point(-1,-1),1);*/
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
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img3);
    // cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);
    bitwise_and(img3, img3, img2, mask);
    int64 t1 = cv::getTickCount();
    double secs = (t1-t0)/cv::getTickFrequency();
    __android_log_print(ANDROID_LOG_INFO, "sometag", "time  = %d", secs);
    img2.convertTo(dest,CV_8U);

}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_App_StartActivity_hsvsamta(JNIEnv *env, jobject thiz, jlong input_image,
                                                    jlong output_image) {
    // TODO: implement hsvsamta()

    Mat &img1= *(Mat *)  input_image;
    Mat &img2 = *(Mat *) output_image;
///hsvtype1 :
    Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3],channel1[3];
    split(hsvImg, channel);
    channel[0] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);
    Mat blurcha,ada;
    GaussianBlur( channel[2], blurcha, Size(3,3), 0, 0, BORDER_DEFAULT);



    Mat src_hsv;
    Mat bin;
    Mat src_h;

    // pyrMeanShiftFiltering(img1, img1, 5, 35, 3);
    cvtColor(img1,src_hsv,COLOR_BGR2HSV);
    vector<Mat> rgb_planes;
    split(src_hsv, rgb_planes );
    src_h = rgb_planes[2]; // h channel is useful


    medianBlur( src_h,  src_h, 5);
    //cvtColor(img1,src_gray,COLOR_BGR2HSV);
    //find and draw the biggest contour
    cv::adaptiveThreshold(src_h, src_h, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    Mat xorg;

    cv::Canny(src_h,img2,10,140);




}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_nGRIS(JNIEnv *env, jobject thiz, jlong input_image,
                                             jlong output_image) {
    // TODO: implement nGRIS()
    Mat &img1= *(Mat *)  input_image;
    //  resize(img1, img1, Size(), 0.25, 0.25);
    Mat &img2 = *(Mat *) output_image;

    cvtColor(img1, img2, CV_RGB2GRAY);

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_orbcall(JNIEnv *env, jobject thiz, jlong input_image,
                                               jlong output_image) {
    // TODO: implement orb()
    // TODO: implement Orbcal()
    Mat &captured = *(Mat *)  input_image;

    Mat &target = *(Mat *) output_image;
    // Mat &target = *(Mat *) img2;
    BFMatcher matcher(NORM_L2);

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
    drawKeypoints(captured, keypointsCaptured, target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    //__android_log_print(ANDROID_LOG_INFO, "sometag", "keypoints size = %d", keypointsCaptured.size());

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_filtregaussin(JNIEnv *env, jobject thiz, jlong input_image,
                                                     jlong output_image) {
    // TODO: implement filtregaussin()
    Mat &img1= *(Mat *)  input_image;
    Mat &img2 = *(Mat *) output_image;
    Mat gray;
    //  resize(img1, img1, Size(), 0.25, 0.25);
    Mat imgg,rg;
    imgg=exc(img1);
    cv::cvtColor(imgg , rg , CV_RGBA2RGB);
    cv::cvtColor(rg , gray , CV_RGB2GRAY);
    GaussianBlur(gray, img2, Size(3,3), 0, 0, BORDER_DEFAULT);


}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_grabcut(JNIEnv *env, jobject thiz, jlong input_image,
                                               jlong output_image) {
    // TODO: implement grabcutshit()

    Mat &img1= *(Mat *)  input_image;
    Mat img3=img1;  Mat gray;
    Mat img5;
    resize(img1, img1, Size(), 0.25, 0.25);
    clock_t start, stop;

    start = clock();
    cv::cvtColor(img1 , img5 , CV_RGBA2RGB);

    //  GaussianBlur(img1, img1, Size(3,3), 0, 0, BORDER_DEFAULT);

    Mat &img2 = *(Mat *) output_image;

    const int channels[] = {0, 1, 2};
    const int histSize[] = {32, 32, 32};
    const float rgbRange[] = {0, 256};
    const float* ranges[] = {rgbRange, rgbRange, rgbRange};

    Mat hist;
    Mat im32fc3, backpr32f, backpr8u, backprBw, kernel;


    img5.convertTo(im32fc3, CV_32FC3);
    calcHist(&im32fc3, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    calcBackProject(&im32fc3, 1, channels, hist, backpr32f, ranges);
    Mat img4,edges;
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


    //grabCut(img5, mask, Rect(), bgdModel, fgdModel,GC_INIT_WITH_MASK);
    cv::parallel_for_(cv::Range(0, 8), Parallel_process(img5, mask, backprBw, 5, 8));
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
    stop = clock();
    double b=(double)(stop - start)/CLOCKS_PER_SEC*1000 ;
    __android_log_print(ANDROID_LOG_INFO, "sometag", "\"Running time using \\'for\\':\" = %d", b);
    img5.copyTo(img2,largestContour);
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_watershed(JNIEnv *env, jobject thiz, jlong input_image,
                                                 jlong output_image) {

    Mat &img1= *(Mat *)  input_image;
    Mat &img2 = *(Mat *) output_image;
    // resize(img1, img1, Size(), 0.25, 0.25);
    //cv::cvtColor(img1 , img1 , CV_RGBA2RGB);
    medianBlur(img1,img1,3);
    cvtColor(img1,img1,COLOR_BGRA2BGR);
    cv::Mat blank(img1.size(),CV_8U,cv::Scalar(0xFF));
    cv::Mat dest;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));

    // Create markers image
    cv::Mat markers(img1.size(),CV_8U,cv::Scalar(-1));
    //Rect(topleftcornerX, topleftcornerY, width, height);
    //top rectangle
    markers(Rect(0,0,img1.cols, 5)) = Scalar::all(1);
    //bottom rectangle
    markers(Rect(0,img1.rows-5,img1.cols, 5)) = Scalar::all(1);
    //left rectangle
    markers(Rect(0,0,5,img1.rows)) = Scalar::all(1);
    //right rectangle
    markers(Rect(img1.cols-5,0,5,img1.rows)) = Scalar::all(1);
    //centre rectangle
    int centreW = img1.cols/4;
    int centreH = img1.rows/4;
    markers(Rect((img1.cols/2)-(centreW/2),(img1.rows/2)-(centreH/2), centreW, centreH)) = Scalar::all(2);
    markers.convertTo(markers,CV_BGR2GRAY);


    //Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);
    cv::Mat wshedMask = segmenter.process(img1);
    cv::Mat mask;
    convertScaleAbs(wshedMask, mask, 1, 0);
    double thresh = threshold(mask, mask, 1, 255, THRESH_BINARY);


    bitwise_and(img1, img1, img2, mask);
    img2.convertTo(dest,CV_8U);
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_orb(JNIEnv *env, jobject thiz, jlong input_image,
                                           jlong output_image,jlong descriptorr) {

    /*  int kernel_size = 3;
      int scale = 1;
      int delta = 0;
      int ddepth = CV_16S;
      vector<vector<Point>>();
      RNG rng(12345);
      vector<Vec4i>();*/

    Mat &captured = *(Mat *)  input_image;
    //resize(captured, captured, Size(), 0.25, 0.25);
    Mat &target = *(Mat *) output_image;
    Mat &desc = *(Mat *) descriptorr;
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( captured,keypoints_1 );


    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( captured, keypoints_1, descriptors_1 );

    desc= descriptors_1;
    Mat outimg1;
    Mat img;
    img=exc(captured);
    drawKeypoints( img, keypoints_1,target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    // Mat &target = *(Mat *) img2;
    /* BFMatcher matcher(NORM_L2);

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

     //cvErode (dst,0,0,1);

     // GaussianBlur( dst, dst, Size(5,5),0,0 );
     medianBlur(dst, dst, 5);
     Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
     dilate(dst, dst, kernel);


     erode(dst, dst, kernel);

     Mat edges;
     Mat dst1=dst;
     int scale1 = 1;
     int delta1 = 0;
     int ddepth1 = CV_16S;
     Mat edges_x, edges_y;
     Mat abs_edges_x, abs_edges_y;
     Sobel(dst1, edges_x, ddepth1, 1, 0, 3, scale1, delta1, BORDER_DEFAULT);
     convertScaleAbs( edges_x, abs_edges_x );
     Sobel(dst1, edges_y, ddepth1, 0, 1, 3, scale1, delta1, BORDER_DEFAULT);
     convertScaleAbs(edges_y, abs_edges_y);
     addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, dst);

     Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
     // drawContours(dst, contours, largest_contour_index, color);

     orb->detectAndCompute(dst, noArray(), keypointsCaptured, target);
     if(descriptorsCaptured.type()!=CV_32F) {
         descriptorsCaptured.convertTo(descriptorsCaptured, CV_32F); }
     drawKeypoints(captured, keypointsCaptured, target, Scalar::all(-1), DrawMatchesFlags::DEFAULT );*/
}


vector<int> convertToBinary1(int x) {
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

int countTransitions1(vector<int> x) {
    int result = 0;
    for(int i = 0; i < 8; ++i)
        result += (x[i] != x[(i+1) % 8]);
    return result;
}

Mat uniformPatternHistogram1(const Mat& src, int numPatterns) {
    Mat hist;
    hist = Mat::zeros(1, (numPatterns+1), CV_32SC1);

    for (int i = 0; i < numPatterns; ++i) {
        if (countTransitions1(convertToBinary1(i)) > 2)
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

void uniformPatternSpatialHistogram1(const Mat& src, Mat& hist, int numPatterns,
                                     int gridX, int gridY, int overlap) {

    int width = src.cols;
    int height = src.rows;
    vector<Mat> histograms;

    Size window = Size(static_cast<int>(floor(src.cols/gridX)),
                       static_cast<int>(floor(src.rows/gridY)));

    for (int x = 0; x <= (width - window.width); x+= (window.width - overlap)) {
        for (int y = 0; y <= (height - window.height); y+= (window.height - overlap)) {
            Mat cell = Mat(src, Rect(x, y, window.width, window.height));
            histograms.push_back(uniformPatternHistogram1(cell, numPatterns));
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

vector<int> getFeatureVector1(Mat spatial_hist) {
    vector<int> feature_vector;
    for(int j = 0; j < spatial_hist.cols; ++j) {
        if(spatial_hist.at<int>(0, j) != -1)
            feature_vector.push_back(spatial_hist.at<int>(0, j));
    }
    return feature_vector;
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_myshite_MainActivity7_lbp(JNIEnv *env, jobject thiz, jlong input_image,
                                           jlong output_image ) {
    // TODO: implement lbpshit()

    Mat &img_input = *(Mat *) input_image;
    //resize(img_input, img_input, Size(), 0.25, 0.25);
    Mat &img_output = *(Mat *) output_image;

    Mat &dst = *(Mat *) output_image;
    //  Mat  &spatial_histogram=*(Mat *) hisss;
    Mat img;
    img=exc(img_input);

    cvtColor( img, dst, COLOR_RGB2GRAY);

    Mat lbp_image;
    LBPsht(dst, lbp_image);
    Mat spat;

    uniformPatternSpatialHistogram1(lbp_image, spat, 256, 3, 3, 0);
    //  spatial_histogram=spat;
    img_output=lbp_image;
    vector<int> feature_vector = getFeatureVector1(spat);

    Mat lbp_hist, lbp1_hist;
    int histSize[] = {256};
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0 };

    calcHist( &lbp_image, 1, channels, Mat(), lbp_hist, 1, histSize, ranges, true, false );
    FileStorage fs;
    fs.open("/sdcard/lbp/file.xml", FileStorage::WRITE);
    fs << "Des" <<lbp_hist;
    fs.release();
    /*   jintArray arr = env->NewIntArray( feature_vector.size() );
       env->SetIntArrayRegion( arr, 0, feature_vector.size(), ( jint * ) &feature_vector[0] );*/

    /*int n = feature_vector.size();

    int arr[n];
    for (int i = 0; i < n; i++)
        arr[i] = feature_vector[i];
    const char* file_path = (*env).GetStringUTFChars(path,NULL);
    if(file_path != NULL){
        LOGD("From c file_path %s", file_path);
    }

    //打开文件
    FILE* file = fopen(file_path, "w+");
    if(file != NULL){
        LOGD("From c open file success");
    }
    */

    /* stringstream ss;
     copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
     string s1 = ss.str();
     //s1 = s1.substr(0, s1.length()-1);

     char* pString = new char[s1.length() + 1];
     std::copy(s1.c_str(), s1.c_str() + s1.length() + 1, pString);
 */
    /*  env->NewStringUTF(pString);
      fwrite(&pString,  s1.size() +1,1,file);
      //写入文件
      //fwrite(feature_vector.data(), sizeof feature_vector[0], feature_vector.size(), file);


      //关闭文件
      if(file != NULL){
          fclose(file);
      }

      (*env).ReleaseStringUTFChars(path, file_path);
      /* stringstream ss;
       copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));

    /* string s1 = ss.str();
       s1 = s1.substr(0, s1.length()-1);

       char cstr[s1.size() + 1];

       std::copy(s1.begin(), s1.end(), cstr);
       return env->NewStringUTF(cstr);*/
    /* jintArray arr = env->NewIntArray( feature_vector.size() );
     env->SetIntArrayRegion( arr, 0, feature_vector.size(), ( jint * ) &feature_vector[0] );
     return arr;*/


    /*  stringstream ss;
      copy( feature_vector.begin(), feature_vector.end(), ostream_iterator<int>(ss, " "));
      string s1 = ss.str();
      char* pString = new char[s1.length() + 1];
      std::copy(s1.c_str(), s1.c_str() + s1.length() + 1, pString);


      return env->NewStringUTF(pString);*/

    jintArray arr = env->NewIntArray( feature_vector.size() );
    env->SetIntArrayRegion( arr, 0, feature_vector.size(), ( jint * ) &feature_vector[0] );
    return arr;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_edge(JNIEnv *env, jobject thiz, jlong input_imag,
                                            jlong output_imag) {
    // TODO: implement edge()
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    vector<vector<Point>>();
    RNG rng(12345);
    vector<Vec4i>();
    Mat &img_input = *(Mat *) input_imag;

    Mat &img_output = *(Mat *) output_imag;

    Mat &dst = *(Mat *) output_imag;
    //resize(img_input, img_input, Size(), 0.25, 0.25);
    Mat img;
    img=exc(img_input);
    cvtColor( img, dst, COLOR_RGB2GRAY);

    //cvErode (dst,0,0,1);

    // GaussianBlur( dst, dst, Size(5,5),0,0 );
    medianBlur(dst, dst, 5);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(dst, dst, kernel);


    erode(dst, dst, kernel);

    // Canny( dst, dst, THRESH_OTSU, THRESH_OTSU*2);
    Mat edges;
    Mat dst1=dst;
    int scale1 = 1;
    int delta1 = 0;
    int ddepth1 = CV_16S;
    Mat edges_x, edges_y;
    Mat abs_edges_x, abs_edges_y;
    Sobel(dst1, edges_x, ddepth1, 1, 0, 3, scale1, delta1, BORDER_DEFAULT);
    convertScaleAbs( edges_x, abs_edges_x );
    Sobel(dst1, edges_y, ddepth1, 0, 1, 3, scale1, delta1, BORDER_DEFAULT);
    convertScaleAbs(edges_y, abs_edges_y);
    addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, dst);
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_hsvhisto(JNIEnv *env, jobject thiz, jlong input_image,
                                                jlong output_image) {
    // TODO: implement hsvhisto()

}
typedef struct t_color_node {
    cv::Mat       mean;       // The mean of this node
    cv::Mat       cov;
    uchar         classid;    // The class ID

    t_color_node  *left;
    t_color_node  *right;
} t_color_node;

cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
    const int tile_size = 64;
    cv::Mat ret = cv::Mat(tile_size, tile_size*colors.size(), CV_8UC3, cv::Scalar(0));

    for(int i=0;i<colors.size();i++) {
        cv::Rect rect(i*tile_size, 0, tile_size, tile_size);
        cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
    }

    return ret;
}

std::vector<t_color_node*> get_leaves(t_color_node *root) {
    std::vector<t_color_node*> ret;
    std::queue<t_color_node*> queue;
    queue.push(root);

    while(queue.size() > 0) {
        t_color_node *current = queue.front();
        queue.pop();

        if(current->left && current->right) {
            queue.push(current->left);
            queue.push(current->right);
            continue;
        }

        ret.push_back(current);
    }

    return ret;
}

std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
    std::vector<t_color_node*> leaves = get_leaves(root);
    std::vector<cv::Vec3b> ret;

    for(int i=0;i<leaves.size();i++) {
        cv::Mat mean = leaves[i]->mean;
        ret.push_back(cv::Vec3b(mean.at<double>(0)*255.0f,
                                mean.at<double>(1)*255.0f,
                                mean.at<double>(2)*255.0f));
    }

    return ret;
}

int get_next_classid(t_color_node *root) {
    int maxid = 0;
    std::queue<t_color_node*> queue;
    queue.push(root);

    while(queue.size() > 0) {
        t_color_node* current = queue.front();
        queue.pop();

        if(current->classid > maxid)
            maxid = current->classid;

        if(current->left != NULL)
            queue.push(current->left);

        if(current->right)
            queue.push(current->right);
    }

    return maxid + 1;
}

void get_class_mean_cov(cv::Mat img, cv::Mat classes, t_color_node *node) {
    const int width = img.cols;
    const int height = img.rows;
    const uchar classid = node->classid;

    cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
    cv::Mat cov = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

    // We start out with the average color
    double pixcount = 0;
    for(int y=0;y<height;y++) {
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
        uchar* ptrClass = classes.ptr<uchar>(y);
        for(int x=0;x<width;x++) {
            if(ptrClass[x] != classid)
                continue;

            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
            scaled.at<double>(0) = color[0]/255.0f;
            scaled.at<double>(1) = color[1]/255.0f;
            scaled.at<double>(2) = color[2]/255.0f;

            mean += scaled;
            cov = cov + (scaled * scaled.t());

            pixcount++;
        }
    }

    cov = cov - (mean * mean.t()) / pixcount;
    mean = mean / pixcount;

    // The node mean and covariance
    node->mean = mean.clone();
    node->cov = cov.clone();

    return;
}

void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
    const int width = img.cols;
    const int height = img.rows;
    const int classid = node->classid;

    const uchar newidleft = nextid;
    const uchar newidright = nextid+1;

    cv::Mat mean = node->mean;
    cv::Mat cov = node->cov;
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov, eigenvalues, eigenvectors);

    cv::Mat eig = eigenvectors.row(0);
    cv::Mat comparison_value = eig * mean;

    node->left = new t_color_node();
    node->right = new t_color_node();

    node->left->classid = newidleft;
    node->right->classid = newidright;

    // We start out with the average color
    for(int y=0;y<height;y++) {
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
        uchar* ptrClass = classes.ptr<uchar>(y);
        for(int x=0;x<width;x++) {
            if(ptrClass[x] != classid)
                continue;

            cv::Vec3b color = ptr[x];
            cv::Mat scaled = cv::Mat(3, 1,
                                     CV_64FC1,
                                     cv::Scalar(0));

            scaled.at<double>(0) = color[0]/255.0f;
            scaled.at<double>(1) = color[1]/255.0f;
            scaled.at<double>(2) = color[2]/255.0f;

            cv::Mat this_value = eig * scaled;

            if(this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
                ptrClass[x] = newidleft;
            } else {
                ptrClass[x] = newidright;
            }
        }
    }
    return;
}

cv::Mat get_quantized_image(cv::Mat classes, t_color_node *root) {
    std::vector<t_color_node*> leaves = get_leaves(root);

    const int height = classes.rows;
    const int width = classes.cols;
    cv::Mat ret(height, width, CV_8UC3, cv::Scalar(0));

    for(int y=0;y<height;y++) {
        uchar *ptrClass = classes.ptr<uchar>(y);
        cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
        for(int x=0;x<width;x++) {
            uchar pixel_class = ptrClass[x];
            for(int i=0;i<leaves.size();i++) {
                if(leaves[i]->classid == pixel_class) {
                    ptr[x] = cv::Vec3b(leaves[i]->mean.at<double>(0)*255,
                                       leaves[i]->mean.at<double>(1)*255,
                                       leaves[i]->mean.at<double>(2)*255);
                }
            }
        }
    }

    return ret;
}

cv::Mat get_viewable_image(cv::Mat classes) {
    const int height = classes.rows;
    const int width = classes.cols;

    const int max_color_count = 12;
    cv::Vec3b *palette = new cv::Vec3b[max_color_count];
    palette[0]  = cv::Vec3b(  0,   0,   0);
    palette[1]  = cv::Vec3b(255,   0,   0);
    palette[2]  = cv::Vec3b(  0, 255,   0);
    palette[3]  = cv::Vec3b(  0,   0, 255);
    palette[4]  = cv::Vec3b(255, 255,   0);
    palette[5]  = cv::Vec3b(  0, 255, 255);
    palette[6]  = cv::Vec3b(255,   0, 255);
    palette[7]  = cv::Vec3b(128, 128, 128);
    palette[8]  = cv::Vec3b(128, 255, 128);
    palette[9]  = cv::Vec3b( 32,  32,  32);
    palette[10] = cv::Vec3b(255, 128, 128);
    palette[11] = cv::Vec3b(128, 128, 255);

    cv::Mat ret = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    for(int y=0;y<height;y++) {
        cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
        uchar *ptrClass = classes.ptr<uchar>(y);
        for(int x=0;x<width;x++) {
            int color = ptrClass[x];
            if(color >= max_color_count) {

                continue;
            }
            ptr[x] = palette[color];
        }
    }

    return ret;
}

t_color_node* get_max_eigenvalue_node(t_color_node *current) {
    double max_eigen = -1;
    cv::Mat eigenvalues, eigenvectors;

    std::queue<t_color_node*> queue;
    queue.push(current);

    t_color_node *ret = current;
    if(!current->left && !current->right)
        return current;

    while(queue.size() > 0) {
        t_color_node *node = queue.front();
        queue.pop();

        if(node->left && node->right) {
            queue.push(node->left);
            queue.push(node->right);
            continue;
        }

        cv::eigen(node->cov, eigenvalues, eigenvectors);
        double val = eigenvalues.at<double>(0);
        if(val > max_eigen) {
            max_eigen = val;
            ret = node;
        }
    }

    return ret;
}

Mat find_dominant_colors(cv::Mat img, int count) {
    const int width = img.cols;
    const int height = img.rows;

    cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
    t_color_node *root = new t_color_node();

    root->classid = 1;
    root->left = NULL;
    root->right = NULL;

    t_color_node *next = root;
    get_class_mean_cov(img, classes, root);
    for(int i=0;i<count-1;i++) {
        next = get_max_eigenvalue_node(root);
        partition_class(img, classes, get_next_classid(root), next);
        get_class_mean_cov(img, classes, next->left);
        get_class_mean_cov(img, classes, next->right);
    }

    std::vector<cv::Vec3b> colors = get_dominant_colors(root);

    cv::Mat quantized = get_quantized_image(classes, root);
    cv::Mat viewable = get_viewable_image(classes);
    cv::Mat dom = get_dominant_palette(colors);



    return dom;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_calculcolordescriptors(JNIEnv *env, jobject thiz,
                                                              jlong input_image,
                                                              jlong output_image) {
    // TODO: implement calculcolordescriptors()
    Mat &img_input = *(Mat *) input_image;
    Mat &img_output = *(Mat *) output_image;
    int count =3;
    // std::vector<cv::Vec3b> colors = find_dominant_colors(img_input , count);
    img_output=find_dominant_colors(img_input,count);


}
Mat showCenters(const Mat &centers , int siz=64) {
    Mat cent = centers.reshape(3, centers.rows);
    // make  a horizontal bar of K color patches:
    Mat draw(siz , siz * cent.rows, cent.type(), Scalar::all(0));
    for (int i=0; i<cent.rows; i++) {
        // set the resp. ROI to that value (just fill it):
        draw( Rect(i * siz, 0, siz, siz)) = cent.at<Vec3f>(i,0);
    }
    draw.convertTo(draw, CV_8U);



    return draw;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_dominatwithkmeans(JNIEnv *env, jobject thiz,
                                                         jlong input_image, jlong output_image) {
    // TODO: implement dominatwithkmeans()

    Mat &img_input = *(Mat *) input_image;
    Mat &img_output = *(Mat *) output_image;
    Mat data,ocv,img1,ii,tmp;
    cvtColor(img_input, tmp, CV_BGR2RGB);
    cvtColor(tmp,ii, CV_RGB2Lab);

    Mat channel[3];
    Mat channel1[3];
    split(ii,channel);




/*
double x= -0.127;
    double x1= - 0.339;
    double y=-0.083;

/*
    channel1[0]  = 0.299*channel[0] + 0.587*channel[1]+ 0.1148*channel[2];

    channel1[2] = 0.511*channel[0] - 0.428*channel[1] +y*channel[2] + 128;
   channel1[0]=0.299*channel[0]+0.578*channel[1]+0.114*channel[2];
    //channel1[2]=-0.1687*channel[0]-0.3313*channel[1]+0.5*channel[2]+0.5;

//
    //channel1[1]=0.53*channel[0]-0.4187*channel[1]-0.0813*channel[2]+0.5;
    channel1[1]= x*channel[0] + x1*channel[1] + 0.5211*channel[2] + 128;
    tmp=channel[1];*/

    tmp.convertTo(data, CV_32F);
    data = data.reshape(3, data.total());

// do kmeans$

    Mat labels, centers;
    kmeans(data, 3, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
           KMEANS_PP_CENTERS, centers);
    img_output=showCenters(centers,64);
/*
// reshape both to a single row of Vec3f pixels:
    centers = centers.reshape(3, centers.rows);
    data = data.reshape(3, data.rows);

// replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i = 0; i<data.rows; i++) {
        int center_id = labels.at<int>(i);
        p[i] = centers.at<Vec3f>(center_id);
    }

// back to 2d, and uchar:
    ocv = data.reshape(3, ocv.rows);
    ocv.convertTo(ocv, CV_8U);*/
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_remover(JNIEnv *env, jobject thiz, jlong input_image,
                                               jlong output_image) {
    // TODO: implement remover()

    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
//1. Remove Shadows
//Convert to HSV
    Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

//
//2. Convert to gray and normalize
    Mat gray(rgbImg.rows, img1.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    //avoid the influence of  high frequency noise and very low noise .And on the other hand ,it make image data satisfy  nomal distribution
    //is not only to remove noise but at the same time to bring the image into a range of intensity values that is 'normal'...(meaning statistically it follows a normal distribution as far as possible),physically less stressful to our (visual) sense. The mean value will depend on the actual intensity distribution in the image...but the aim will be to ultimately state this mean with high confidence level
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);


//3. Edge detector
    GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    Mat edges;
    bool useCanny = false;

//edges = canny(gray);

//Use Sobel filter and thresholding.
    edges = sobel(gray);
//Automatic thresholding
    threshold(edges, edges, 0, 255, cv::THRESH_OTSU);





//4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);

//5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

//7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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



    img1.copyTo(img2, largestContour);

}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_Kmeans_rempve(JNIEnv *env, jobject thiz, jlong input_image,
                                       jlong output_image) {
    // TODO: implement rempve()

    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
//1. Remove Shadows
//Convert to HSV
    Mat hsvImg;
    cvtColor(img1, hsvImg, CV_BGR2HSV);
    Mat channel[3];
    split(hsvImg, channel);
    channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
//Merge channels
    merge(channel, 3, hsvImg);
    Mat rgbImg;
    cvtColor(hsvImg, rgbImg, CV_HSV2BGR);

//
//2. Convert to gray and normalize
    Mat gray(rgbImg.rows, img1.cols, CV_8UC1);
    cvtColor(rgbImg, gray, CV_BGR2GRAY);
    //avoid the influence of  high frequency noise and very low noise .And on the other hand ,it make image data satisfy  nomal distribution
    //is not only to remove noise but at the same time to bring the image into a range of intensity values that is 'normal'...(meaning statistically it follows a normal distribution as far as possible),physically less stressful to our (visual) sense. The mean value will depend on the actual intensity distribution in the image...but the aim will be to ultimately state this mean with high confidence level
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);


//3. Edge detector
    GaussianBlur(gray, gray, Size(3,3), 0, 0, BORDER_DEFAULT);
    Mat edges;
    bool useCanny = false;

//edges = canny(gray);

//Use Sobel filter and thresholding.
    edges = sobel(gray);
//Automatic thresholding
    threshold(edges, edges, 0, 255, cv::THRESH_OTSU);





//4. Dilate
    Mat dilateGrad = edges;
    int dilateType = MORPH_ELLIPSE;
    int dilateSize = 3;
    Mat elementDilate = getStructuringElement(dilateType,
                                              Size(2*dilateSize + 1, 2*dilateSize+1),
                                              Point(dilateSize, dilateSize));
    dilate(edges, dilateGrad, elementDilate);

//5. Floodfill
    Mat floodFilled = cv::Mat::zeros(dilateGrad.rows+2, dilateGrad.cols+2, CV_8U);
    floodFill(dilateGrad, floodFilled, cv::Point(0, 0), 0, 0, cv::Scalar(), cv::Scalar(), 4 + (255 << 8) + cv::FLOODFILL_MASK_ONLY);
    floodFilled = cv::Scalar::all(255) - floodFilled;
    Mat temp;
    floodFilled(Rect(1, 1, dilateGrad.cols-2, dilateGrad.rows-2)).copyTo(temp);
    floodFilled = temp;


//6. Erode
    int erosionType = MORPH_ELLIPSE;
    int erosionSize = 4;
    Mat erosionElement = getStructuringElement(erosionType,
                                               Size(2*erosionSize+1, 2*erosionSize+1),
                                               Point(erosionSize, erosionSize));
    erode(floodFilled, floodFilled, erosionElement);

//7. Find largest contour
    int largestArea = 0;
    int largestContourIndex = 0;
    Rect boundingRectangle;
    Mat largestContour(img1.rows, img1.cols, CV_8UC1, Scalar::all(0));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(floodFilled, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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



    img1.copyTo(img2, largestContour);
}
Point2i center(vector<Point2i> contour)
{
    Moments m = moments(contour);

    return Point2i(m.m10/m.m00, m.m01/m.m00);
}


vector<Point2i> getCenters(Mat img)
{
    vector<vector<Point2i> > contours;
    Mat edge;
    edge= sobel(img);

    findContours(edge, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    vector<Point2i> result(contours.size());

    for (int i=0; i<contours.size(); i++)
    {
        result[i] = center(contours[i]);
    }

    return result;
}


double distanceCenter(Point2i center1, Point2i center2)
{
    return (center1.x - center2.x)*(center1.x - center2.x) + (center1.y - center2.y)*(center1.y - center2.y);
}


double distanceCenters(vector<Point2i> centers1, vector<Point2i> centers2)
{
    if (centers1.size() != centers2.size())
    {
        return -1;
    }
    else
    {
        double result = 0;

        for (int i=0; i<centers1.size(); i++)
        {
            double min = INT_MAX;

            for (int j=0; j<centers2.size(); j++)
            {
                double dist = distanceCenter(centers1[i], centers2[j]);
                if (dist < min)
                {
                    min = dist;
                }
            }

            result += min;
        }

        return result;
    }
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_myshite_MainActivity7_shapedescr(JNIEnv *env, jobject thiz, jlong input_imag,
                                                  jlong output_imag) {
    // TODO: implement shapedescr()
    Mat &img1 = *(Mat *) input_imag;
    Mat &img2 = *(Mat *) output_imag;
    cvtColor(img1,img1,CV_RGB2GRAY);
    cvtColor(img2,img2,CV_RGB2GRAY);


    return distanceCenters(getCenters(img1), getCenters(img2)) ;
}


/*uchar lbp(const Mat_<uchar> & img, int x, int y)
{
    // this is pretty much the same what you already got..
    uchar v = 0;
    uchar c = img(y,x);
    v += (img(y-1,x  ) > c) << 0;
    v += (img(y-1,x+1) > c) << 1;
    v += (img(y  ,x+1) > c) << 2;
    v += (img(y+1,x+1) > c) << 3;
    v += (img(y+1,x  ) > c) << 4;
    v += (img(y+1,x-1) > c) << 5;
    v += (img(y  ,x-1) > c) << 6;
    v += (img(y-1,x-1) > c) << 7;
    return v;
}
*/
void histogram(const Mat& src, Mat& hist, int numPatterns) {
    hist = Mat::zeros(1, numPatterns, CV_32SC1);
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            int bin = src.at<int>(i,j);
            hist.at<int>(0,bin) += 1;
        }
    }
}
Mat histogram(const Mat& src, int numPatterns) {
    Mat hist;
    histogram(src, hist, numPatterns);
    return hist;
}


void lbpspatial_histogram(const Mat& src, Mat& hist, int numPatterns, const Size& window, int overlap) {
    int width = src.cols;
    int height = src.rows;
    vector<Mat> histograms;
    for(int x=0; x < width - window.width; x+=(window.width-overlap)) {
        for(int y=0; y < height-window.height; y+=(window.height-overlap)) {
            Mat cell = Mat(src, Rect(x,y,window.width, window.height));
            histograms.push_back(histogram(cell, numPatterns));
        }
    }
    hist.create(1, histograms.size()*numPatterns, CV_32SC1);
    // i know this is a bit lame now... feel free to make this a bit more efficient...
    for(int histIdx=0; histIdx < histograms.size(); histIdx++) {
        for(int valIdx = 0; valIdx < numPatterns; valIdx++) {
            int y = histIdx*numPatterns+valIdx;
            hist.at<int>(0,y) = histograms[histIdx].at<int>(valIdx);
        }
    }
}
int* uniform_circular_LBP_histogram(Mat& src) {
    int i, j;
    int radius = 1;
    int neighbours = 8;
    Size size = src.size();
    int *hist_array = (int *)calloc(59,sizeof(int));
    int uniform[] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};
    Mat  dst = Mat::zeros(size.height - 2 * radius, size.width - 2 * radius, CV_8UC1);

    for (int n = 0; n < neighbours; n++) {
        float x = static_cast<float>(radius) *  cos(2.0 * M_PI * n / static_cast<float>(neighbours));
        float y = static_cast<float>(radius) * -sin(2.0 * M_PI * n / static_cast<float>(neighbours));

        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(x));

        float ty = y - fy;
        float tx = y - fx;

        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 = 1 - w1 - w2 - w3;

        for (i = 0; i < 59; i++) {
            hist_array[i] = 0;
        }

        for (i = radius; i < size.height - radius; i++) {
            for (j = radius; j < size.width - radius; j++) {
                float t = w1 * src.at<uchar>(i + fy, j + fx) + \
                 w2 * src.at<uchar>(i + fy, j + cx) + \
                 w3 * src.at<uchar>(i + cy, j + fx) + \
                 w4 * src.at<uchar>(i + cy, j + cx);
                dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i,j)) && \
                                                 (abs(t - src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }

    for (i = radius; i < size.height - radius; i++) {
        for (j = radius; j < size.width - radius; j++) {
            int val = uniform[dst.at<uchar>(i - radius, j - radius)];
            dst.at<uchar>(i - radius, j - radius) = val;
            hist_array[val] += 1;
        }
    }
    return hist_array;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myshite_MainActivity7_lbpshiiit(JNIEnv *env, jobject thiz, jlong input_image5,
                                                 jlong output_image5) {
    // TODO: implement lbpshiiit()
    Mat &img1 = *(Mat *) input_image5;
    Mat &img2 = *(Mat *) output_image5;
    //cvtColor(img1,img1,CV_RGB2GRAY);
    const int width = img1.cols;
    const int height = img1.rows;
    int count = 1;

    int arr1[59];
    //vector<int> feature_vector;
    string s1;
    for (int i = 0; i <= width - 8; i += 25) {
        for (int j = 0; j <= height - 8; j += 25) {
            //  Mat new_mat = resized_src.rowRange(i, i + 25).colRange(j, j + 25);
            int *hist = uniform_circular_LBP_histogram(img1);
            int z;
            for (z = 0; z < 58; z++) {
                std::cout << hist[z] << ",";
                s1 = hist[z]+" ";
            }
            std::cout << hist[z] << "\n";
            count += 1;
        }
    }



    /*   jintArray arr = env->NewIntArray(feature_vector.size());
       env->SetIntArrayRegion( arr, 0,feature_vector.size(), ( jint * ) &feature_vector[0] );*/

    char* pString = new char[s1.length() + 1];
    std::copy(s1.c_str(), s1.c_str() + s1.length() + 1, pString);


    return env->NewStringUTF(pString);


}
void fuck(Mat I, Mat fI)
{
    Mat_<uchar> feature(I.size(),0);
    Mat_<uchar> img(I);
    const int m=1;
    for (int r=m; r<img.rows-m; r++)
    {
        for (int c=m; c<img.cols-m; c++)
        {
            uchar v = 0;
            uchar cen = img(r,c);
            v |= (img(r-1,c  ) > cen) << 0;
            v |= (img(r-1,c+1) > cen) << 1;
            v |= (img(r  ,c+1) > cen) << 2;
            v |= (img(r+1,c+1) > cen) << 3;
            v |= (img(r+1,c  ) > cen) << 4;
            v |= (img(r+1,c-1) > cen) << 5;
            v |= (img(r  ,c-1) > cen) << 6;
            v |= (img(r-1,c-1) > cen) << 7;
            feature(r,c) = v;
        }
    }
    fI = feature;

}
static void hist_patch_uniform(const Mat_<uchar> &fI, Mat &histo)
{
    static int uniform[256] =
            {   // the well known original uniform2 pattern
                    0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
                    14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
                    58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
                    58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
                    58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
                    58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
                    58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
                    58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
                    58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
                    58,58,58,50,51,52,58,53,54,55,56,57
            };

    Mat_<float> h(1, 60, 0.0f); // mod4
    for (int i=0; i<fI.rows; i++)
    {
        for (int j=0; j<fI.cols; j++)
        {
            int v = int(fI(i,j));
            h( uniform[v] ) += 1.0f;
        }
    }
    histo.push_back(h.reshape(1,1));
}
vector<int> getHist( bool norm ,Mat hist ) {
    vector<int> h;

    for(int j = 0; j < hist.cols; ++j) {

        h.push_back(hist.at<int>(0, j));
    }


    return h;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_elbbp(JNIEnv *env, jobject thiz, jlong input_image,
                                             jlong output_image , jlong hist) {

    // TODO: implement elbbp()

    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
    Mat &histt = *(Mat *) hist;
    cvtColor(img1,img1,CV_RGB2GRAY);
    Mat img3;
    string s="";
    img3=Expand_LBP_demo(img1);
    Mat spat;

    // uniformPatternSpatialHistogram1(img3, spat, 256, 3, 3, 0);


    Mat lbp_hist, lbp1_hist;
    int histSize[] = {256};
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0 };

    calcHist( &img3, 1, channels, Mat(), lbp_hist, 1, histSize, ranges, true, false );
    //  normalize( lbp_hist, lbp_hist, 0, 1, NORM_MINMAX, -1, Mat() );
    //  spatial_histogram=spat;
    /* int uniform[] = {0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,58,58,58,50,51,52,58,53,54,55,56,57};
    int i,j;

 int hist_array[256];
     for (i = 0; i < 256; i++) {
         hist_array[i] = 0;
     }

     Size size = img3.size();

 for (i =0; i < img3.rows ; i++) {
 for (j =0; j < img3.cols ; j++) {

  int val=int(img3(i , j )) ;
 hist_array[val] += 1;
 }
 }

 int z;
     for (z = 0; z < 256; z++) {
         s=hist_array[z]+" " ;
     }

     char* pString = new char[s.length() + 1];
     std::copy(s.c_str(), s.c_str() + s.length() + 1, pString);



     int histSize = 256;
     float range[] = { 0, 256 };
     const float* histRange = { 256};
     if( mask == NULL ) {
         cv::calcHist( lbpImg, 1, 0, Mat(), // do not use mask
                       hist, 1, &histSize, &histRange, true, // the histogram is uniform
                       false // do not accumulate
         );*/

    //Mat hist= histogram(img3, 255);
    // vector<int> feature_vector = getHist(true,lbp_hist);
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
double chi_square_(Mat& histogram0,  Mat& histogram1) {
    if(histogram0.type() != histogram1.type())
        CV_Error(CV_StsBadArg, "Histograms must be of equal type.");
    if(histogram0.rows != 1 || histogram0.rows != histogram1.rows || histogram0.cols != histogram1.cols)
        CV_Error(CV_StsBadArg, "Histograms must be of equal dimension.");
    double result = 0.0;
    for(int i=0; i < histogram0.cols; i++) {
        double a = histogram0.at<int>(0,i) - histogram1.at<int>(0,i);
        double b = histogram0.at<int>(0,i) + histogram1.at<int>(0,i);
        if(abs(b) > numeric_limits<double>::epsilon()) {
            result+=(a*a)/b;
        }
    }
    return result;
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_myshite_MainActivity7_chisquare(JNIEnv *env, jobject thiz, jlong input_image,
                                                 jlong output_image) {
    // TODO: implement chisquare()
    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
    double a=chi_square_(img1,img2);

    return a;

}
int  processImageWithHsv(Mat &image)
{
    Mat image_hsv;

    cvtColor(image, image_hsv, CV_BGR2HSV);

    int hbins = 50, sbins = 60;
    int histSize[] = {hbins, sbins};


    float hranges[] = { 0, 180 };
    float sranges[] = { 0, 256 };
    float vranges[] = { 0, 256 };
    const float* ranges[] = { hranges };
    MatND hist;

    int channels[] = {0};

    calcHist( &image_hsv, 1, channels, Mat(), // do not use mask
              hist, 1, histSize, ranges,
              true, // the histogram is uniform
              false );

    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    return maxVal;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_myshite_MainActivity7_hsvhistt(JNIEnv *env, jobject thiz, jlong input_image
) {
    // TODO: implement hsvhistt()
    Mat &img1 = *(Mat *) input_image;
    int a =processImageWithHsv(img1);

    return a;
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_fuckhsv(JNIEnv *env, jobject thiz, jlong input_image,
                                               jlong output_image) {
    // TODO: implement fuckhsv()
    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
    Mat hsv; Mat hue;
    int bins = 40;

    cvtColor( img1, hsv, CV_BGR2HSV );

    /// Use only the Hue value
    hue.create( hsv.size(), hsv.depth() );
    int ch[] = { 0, 0 };
    mixChannels( &hsv, 1, &hue, 1, ch, 1 );

    MatND hist;
    int histSize = MAX( bins, 2 );
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    /// Get the Histogram and normalize it
    calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
    normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

    /// Get Backprojection
    MatND backproj;
    calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );


    /// Draw the histogram
    int w = 400; int h = 400;
    int bin_w = cvRound( (double) w / histSize );
    Mat histImg = Mat::zeros( w, h, CV_8UC3 );

    for( int i = 0; i < bins; i ++ )
    { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }
    img2=histImg;
    /*  Mat hsv ;
      cvtColor(img1, hsv, CV_BGR2HSV);

      // Quantize the hue to 30 levels
      // and the saturation to 32 levels
      int hbins = 30, sbins = 32;
      int histSize[] = {hbins, sbins};
      // hue varies from 0 to 179, see cvtColor
      float hranges[] = { 0, 180 };
      // saturation varies from 0 (black-gray-white) to
      // 255 (pure spectrum color)
      float sranges[] = { 0, 256 };
      const float* ranges[] = { hranges, sranges };
      MatND hist;
      // we compute the histogram from the 0-th and 1-st channels
      int channels[] = {0, 1};

      calcHist( &hsv, 1, channels, Mat(), // do not use mask
                hist, 2, histSize, ranges,
                true, // the histogram is uniform
                false );
      double maxVal=0;
      minMaxLoc(hist, 0, &maxVal, 0, 0);

      int scale = 10;
      Mat histI = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);

  /*
      for( int h = 0; h < hbins; h++ )
          for( int s = 0; s < sbins; s++ )
          {
              float binVal = hist.at<float>(h, s);
              int intensity = cvRound(binVal*255/maxVal);
              rectangle( histImg, Point(h*scale, s*scale),
                         Point( (h+1)*scale - 1, (s+1)*scale - 1),
                         Scalar::all(intensity),
                         CV_FILLED );
          }
img2=histImg;*/
}extern "C"
JNIEXPORT jdoubleArray JNICALL
Java_com_example_myshite_MainActivity7_humom(JNIEnv *env, jobject thiz, jlong input_image,
                                             jlong output_image) {
    // TODO: implement humom()

    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
    Mat im ;
    cvtColor(img1,im, CV_RGB2GRAY);



    Mat edges =sobel(im);


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
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_histths(JNIEnv *env, jobject thiz, jlong input_image,jlong output_image) {
    // TODO: implement histths()
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
    /*  int channels[] = {0,  1};
      int histSize[] = {hbins, sbins};
      float hranges[] = { 0, 180 };
      float sranges[] = { 0, 255 };
      const float* ranges[] = { hranges, sranges};*/
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
JNIEXPORT void JNICALL
Java_com_example_myshite_MainActivity7_paraellegrabcut(JNIEnv *env, jobject thiz, jlong input_image,
                                                       jlong output_image) {
    // TODO: implement paraellegrabcut()
    Mat &img1 = *(Mat *) input_image;
    Mat &img2 = *(Mat *) output_image;
    clock_t start, stop;

    start = clock();
    img2 = cv::Mat::zeros(img1.size(), CV_8UC3);
//GaussianBlur(img1,img2,Size(5,5),0);
    // create 8 threads and use TBB
    // cv::parallel_for_(cv::Range(0, 8), Parallel_process(img1, img2, 5, 8));
    stop = clock();
    double b=(double)(stop - start)/CLOCKS_PER_SEC*1000 ;
    __android_log_print(ANDROID_LOG_INFO, "sometag", "\"Running time using \\'for\\':\" = %d", b);
}

float calcul_p(int p,int N)
{int i; float _p=1,float_N=(float)N;
    if (p==0)
        _p=(float)N;
    else {
        for (i=1;i<=p;i++)
            _p=_p*(1-((i*i)/(N*N)));

        _p=(_p*float_N)/(2*p+1);
    }
    return _p;
}
float calcul_ro(int p,int N)
{int i; float ro_p=1,float_N=(float)N;
    if (p==0)
        ro_p=N;
    else {
        for (i=1;i<=p;i++)
            ro_p=ro_p*(1-((i*i)/(N*N)));

        ro_p=(ro_p*float_N)/(2*p+1);
    }
    return ro_p;
}
///______________________________________________________________________________///
float calcul_tp(int x,int p,int N,float tp_1,float tp_2)
{float tp, float_N=(float)N;
    if (p==0)
        tp=1;
    else if(p==1)
        tp=(2*x+1-N)/N;
    else
    {  //tp=((2*p-1)*tp_1)-((p-1)*(1-(pow((float)(p-1),2)/pow(float_N,2)))*tp_2);
        tp=((2*p-1)*tp_1)-(((p-1)*(1-((p-1)*(p-1))/(N*N)))*tp_2);
        tp=tp/p;}
    return tp;
}
///______________________________________________________________________///
std::vector<float> chebychev_moment(Mat image,int N)
{int p,q,x=0,y=0,i=0,j,compt=0,hml=0; float rslt,ro_p=1,ro_q=1,tp,tq;
    std::vector<float> vect(55);
    float tp_moins_1[100][100], tp_moins_2[100][100], tq_moins_1[100][100],tq_moins_2[100][100];
///******************************************************************
///initialisation de tp_moins_1
    for(i=0;i<100;i++)
    {
        for(j=0;j<100;j++)
            tp_moins_1[i][j]=1;
        tq_moins_1[i][j]=1;
    }
///************************************************************************
    for (p=0;p<9;p++)
    {

        for(q=0;q<=9-p;q++)
        {

            ro_p=calcul_ro(p,N);
            ro_q=calcul_ro(q,N);
            ///************************************
            for (x=0;x<image.rows;x++)
            {   y=0;
                tp=calcul_tp(x,p,N, tp_moins_1[x][y], tp_moins_2[x][y]);
                tp_moins_2[x][y]=tp_moins_1[x][y];
                tp_moins_1[x][y]=tp;
                for(y=0;y<image.cols;y++)
                { if(image.at<int>(x,y)!=0)
                    {tq=calcul_tp(y,q,N,tq_moins_1[x][y],tq_moins_2[x][y]);
                        tq_moins_2[x][y]=tq_moins_1[x][y];
                        tq_moins_1[x][y]=tq;
                        rslt=rslt+tp*tq*image.at<int>(x,y);}
                }
            }
///************************
            rslt=rslt*(1/(ro_p*ro_q));
            // printf("rslt %d ,p=%d,q=%d, =%f \n",hml,p,q,rslt);
            vect[compt]=rslt;
            compt++;
            rslt=0;

        }}
    return vect;
}
float _t(int p, int x, int N)
{
    if (p == 0)
    {

        return 1.0;

    }
    else if (p == 1)
    {
        return (2 * x + 1 - N) * 1.0 / N * 1.0;
    }
    else
    {
        return ((2 * p - 1) * _t(1, x, N) * _t(p - 1, x, N) - (p - 1) * (1 - (pow(p - 1, 2) / pow(N, 2))) * _t(p - 2, x, N)) * 1.0 / p * 1.0;
    }
}

float _p(int q, int N)
{
    float temp =(float) N;
    for (int i = 1; i <= q; i++)
    {
        temp *= 1 - (pow(i, 2) / pow(N, 2));
    }
    return temp * 1.0 / (2 * q + 1) * 1.0;
}

float _T(int p, int q, Mat img1)
{ float tp , tp_moins_2=1,tp_moins_1=1,tq_moins_2=1,tq_moins_1=1,tq;
    float somme = 0.0;

    for(int x=0;x<img1.rows;x++){
        for (int y = 0; y < img1.cols; y++) {
            if(  (int) img1.at<unsigned char>(x, y)!=0) {

                somme += _t(p, x,img1.rows) * _t(q, y, img1.cols) *(int) img1.at<unsigned char>(x, y);
            }
        }
    }
    float r;
    // r=sin(CV_PI*2+p*CV_PI);
    float tt;

    tt=somme / (_p(q, img1.cols) * _p(p, img1.rows)) * 1.0;

    return tt;

}
float eculiddistance(float *a,float *b){
    float distance ;
    float somm=0.0;
    for(int j=0;j<10;j++){
        somm+=pow(a[j]-b[j],2);
    }
    distance=sqrt(somm);

}
extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myshite_MainActivity7_bychev(JNIEnv *env, jobject thiz, jlong input_image3) {
    // TODO: implement bychev()
    Mat &img1 = *(Mat *) input_image3;
    int compt=0;

    Mat dst;
    //resize(img_input, img_input, Size(), 0.25, 0.25);
    cvtColor( img1, dst, COLOR_RGB2GRAY);


    Mat edges;
    edges = sobel(dst);
    __android_log_print(ANDROID_LOG_ERROR, "sometag", "error 1");
    int ordre=3;float t=0.0;
    float vectt[4];
    int j;
    for (int i = 0, j = ordre; i <= ordre; i++, j--)
    {

        t = _T(i, j,dst);
        vectt[compt]=abs(t);
        __android_log_print(ANDROID_LOG_ERROR, "sometag", "\"cheychev\\':\" =  %f",  vectt[compt]);
        compt++;
    }
    jfloatArray arr = env->NewFloatArray( 4);
    env->SetFloatArrayRegion( arr, 0,4, ( jfloat * ) &vectt[0] );
    return arr;
}