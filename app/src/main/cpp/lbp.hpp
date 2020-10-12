//
// Created by DELL on 19/02/2020.
//

#ifndef MYSHITE_LBP_HPP
#define MYSHITE_LBP_HPP





#include "../../../../../../AppData/Local/Android/Sdk/ndk/16.1.4479499/sources/cxx-stl/gnu-libstdc++/4.9/include/limits"
#include "../../../../opencv/native/jni/include/opencv2/core.hpp"
#include "../../../../../../AppData/Local/Android/Sdk/ndk/16.1.4479499/sources/cxx-stl/gnu-libstdc++/4.9/include/limits"
#include "../../../../opencv/native/jni/include/opencv2/core/cvstd.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/mat.hpp"

using namespace cv;
using namespace std;

namespace lbp {

// templated functions
    template <typename _Tp>
    void OLBP_(const cv::Mat& src, cv::Mat& dst);

    template <typename _Tp>
    void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

    template <typename _Tp>
    void VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

// wrapper functions
    void OLBP(const Mat& src, Mat& dst);
    void ELBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);
    void VARLBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);

// Mat return type functions
    Mat OLBP(const Mat& src);
    Mat ELBP(const Mat& src, int radius = 1, int neighbors = 8);
    Mat VARLBP(const Mat& src, int radius = 1, int neighbors = 8);

}

#endif MYSHITE_LBP_HPP
