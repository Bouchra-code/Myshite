//
// Created by DELL on 19/02/2020.
//

#ifndef MYSHITE_HISTOGRAM_HPP
#define MYSHITE_HISTOGRAM_HPP




#include "../../../../../../AppData/Local/Android/Sdk/ndk/16.1.4479499/sources/cxx-stl/gnu-libstdc++/4.9/include/limits"
#include "../../../../opencv/native/jni/include/opencv2/core.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/cvstd.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/types.hpp"
#include "../../../../opencv/native/jni/include/opencv2/core/mat.hpp"
#include <limits>
using namespace cv;
using namespace std;

namespace lbp {

// templated functions
    template <typename _Tp>
    void histogram_(const Mat& src, Mat& hist, int numPatterns, bool normed=false);

    template <typename _Tp>
    double chi_square_(const Mat& histogram0, const Mat& histogram1);

// non-templated functions
    void spatial_histogram(const Mat& src, Mat& spatialhist, int numPatterns, const Size& window, int overlap=0);
    void spatial_histogram(const Mat& src, Mat& spatialhist, int numPatterns, const Size& window, bool normed=false, int overlap=0);

// wrapper functions
    void spatial_histogram(const Mat& src, Mat& spatialhist, int numPatterns, int gridx=8, int gridy=8, int overlap=0);
    void histogram(const Mat& src, Mat& hist, int numPatterns, bool normed);
    double chi_square(const Mat& histogram0, const Mat& histogram1);

// Mat return type functions
    Mat histogram(const Mat& src, int numPatterns, bool normed);
    Mat spatial_histogram(const Mat& src, int numPatterns, const Size& window, int overlap=0);
    Mat spatial_histogram(const Mat& src, int numPatterns, int gridx=8, int gridy=8, int overlap=0);
}
#endif MYSHITE_HISTOGRAM_HPP
