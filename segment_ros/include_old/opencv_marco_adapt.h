#ifndef _OPENCV3_MACRO_ADAPT_H
#define _OPENCV3_MACRO_ADAPT_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY (cv::COLOR_BGR2GRAY)
#endif

#ifndef CV_FILLED
#define CV_FILLED (cv::FILLED)
#endif 

#ifndef CV_CHAIN_APPROX_NONE
#define CV_CHAIN_APPROX_NONE (cv::CHAIN_APPROX_NONE)
#endif

#ifndef CV_LOAD_IMAGE_ANYCOLOR
#define CV_LOAD_IMAGE_ANYCOLOR (cv::IMREAD_ANYCOLOR)
#endif

#ifndef CV_LOAD_IMAGE_ANYDEPTH
#define CV_LOAD_IMAGE_ANYDEPTH (cv::IMREAD_ANYDEPTH)
#endif


#endif