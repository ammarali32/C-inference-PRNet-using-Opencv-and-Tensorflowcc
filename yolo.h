#pragma once
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include<iostream>
#include<cmath>
using namespace std;
using namespace cv;
using namespace dnn;

namespace YOLO {
	class faceDetection {
	private:
		Point classIdPoint;
		Mat blob;
		vector<Mat> out;
		dnn::Net net;
		Ptr<dnn::Layer> lastLayer;
		vector <string> names;
		vector <string> lnames;
	public:
		explicit faceDetection(const string& modelWeights = "YOLO.weights", const string modelConf = "YOLO.cfg");
		void InitInput(cv::Mat& img);
		void detectFace(cv::Mat& img, cv::Mat& face);
		void clear();
	};
}