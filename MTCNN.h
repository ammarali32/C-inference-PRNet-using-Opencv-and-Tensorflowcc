#pragma once
#include<opencv2/dnn/dnn.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace dnn;

class MTCNN {
private:
	const string modelName = "res10_300x300_ssd_iter_140000.caffemodel";
	const string modelConf = "deploy.prototxt";
	Net net;
	Mat blob;
	vector<Mat> out;
	Point classIdPoint;
public:
	void initModel() {
		net = cv::dnn::readNetFromCaffe(modelConf, modelName);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	void InitInput(cv::Mat& img) {
		auto size = Size(300, 300);
		resize(img, img, size);
		cv::dnn::blobFromImage(img, blob, 1.0, size, cv::Scalar(104.0, 177.0, 123.0), false, false);
		net.setInput(blob);
	}
	void detectFace(cv::Mat& img, cv::Mat& face) {
		net.forward(out);
		for (auto& outIter : out) {
			classIdPoint.x = 0;
			classIdPoint.y = 0;
			float* data = (float*)outIter.data;
			int startx = (int)(data[3] * img.cols);
			int starty = (int)(data[4] * img.rows);
			int width = (int)(data[5] * img.cols);
			int height = (int)(data[6] * img.rows);
			cv::Rect roi;
			roi.x = startx;
			roi.y = starty;
			roi.width = width - startx;
			roi.height = height - starty;
			cv::Mat crop = img(roi);
			cv::Mat outImg;
			cv::resize(crop, outImg, Size(256, 256));
			face = outImg;
			
		}
	}
};