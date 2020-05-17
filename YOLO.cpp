#include <stdio.h>
#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <cstring>
#include "yolo.h"
using namespace std;

namespace YOLO{ 
	faceDetection::faceDetection(const string& modelWeights, const string modelConf) {
		net = cv::dnn::readNet(modelConf, modelWeights);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		names = net.getUnconnectedOutLayersNames();
		
		lnames = net.getLayerNames();
		lastLayer = net.getLayer(static_cast<unsigned int>(lnames.size()));
	}

	void faceDetection::clear() {
		lnames.clear();
		names.clear();
	}
	void faceDetection::InitInput(cv::Mat& img) {
		auto size = cvSize(480, 480);
		cv::dnn::blobFromImage(img, blob, 1 / 255.0, size, cv::Scalar(0, 0, 0), true, false);
		net.setInput(blob);
	}
	void faceDetection::detectFace(cv::Mat& img,cv::Mat & face) {
		net.forward(out, names);
		if (lastLayer->type.compare("Region") == 0) {
			for (auto& outIter : out) {
				float* data = (float*)outIter.data;
				for (int j = 0; j < outIter.rows; j++, data += outIter.cols) {
					cv::Mat scores = outIter.row(j).colRange(5, outIter.cols);
					double confidence;
					classIdPoint.x = 0;
					classIdPoint.y = 0;
					minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
					if (confidence > 0.2) {
						int centerX = (int)(data[0] * img.cols);
						int centerY = (int)(data[1] * img.rows);
						int width = (int)(data[2] * img.cols);
						int height = (int)(data[3] * img.rows);
						cv::Rect roi;
						roi.x = centerX - width / 2;
						roi.y = centerY - height / 2;
						roi.width = width;
						roi.height = height;
						cv::Mat crop = img(roi);
						cv::Mat outImg;
						cv::resize(crop, outImg, cvSize(256, 256));
						face = outImg;

						
					}
				}
			}
		}

	}

 }