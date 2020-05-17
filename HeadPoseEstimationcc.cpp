
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "facedata.h"
#include "predictor.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include "yolo.h"
#include "MTCNN.h"

using namespace prnet;
using namespace cv;
using namespace YOLO;
template <typename T>
inline T clamp(T f, T fmin, T fmax) {
	return std::max(std::min(fmax, f), fmin);
}
static bool Mat2Image(cv::Mat& img, Image<float>& image) {
	int width, height, channels;
	width = img.cols;
	height = img.rows;
	channels = 3;
	std::vector<float> data;
	std::cout << height << " " << width << " from functions " << std::endl;
	data.resize(width * height * channels);
	std::vector<cv::Mat> three_channels;
	split(img,three_channels);
	int id = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			data[id++] = three_channels[0].at<uchar>(i, j);
			data[id++] = three_channels[1].at<uchar>(i, j);
			data[id++] = three_channels[2].at<uchar>(i, j);
		}
	}
	image.create(size_t(width), size_t(height), 3);
	image.foreach([&](size_t x, size_t y, size_t c, float& v) {
		float p = static_cast<float>(
			data[(y * size_t(width) + x) * size_t(channels) + c]) /
			255.f;
		v = std::pow(p, 2.2f);
		});
	return true;

}

static bool Image2Mat(Mat& img, Image<float>& image,
	const float scale = 1.0f) {
	const size_t height = image.getHeight();
	const size_t width = image.getWidth();
	const size_t channels = image.getChannels();
	std::vector<unsigned char> data(height * width * channels);
	image.foreach([&](size_t x, size_t y, size_t c, float& v) {
		data[(y * width + x) * channels + c] =
			static_cast<unsigned char>(clamp(scale * v * 255.f, 0.0f, 255.0f));
		});
	std::vector<cv::Mat> three_channels;
	split(img, three_channels);
	int id = 0;
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			three_channels[0].at<uchar>(i, j)= data[id++] ;
			three_channels[1].at<uchar>(i, j)= data[id++];
			three_channels[2].at<uchar>(i, j)= data[id++];
		}
	}
	merge(three_channels, img);
	return true;
}


static void RemapPosition(Image<float>* posImage, const float scale,
	const float shift_x, const float shift_y) {
	size_t n = posImage->getWidth() * posImage->getHeight();

	for (size_t i = 0; i < n; i++) {
		float x = posImage->getData()[3 * i + 0];
		float y = posImage->getData()[3 * i + 1];
		float z = posImage->getData()[3 * i + 2];

		posImage->getData()[3 * i + 0] = x * scale + shift_x;
		posImage->getData()[3 * i + 1] = y * scale + shift_y;
		posImage->getData()[3 * i + 2] =
			z * scale;  
	}
}

static void FindEulerAngles(const Image<float>& croppedImage,
	const Image<float>& posImage, const FaceData& face_data,
	Image<float>* out_img, float radius = 1.f) {
	*out_img = croppedImage; 
	vector<Point3f> face;
	vector<Point2f> object;
	cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
	cameraMatrix.at<double>(0, 0) = 480;
	cameraMatrix.at<double>(0, 1) = 0;
	cameraMatrix.at<double>(0, 2) = 320;
	cameraMatrix.at<double>(1, 0) = 0;
	cameraMatrix.at<double>(1, 1) = 640;
	cameraMatrix.at<double>(1, 2) = 240;
	cameraMatrix.at<double>(2, 0) = 0;
	cameraMatrix.at<double>(2, 1) = 0;
	cameraMatrix.at<double>(2, 2) = 1;
	
	cv::Mat distCoeffs(5, 1, cv::DataType<double>::type);
	distCoeffs.at<double>(0) = 0.070834633684407095;
	distCoeffs.at<double>(1) = 0.069140193737175351;
	distCoeffs.at<double>(2) = 0.0;
	distCoeffs.at<double>(3) = 0.0; 
	distCoeffs.at<double>(4) = -1.3073460323689292;
	cv::Mat rvec(3, 1, cv::DataType<double>::type);
	cv::Mat tvec(3, 1, cv::DataType<double>::type);

	///////
	face.push_back(Point3f(0.000000, -7.415691, 4.070434));//chin
	face.push_back(Point3f(6.825897, 6.760612, 4.402142));//left eyebow left corner
	face.push_back(Point3f(1.330353, 7.122144, 6.903745));//left eyebow right corner
	face.push_back(Point3f(-1.330353, 7.122144, 6.903745));//right eyebow left corner
	face.push_back(Point3f(-6.825897, 6.760612, 4.402142));//right eyebow right corner
	face.push_back(Point3f(2.005628, 1.409845, 6.165652));//left tip nose
	face.push_back(Point3f(-2.005628, 1.409845, 6.165652));//right tip nose
	face.push_back(Point3f(5.311432, 5.485328, 3.987654));//left eye left corner
	face.push_back(Point3f(1.789930, 5.393625, 4.413414));//left eye right corner
	face.push_back(Point3f(-1.789930, 5.393625, 4.413414));//right eye left corner
	face.push_back(Point3f(-5.311432, 5.485328, 3.987654));//right eye right corner
	face.push_back(Point3f(2.774015, -2.080775, 5.048531));//left mouth corner
	face.push_back(Point3f(-2.774015, -2.080775, 5.048531));//right mouth corner
	face.push_back(Point3f(0.000000, -3.116408, 6.097667));//bottom lip
	///////

	const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
	const int ksize = int(std::ceil(radius));
	for (size_t i = 0; i < n_pt; i++) {
		const uint32_t x_idx = face_data.uv_kpt_indices[i];
		const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];
		const int x = int(posImage.fetch(x_idx, y_idx, 0));
		const int y = int(posImage.fetch(x_idx, y_idx, 1));
		if (i == 8 || i == 17 || i == 21 || i == 22 ||
			i == 26 || i == 31 || i == 35 || i == 36 ||
			i == 39 || i == 42 || i == 45 || i == 48 ||
			i == 54 || i == 57) {
			object.push_back(Point2f(x, y));
		}
	}
	Mat rotation_mat(3, 3, cv::DataType<double>::type);
	cv::solvePnP(face, object, cameraMatrix, distCoeffs, rvec, tvec);
	Rodrigues(rvec, rotation_mat);
	cv::Mat P(3, 4, cv::DataType<double>::type);

	P.at<double>(0, 0) = rotation_mat.at<double>(0, 0);
	P.at<double>(1, 0) = rotation_mat.at<double>(1, 0);
	P.at<double>(2, 0) = rotation_mat.at<double>(2, 0);
	P.at<double>(0, 1) = rotation_mat.at<double>(0, 1);
	P.at<double>(1, 1) = rotation_mat.at<double>(1, 1);
	P.at<double>(2, 1) = rotation_mat.at<double>(2, 1);
	P.at<double>(0, 2) = rotation_mat.at<double>(0, 2);
	P.at<double>(1, 2) = rotation_mat.at<double>(1, 2);
	P.at<double>(2, 2) = rotation_mat.at<double>(2, 2);
	P.at<double>(0, 3) = tvec.at<double>(0);
	P.at<double>(1, 3) = tvec.at<double>(1);
	P.at<double>(2, 3) = tvec.at<double>(2);

	cv::Mat K(3, 3, cv::DataType<double>::type); 
	cv::Mat R(3, 3, cv::DataType<double>::type); 
	cv::Mat T(4, 1, cv::DataType<double>::type);
	cv::Mat Euler(1, 3, cv::DataType<double>::type);
	cv::Mat a1(3, 3, cv::DataType<double>::type);
	cv::Mat a2(3, 3, cv::DataType<double>::type);
	cv::Mat a3(3, 3, cv::DataType<double>::type);
	decomposeProjectionMatrix(P,K,R,T,a1,a2,a3,Euler);
	cout << Euler<< endl;
	}


static void DrawLandmark(const Image<float>& croppedImage,
	const Image<float>& posImage, const FaceData& face_data,
	Image<float>* out_img, float radius = 1.f) {
	*out_img = croppedImage;  
	const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
	
	const int ksize = int(std::ceil(radius));
	for (size_t i = 0; i < n_pt; i++) {
		const uint32_t x_idx = face_data.uv_kpt_indices[i];
		const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];
		const int x = int(posImage.fetch(x_idx, y_idx, 0));
		const int y = int(posImage.fetch(x_idx, y_idx, 1));
	
		for (int rx = -ksize; rx <= ksize; rx++) {
			for (int ry = -ksize; ry <= ksize; ry++) {
				if (radius < float(rx * rx + ry * ry)) {
					continue;
				}
				if (((x + rx) < 0) || ((x + rx) >= out_img->getWidth()) ||
					((y + ry) < 0) || ((y + ry) >= out_img->getWidth())) {
					continue;
				}
				out_img->fetch(size_t(x + rx), size_t(y + ry), 0) = 0.f;
				out_img->fetch(size_t(x + rx), size_t(y + ry), 1) = 1.f;
				out_img->fetch(size_t(x + rx), size_t(y + ry), 2) = 0.f;
				
			}
		}
	}
}



int main() {


	cv::Mat img;
	Mat face;
	//using YOLO to detect the face
	faceDetection detector;
	//using resnset to detect the face
	/*MTCNN detect;
	detector.initModel();*/
	std::string graphName = "model.pb";
	std::string dataName = "Data";
	FaceData face_data;
	Image<float> croppedImage;
	TensorflowPredictor predictor;
	predictor.load(graphName, "Placeholder",
		"resfcn256/Conv2d_transpose_16/Sigmoid");
	Image<float> modelOutput;
	Image<float> outputImage;
	Size size(256, 256);
	//img = imread("test.jpg.JPG", IMREAD_COLOR);
	//Using Dlib to detect the face
	//dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	
	

	if (!LoadFaceData(dataName + "/uv-data", &face_data)) {
		std::cout<< "Unable to load Face data" << std::endl;
		return -1;
	}
	//crop image
	//VideoCapture cap(0);
	
	while (true) {

		img = imread("test.jpg.JPG");
		if (img.empty())
		{
			std::cout << "Could not read the image: " << std::endl;
			return 1;
		}
		cv::resize(img, img, size);
		detector.InitInput(img);
		detector.detectFace(img, face);
		/*dlib::cv_image<dlib::bgr_pixel> cimg(img);
		std::vector<dlib::rectangle> dets = detector(cimg);
		if (dets.size() == 0)continue;
		cv::Rect roi = dlibRectangleToOpenCV(dets[0]);
		cv::Mat crop = img(roi);
		cv::Mat outImg;
		cv::resize(crop, outImg, cvSize(256, 256));
		face = outImg;*/
		Mat2Image(face, croppedImage);
		float scaleCropping = 1.f, xCropShift = 0.f, yCropShift = 0.f;



		//std::cout << "neural network running... " << std::endl << std::flush;
		auto startT = std::chrono::system_clock::now();
		predictor.predict(croppedImage, modelOutput);
		auto endT = std::chrono::system_clock::now();
		std::chrono::duration<double, std::milli> ms = endT - startT;
		//std::cout << "Time required = " << ms.count() << " ms " << std::endl;

		const float kMaxPos = modelOutput.getWidth() * 1.1f;
		Image<float> posImage = modelOutput;
		RemapPosition(&posImage, scaleCropping * kMaxPos, xCropShift, yCropShift);
		Image<float> color_img = croppedImage;

		FindEulerAngles(color_img, posImage, face_data, &outputImage);
		DrawLandmark(color_img, posImage, face_data, &outputImage);
		Mat res = img;
		Image2Mat(res, outputImage);
		imshow("res", res);
		char c = (char)waitKey(25);
		if (c == 27)
			break;
	}
	destroyAllWindows();
	return 0;
}