#include "demo.h"



void DEMO_SilmOrExpand_MonaLisa()
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");

	cv::imshow("original", img);
	cv::waitKey(20);

	frontal_face_detector ffd = InitFaceDetector();
	shape_predictor pm = InitPoseDetector();
	std::vector<FACE> faces = ShapeDetect(img, ffd, pm, FPD_SELECT_MAX, false);

	int rows = img.rows - 1, cols = img.cols - 1;

	std::vector<std::vector<float>> original,target;
	std::vector<std::vector<float>> addition;
	addition = { { 0,0 },{ 0,cols / 2.0f },{ 0,(float)cols-0 },{ rows / 2.0f,0 },{ rows / 2.0f,(float)cols-0 },{ (float)rows,0 },{ (float)rows,cols / 2.0f },{ (float)rows,(float)cols } };
	for (int i = 0; i < addition.size(); i++)
	{
		original.push_back(addition[i]);
		target.push_back(addition[i]);
	}
	std::vector<float> ori, tar;
	for (int k = 0; k < faces.size(); k++) 
	{
		std::vector<float> center = {faces[k].allPoints[33][0],faces[k].allPoints[33][1]};
		std::vector<int> pindex = {0,16,13,3};
		for (int p = 0; p < pindex.size(); p++) 
		{
			ori = faces[k].allPoints[pindex[p]];
			tar = {center[0] - (center[0] - ori[0]) * 0.9f, center[1] - (center[1] - ori[1]) * 0.9f};
			original.push_back(ori);
			target.push_back(tar);
		}
	}
	
	WP_INPUT DATA;
	DATA.img = img;
	DATA.originalPoints = original;
	DATA.targetPoints = target;

	WRAPPING* wp = new WRAPPING();
	wp->LoadNewData(DATA, WP_RIGID, 1.0f);
	cv::Mat result = wp->MLS(WP_RIGID, 4.0f);

	cv::imwrite("./demo/MonoLisa_Silm.jpg",result);
	cv::imshow("processed", result);
	cv::waitKey(20);
}


void DEMO_BetterSkin_MonaLisa() 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);
	cv::Mat ans = SmoothSkin(img, 7, 3, 50.0f, 0, 0.7f, 128.0f);
	cv::imshow("processed", ans);
	cv::waitKey(20);
	cv::imwrite("./demo/butterSkin_MonaLisa.jpg", ans);
	cout << "done" << endl;
}

void DEMO_BiggerEye_MonaLisa() 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");

	cv::imshow("original", img);
	cv::waitKey(20);

	frontal_face_detector ffd = InitFaceDetector();
	shape_predictor pm = InitPoseDetector();
	FACE face = ShapeDetect(img, ffd, pm, FPD_SELECT_MAX, false)[0];

	std::vector<float> leftEye = { (face.allPoints[36][0] + face.allPoints[39][0]) / 2.0f,(face.allPoints[36][1] + face.allPoints[39][1]) / 2.0f };
	std::vector<float> rightEye = { (face.allPoints[42][0] + face.allPoints[45][0]) / 2.0f,(face.allPoints[42][1] + face.allPoints[45][1]) / 2.0f };

	cv::Mat result = Local_Scale(img, leftEye, -0.15f, 30.0f);
	result = Local_Scale(result, rightEye, -0.15f, 30.0f);

	cv::imshow("processed", result);
	cv::waitKey(20);
	cv::imwrite("./demo/biggerEyes_MonaLisa.jpg", result);
	cout << "done" << endl;
}

void DEMO_SmallerMouth_MonaLisa() 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);
	frontal_face_detector ffd = InitFaceDetector();
	shape_predictor pm = InitPoseDetector();
	FACE face = ShapeDetect(img, ffd, pm, FPD_SELECT_MAX, false)[0];

	std::vector<float> mouth = {(face.allPoints[48][0] + face.allPoints[54][0]) / 2.0f, (face.allPoints[48][1] + face.allPoints[54][1]) / 2.0f };

	cv::Mat result = Local_Scale(img, mouth, 0.16f, 30.0f);

	cv::imshow("processed", result);
	cv::waitKey(20);
	cv::imwrite("./demo/smallerMouth_MonaLisa.jpg", result);
	cout << "done" << endl;

}

void DEMO_All() 
{
	cv::Mat img = cv::imread("./MonaLisa.jpg");
	cv::imshow("original", img);
	cv::waitKey(20);

	img = SmoothSkin(img, 7, 3, 50.0f, 0, 0.7f, 128.0f);

	frontal_face_detector ffd = InitFaceDetector();
	shape_predictor pm = InitPoseDetector();


	std::vector<FACE> faces = ShapeDetect(img, ffd, pm, FPD_SELECT_MAX, false);
	int rows = img.rows - 1, cols = img.cols - 1;
	std::vector<std::vector<float>> original, target;
	std::vector<std::vector<float>> addition;
	addition = { { 0,0 },{ 0,cols / 2.0f },{ 0,(float)cols - 0 },{ rows / 2.0f,0 },{ rows / 2.0f,(float)cols - 0 },{ (float)rows,0 },{ (float)rows,cols / 2.0f },{ (float)rows,(float)cols } };
	for (int i = 0; i < addition.size(); i++)
	{
		original.push_back(addition[i]);
		target.push_back(addition[i]);
	}
	std::vector<float> ori, tar;
	for (int k = 0; k < faces.size(); k++)
	{
		std::vector<float> center = { faces[k].allPoints[33][0],faces[k].allPoints[33][1] };
		std::vector<int> pindex = { 0,16,13,3 };
		for (int p = 0; p < pindex.size(); p++)
		{
			ori = faces[k].allPoints[pindex[p]];
			tar = { center[0] - (center[0] - ori[0]) * 0.9f, center[1] - (center[1] - ori[1]) * 0.9f };
			original.push_back(ori);
			target.push_back(tar);
		}
	}

	WP_INPUT DATA;
	DATA.img = img;
	DATA.originalPoints = original;
	DATA.targetPoints = target;

	WRAPPING* wp = new WRAPPING();
	wp->LoadNewData(DATA, WP_RIGID, 1.0f);
	img = wp->MLS(WP_RIGID, 4.0f);

	FACE face = ShapeDetect(img, ffd, pm, FPD_SELECT_MAX, false)[0];
	std::vector<float> leftEye = { (face.allPoints[36][0] + face.allPoints[39][0]) / 2.0f,(face.allPoints[36][1] + face.allPoints[39][1]) / 2.0f };
	std::vector<float> rightEye = { (face.allPoints[42][0] + face.allPoints[45][0]) / 2.0f,(face.allPoints[42][1] + face.allPoints[45][1]) / 2.0f };
	img = Local_Scale(img, leftEye, -0.15f, 30.0f);
	img = Local_Scale(img, rightEye, -0.15f, 30.0f);

	std::vector<float> mouth = { (face.allPoints[48][0] + face.allPoints[54][0]) / 2.0f, (face.allPoints[48][1] + face.allPoints[54][1]) / 2.0f };
	cv::Mat result = Local_Scale(img, mouth, 0.16f, 30.0f);

	cv::imshow("processed", result);
	cv::waitKey(20);
	cv::imwrite("./demo/Better_MonaLisa.jpg", result);
	cout << "done" << endl;

}