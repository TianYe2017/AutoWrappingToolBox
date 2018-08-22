#include "face_point_detector.h"

//this .cpp contains interface to convert an image with one or multi-faces to one or multi-sets of {68 key point of human face} 

using namespace std;
using namespace dlib;

void UpdateFace(FACE& face) 
{
	if (face.allPoints.size() != 68) 
	{
		cout << "ERROR: only " << face.allPoints.size() << " points in this cluster" << endl;
		return;
	}
	face.area = abs((face.allPoints[0][1] - face.allPoints[16][1]) * (face.allPoints[19][0] - face.allPoints[8][0]));
}

frontal_face_detector InitFaceDetector(void) 
{
	return get_frontal_face_detector();
}

shape_predictor InitPoseDetector(void) 
{
	shape_predictor pose_model;
	cout << "loading model..." << endl;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	return pose_model;
}

std::vector<FACE> ShapeDetect(cv::Mat img, frontal_face_detector ffd, shape_predictor pm, int flag = 0, bool debug = false)
{
	std::vector<FACE> result;
	frontal_face_detector detector = ffd;
	cv_image<bgr_pixel> cimg(img);
	// Detect faces 
	cout << "detecting face..." << endl;
	std::vector<rectangle> faces = detector(cimg);
	cout << "detecting pose..." << endl;
	// Find the pose of each face.
	std::vector<full_object_detection> shapes;
	for (unsigned long i = 0; i < faces.size(); ++i)
		shapes.push_back(pm(cimg, faces[i]));
	//Display ALL pose on the screen
	if (debug) 
	{
		cout << "painting..." << endl;
		if (!shapes.empty()) {
			for (int k = 0; k < shapes.size(); k++) 
			{
				for (int i = 0; i < 68; i++)
				{
					circle(img, cvPoint(shapes[k].part(i).x(), shapes[k].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
				}
			}		
		}
		else
		{
			cout << "WARNING: NO FACE IS DETECTED" << endl;
		}
		cv::imshow("DlibÌØÕ÷µã", img);
		cv::waitKey(20);
		cv::imwrite("./demo/test_fpd.jpg",img);
	}
	if (shapes.size() == 1)
	{
		FACE face;
		std::vector<std::vector<float>> tmp;
		std::vector<float> pos;
		for (int i = 0; i < 68; i++)
		{
			CvPoint p = cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y());
			pos = { (float)p.y, (float)p.x };
			tmp.push_back(pos);
		}
		face.allPoints = tmp;
		UpdateFace(face);
		result.push_back(face);
	}
	else if(shapes.size() > 1)
	{
		if (flag == 1) 
		{
			for (int k = 0; k < shapes.size(); k++) 
			{
				FACE face;
				std::vector<std::vector<float>> tmp;
				std::vector<float> pos;
				for (int i = 0; i < 68; i++)
				{
					CvPoint p = cvPoint(shapes[k].part(i).x(), shapes[k].part(i).y());
					pos = { (float)p.y, (float)p.x };
					tmp.push_back(pos);
				}
				face.allPoints = tmp;
				UpdateFace(face);
				result.push_back(face);
			}
		}
		else 
		{
			float max_area = -1.0f;
			for (int k = 0; k < shapes.size(); k++)
			{
				FACE face;
				std::vector<std::vector<float>> tmp;
				std::vector<float> pos;
				for (int i = 0; i < 68; i++)
				{
					CvPoint p = cvPoint(shapes[k].part(i).x(), shapes[k].part(i).y());
					pos = { (float)p.y, (float)p.x };
					tmp.push_back(pos);
				}
				face.allPoints = tmp;
				UpdateFace(face);
				if (face.area > max_area) 
				{
					max_area = face.area;
					result.clear();
					result.push_back(face);
				}
			}
		}
	}
	else
	{
		cout << "WARNING: NO FACE IS DETECTED" << endl;
	}
	return result;
}

void TestFPD(void) 
{
	frontal_face_detector ffd = InitFaceDetector();
	shape_predictor pm = InitPoseDetector();

	cv::Mat img = cv::imread("./MonaLisa.jpg");
	std::vector<FACE> faces = ShapeDetect(img, ffd, pm, FPD_SELECT_ALL, true);

	for (int i = 0; i < faces.size(); i++)
	{
		cout << "face #" << i << endl;
		for (int j = 0; j < 68; j++)
		{
			cout << j << " " << faces[i].allPoints[j][0] << " " << faces[i].allPoints[j][1] << endl;
		}
	}
}


