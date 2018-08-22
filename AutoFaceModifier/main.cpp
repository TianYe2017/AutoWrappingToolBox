#include "face_point_detector.h"
#include "bilinear_interpolation.h"
#include "homographic_transformation.h"
#include "local_wrapping.h"
#include "smooth.h"
#include "global_wrapping.h"
#include "demo.h"

using namespace std;
using namespace dlib;



void main(void) 
{
	TestFPD();
	TestBI_BGR();
	Test_homographic_transformation();
	TestSmooth();
	TestMLS(WP_AFFINE);
	TestMLS(WP_SIMILARITY);
	TestMLS(WP_RIGID);
	TestLT();
	TestLS();
	TestLR();
	DEMO_BetterSkin_MonaLisa();
	DEMO_SilmOrExpand_MonaLisa();
	DEMO_BiggerEye_MonaLisa();
	DEMO_SmallerMouth_MonaLisa();
	DEMO_All();

	while (1) 
	{
		cv::waitKey(20);
	}

	return;
}