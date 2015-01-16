#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/gpu/gpu.hpp"


#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::gpu;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound, double higherBound) {

	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))

	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}

	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

static void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags = INTER_LINEAR,
	            			 int borderType = BORDER_CONSTANT, const Scalar& borderValue = Scalar())
{
	int width = src.cols;
	int height = src.rows;
	dst.create( height, width, CV_8UC1 );

	Mat mask = Mat::zeros(height, width, CV_8UC1);
	const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
         invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = min(BLOCK_SZ/2, height);
    int bw0 = min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
    for( x = 0; x < width; x += bw0 ) {
		int bw = min( bw0, width - x);
        int bh = min( bh0, height - y);

        Mat _XY(bh, bw, CV_16SC2, XY);
		Mat matA;
        Mat dpart(dst, Rect(x, y, bw, bh));

		for( y1 = 0; y1 < bh; y1++ ) {

			short* xy = XY + y1*bw*2;
            double X0 = M[0]*x + M[1]*(y + y1) + M[2];
            double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
            double W0 = M[6]*x + M[7]*(y + y1) + M[8];
            short* alpha = A + y1*bw;

            for( x1 = 0; x1 < bw; x1++ ) {

                double W = W0 + M[6]*x1;
                W = W ? INTER_TAB_SIZE/W : 0;
                double fX = max((double)INT_MIN, min((double)INT_MAX, (X0 + M[0]*x1)*W));
                double fY = max((double)INT_MIN, min((double)INT_MAX, (Y0 + M[3]*x1)*W));

				double _X = fX/double(INTER_TAB_SIZE);
				double _Y = fY/double(INTER_TAB_SIZE);

				if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
					mask.at<uchar>(y+y1, x+x1) = 1;

                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);

                xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
            }
        }

        Mat _matA(bh, bw, CV_16U, A);
        remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
    }
    }

	for( y = 0; y < height; y++ ) {
		const uchar* m = mask.ptr<uchar>(y);
		const uchar* s = prev_src.ptr<uchar>(y);
		uchar* d = dst.ptr<uchar>(y);
		for( x = 0; x < width; x++ ) {
			if(m[x] == 0)
				d[x] = s[x];
		}
	}
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
				  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
	prev_pts.clear();
	pts.clear();

	if(prev_kpts.size() == 0 || kpts.size() == 0)
		return;

	Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

	BFMatcher desc_matcher(NORM_L2);
	std::vector<DMatch> matches;

	desc_matcher.match(desc, prev_desc, matches, mask);

	prev_pts.reserve(matches.size());
	pts.reserve(matches.size());

	for(size_t i = 0; i < matches.size(); i++) {
		const DMatch& dmatch = matches[i];
		// get the point pairs that are successfully matched
		prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
		pts.push_back(kpts[dmatch.queryIdx].pt);
	}

	return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
				const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
				std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
	prev_pts_all.clear();
	prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

	pts_all.clear();
	pts_all.reserve(pts1.size() + pts2.size());

	for(size_t i = 0; i < prev_pts1.size(); i++) {
		prev_pts_all.push_back(prev_pts1[i]);
		pts_all.push_back(pts1[i]);
	}

	for(size_t i = 0; i < prev_pts2.size(); i++) {
		prev_pts_all.push_back(prev_pts2[i]);
		pts_all.push_back(pts2[i]);
	}

	return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f = flow.ptr<float>(y);
		pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
	}
}

void MatchFromFlow_copy(const Mat& prev_grey, const Mat& flow_x, const Mat& flow_y, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
	int width = prev_grey.cols;
	int height = prev_grey.rows;
	prev_pts.clear();
	pts.clear();

	const int MAX_COUNT = 1000;
	goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

	if(prev_pts.size() == 0)
		return;

	for(int i = 0; i < prev_pts.size(); i++) {
		int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
		int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

		const float* f_x = flow_x.ptr<float>(y);
		const float* f_y = flow_y.ptr<float>(y);
		pts.push_back(Point2f(x+f_x[x], y+f_y[y]));
	}
}

int main(int argc, char** argv)
{
	// IO operation
	const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ x  | xFlowFile    | flow_x | filename of flow x component }"
			"{ y  | yFlowFile    | flow_y | filename of flow x component }"
			"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
			"{ t  | type | 0 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id}"
			"{ s  | step  | 1 | specify the step for frame sampling}"
		};
	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	string xFlowFile = cmd.get<string>("xFlowFile");
	string yFlowFile = cmd.get<string>("yFlowFile");
	string imgFile = cmd.get<string>("imgFile");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");

	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow, cflow, human_mask, warp_flow, flow_x, flow_y;
	GpuMat frame_0, frame_1, flow_u, flow_v;
	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);
	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;

	setDevice(device_id);
	FarnebackOpticalFlow alg_farn;
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);

	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		frame_0.upload(prev_grey);
		frame_1.upload(grey);


        // GPU optical flow
		switch(type){
		case 0:
			alg_farn(frame_0,frame_1,flow_u,flow_v);
			break;
		case 1:
			alg_tvl1(frame_0,frame_1,flow_u,flow_v);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	        frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	        frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
			break;
		}

		flow_u.download(flow_x);
		flow_v.download(flow_y);

		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
		MatchFromFlow_copy(prev_grey, flow_x, flow_y, prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);
		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv);

		frame_0.upload(prev_grey);
		frame_1.upload(grey_warp);

        // GPU optical flow
		switch(type){
		case 0:
			alg_farn(frame_0,frame_1,flow_u,flow_v);
			break;
		case 1:
			alg_tvl1(frame_0,frame_1,flow_u,flow_v);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	        frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	        frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
			break;
		}

		flow_u.download(flow_x);
		flow_v.download(flow_y);

		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%04d.jpg",int(frame_num));

		Mat imgX_, imgY_;
		resize(imgX,imgX_,cv::Size(340,256));
		resize(imgY,imgY_,cv::Size(340,256));

		imwrite(xFlowFile + tmp,imgX_);
		imwrite(yFlowFile + tmp,imgY_);

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);
		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;
	}
	return 0;
}
