#ifndef LAPLACIAN_NODE_H
#define LAPLACIAN_NODE_H

#include "node.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


class LaplacianPyramidNode: public Node{
	
	public:
		LaplacianPyramidNode(std::string id, int n_layers);
		void run();
		void set_target(cv::Mat target);
		std::vector<cv::Mat> _layers;
		int _n_layers;
		void print_pyramid();
		bool get_output(std::vector<cv::Mat> &out);
		
	private:
		void gen_next_level(cv::Mat, cv::Mat, int);
		cv::Mat _layer0;
		int _w0;
		int _h0;
};
#endif