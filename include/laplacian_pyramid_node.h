#ifndef LAPLACIAN_NODE_H
#define LAPLACIAN_NODE_H

#include "node.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


class LaplacianPyramidNode: public Node{
	
	public:
		LaplacianPyramidNode(std::string id, cv::Mat layer0, int n_layers);
		void run();
		std::vector<cv::Mat> _layers;
		int _n_layers;
		void print_pyramid();
		std::vector<cv::Mat> get_output(){return _layers};
		
	private:
		void gen_next_level(cv::Mat, cv::Mat, int);
		cv::Mat _layer0;
		int _w0;
		int _h0;
};
#endif