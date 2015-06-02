#ifndef LAPLACIAN_NODE_H
#define LAPLACIAN_NODE_H

#include "node.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#define N_LAYERS 4

class LaplacianPyramidNode: public Node{
	
	public:
		LaplacianPyramidNode(std::string id);
		void *run();
		void init();
		int _n_layers;
		void print_pyramid(std::vector<cv::Mat> layers);
		
	private:
		void gen_next_level(cv::Mat, cv::Mat, std::vector<cv::Mat> *, int);
};
#endif