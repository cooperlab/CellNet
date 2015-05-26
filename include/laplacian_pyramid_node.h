#ifndef LAPLACIAN_NODE_H
#define LAPLACIAN_NODE_H

#include "node.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>


class LaplacianPyramidNode: public Node{
	
	public:
		LaplacianPyramidNode(std::string name, boost::uuids::uuid id, cv::Mat layer0, int n_layers);
		void run();
		std::vector<cv::Mat> _layers;
		int _n_layers;
		void print_pyramid();
		
	private:
		void gen_next_level(cv::Mat, cv::Mat, int);
		cv::Mat _layer0;
		int _w0;
		int _h0;
};
#endif