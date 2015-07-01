#ifndef _WRITE_PNG_NODE_H
#define _WRITE_PNG_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class WritePNGNode: public Node{

	public:
		WritePNGNode(std::string id, std::string fname, int mode);
		void *run();

  	private:
  		void write_to_disk(cv::Mat &out);
  		std::string _fname;
  		int _counter;
};
#endif