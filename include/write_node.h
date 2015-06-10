#ifndef _WRITE_NODE_H
#define _WRITE_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class WriteNode: public Node{

	public:
		WriteNode(std::string id, std::string fname);
		void *run();

  	private:
  		void write_to_disk(cv::Mat &out);
  		std::string _fname;
  		int _counter;
};
#endif