#ifndef _DEBUG_NODE_H
#define _DEBUG_NODE_H

#include "node.h"
#include <vector>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class DebugNode: public Node{

	public:
		DebugNode();
		void set_target(cv::Mat target){};
  	private:
  		std::vector<cv::Mat> _output;
};
#endif