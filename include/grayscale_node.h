#ifndef _GRAYSCALE_NODE_H
#define _GRAYSCALE_NODE_H

#include "node.h"
#include <vector>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class GrayScaleNode: public Node{

	public:
		GrayScaleNode(std::string id);
		void *run();
};
#endif