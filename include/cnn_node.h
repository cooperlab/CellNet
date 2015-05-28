#ifndef _CNN_NODE_H
#define _CNN_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <openslide.h>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class CNNNode: public Node{

	public:
		CNNNode(std::string name, std::string id);
		void run();

  	private:

};
#endif