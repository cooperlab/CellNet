#ifndef _READ_JPG_NODE_H
#define _READ_JPG_NODE_H

#include "node.h"
#include <vector>
//#include <tuple>
#include <iostream>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ReadJPGNode: public Node{

	public:
		ReadJPGNode(std::string id, std::vector<std::string> slides_name, int mode, std::string imagePath);
		void *run();
		void init();

  	private:
  		int get_input();
  		void open_images(std::string image_path);
  	
		std::vector<std::string> _slides_name;
  		int i_ptr;
	    std::vector<cv::Mat> _input_data;
  		std::vector<int> _input_labels;
		std::string	_imagePath;
};
#endif
