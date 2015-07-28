#ifndef _READ_NODE_JPG_H
#define _READ_NODE_JPG_H

#include "node.h"
#include <vector>
#include <tuple>
#include <openslide.h>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ReadNodeJPG: public Node{

	public:
		ReadNodeJPG(std::string id, std::vector<std::string> image_paths, std::vector<std::vector<int>> input_labels, int mode);
		void *run();
		void init();

  	private:
  		int get_input();
  		cv::Mat open_image(std::string image_path);
  		std::vector<std::string> _image_paths;
  		int i_ptr;
  		std::vector<std::vector<int>> _input_labels;
};
#endif
