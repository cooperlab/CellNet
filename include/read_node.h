#ifndef _READ_NODE_H
#define _READ_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <openslide.h>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class ReadNode: public Node{

	public:
		ReadNode(std::string id, std::vector<std::string> image_paths, std::vector<std::tuple<int, int>> cells_coordinates);
		void show_entire_image(cv::Mat);
		void show_cropped_cells(std::vector<cv::Mat> extracted_images);
		std::vector<cv::Mat> crop_cells(cv::Mat entire_image);
		void *run();
		void init();
		bool get_output(std::vector<cv::Mat> &out);

  	private:
  		cv::Mat open_image(std::string image_path);
  		std::vector<std::string> _image_paths;
  		std::vector<std::tuple<int, int>> _cells_coordinates;
};
#endif