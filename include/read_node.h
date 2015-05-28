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
		ReadNode(std::string id, std::string image_path, std::vector<std::tuple<int, int>> cells_coordinates);
		void show_entire_image();
		void show_cropped_cells();
		void crop_cells();
		void run();
		std::vector<cv::Mat> get_output(){return _extracted_images};

  	private:
  		void open_image();
  		std::vector<cv::Mat> _extracted_images;
  		std::string _image_path;
  		cv::Mat _entire_image;
  		std::vector<std::tuple<int, int>> cells_coordinates;
};
#endif