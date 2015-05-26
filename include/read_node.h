#ifndef _READ_NODE_H
#define _READ_NODE_H

#include "node.h"
#include <vector>
#include <tuple>
#include <openslide.h>
#include <iostream>
#include <glib.h>
#include <cv.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>

class ReadNode: public Node{

	public:
		ReadNode(std::string name, boost::uuids::uuid id, std::string image_path);
		void show_entire_image();
		void show_cropped_cells();
		void crop_cells(std::vector<std::tuple<int, int>> cells_coordinates);
		void run();

  	private:
  		void open_image();
  		std::vector<cv::Mat> _extracted_images;
  		std::string _image_path;
  		cv::Mat _entire_image;
};
#endif