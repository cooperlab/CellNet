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
#include <opencv2/imgproc/imgproc.hpp>

class ReadNode: public Node{

	public:
		ReadNode(std::string id, std::vector<std::string> image_paths, std::vector<std::vector<std::tuple<double, double>>> _cells_coordinates_set, std::vector<double> labels, int mode);
		void show_entire_image(cv::Mat);
		void show_cropped_cells(std::vector<cv::Mat> extracted_images);
		std::vector<cv::Mat> crop_cells(cv::Mat entire_image, std::vector<std::tuple<double, double>> cells_coordinates);
		void *run();
		void init();
		bool get_output(std::vector<cv::Mat> &out);

  	private:
  		int get_input();
  		int get_layer(openslide_t *oslide);
  		cv::Mat open_image(std::string image_path);
  		std::vector<std::string> _image_paths;
  		std::vector<std::vector<std::tuple<double, double>>> _cells_coordinates_set;
  		int i_ptr;
};
#endif