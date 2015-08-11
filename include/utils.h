#include "hdf5.h"
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace utils{
	double get_time();
	void get_data(std::string _data_path, std::string dataset_name, std::vector<float> &data_out);
	void get_data(std::string _data_path, std::string dataset_name, std::vector<int> &data_out);
	void get_data(std::string data_path, std::string dataset_name, std::vector<std::string> &data_out);
	void resize_all(std::vector<cv::Mat> &layers, cv::Size size);
	std::vector<cv::Mat> merge_all(std::vector<cv::Mat> &layers);
}

