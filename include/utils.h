//
//	Copyright (c) 2015-2016, Emory University
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification, are
//	permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this list of
//	conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice, this list
// 	of conditions and the following disclaimer in the documentation and/or other materials
//	provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//	DAMAGE.
//
//
#include "hdf5.h"
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <dirent.h>
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
	void fill_data(int N, int num_elem, std::vector<std::vector<std::tuple<float, float>>> &cells_coordinates_set, std::vector<std::vector<int>> &shuffled_labels, std::vector<float> &x_centroid, std::vector<float> &y_centroid, std::vector<int> &labels, std::vector<float> &slide_idx);
	bool has_prefix(const std::string& s, const std::string& prefix);
	std::string get_image_name(std::string name, std::string image_path);
	void remove_slides(std::vector<std::string> &file_paths, std::vector<std::vector<std::tuple<float, float>>> &cells_coordinates_set, std::vector<std::vector<int>> &labels, std::vector<int> slides);
	void remove_slides(std::vector<std::string> &file_paths, std::vector<int> slides);
	std::vector<std::string> get_images_path(std::string path);
}

