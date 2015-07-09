#include "hdf5.h"
#include <vector>
#include <string>
#include <iostream>

namespace utils{
	double get_time();
	void get_data(std::string _data_path, std::string dataset_name, std::vector<float> &data_out);
	void get_data(std::string _data_path, std::string dataset_name, std::vector<int> &data_out);
}

