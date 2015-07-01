#include "hdf5.h"
#include <vector>
#include <string>
#include <iostream>

namespace utils{
	double get_time();
	void get_data(std::string _data_path, std::string dataset_name, std::vector<double> &data_out);

}

