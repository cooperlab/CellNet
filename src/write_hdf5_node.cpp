#include "write_hdf5_node.h"
#include "edge.h"
#define FILE_SIZE_LIMIT 2 * (unsigned long int)1000000000// ~2Gb
// NOTE: Here we are rounding down file size limit in order to avoid files with size over 2Gb

/* Consider dimensions as (numb_items, ..., channels, height, width) */
/* Min: 4D */
WriteHDF5Node::WriteHDF5Node(std::string id, std::string fname, std::vector<hsize_t> dim, std::string dataset_name): Node(id), _fname(fname), _dim(dim), _curr_size(0), _h(0), _w(0), _c(0), _file_buffer(), _f_count(0), _dataset_name(dataset_name), _el_cont(0){}

void *WriteHDF5Node::run(){

	// Initialize variables
	_h = *(_dim.rbegin());
	_w = *(_dim.rbegin()+1);

	_n_dim = _dim.size();
	_curr_size = 0;
	_f_count = 0;
	_file_buffer.clear();

	while(true){

		cv::Mat out;
		copy_from_buffer(out);
		if(!out.empty()){

			// If there is space in the buffer
			if(_curr_size + _w * _h * out.channels() * sizeof(H5::PredType::NATIVE_DOUBLE)/2 <= FILE_SIZE_LIMIT){

				// Copy mat to file buffer
				copy_mat(out);
			}
			else{
			// Write on disk and then copy matrix
				write_to_disk();
				copy_mat(out);
				_curr_size = 0;
				_file_buffer.clear();
			}

			_curr_size += _w * _h * out.channels() * sizeof(H5::PredType::NATIVE_DOUBLE)/2;
			out.release();
		}
		else if(_in_edges.at(0)->is_in_node_done()){

			// Write remaining matrices in buffer
			write_to_disk();
			std::cout << "Stopping Write Node" << std::endl;
			break;
		}
	}
	return NULL;
}

void WriteHDF5Node::copy_mat(cv::Mat out){

	// Get image
	std::vector<double> image;
	image.assign(out.datastart, out.dataend);

	// Concatenate buffers
	std::vector<double> new_buffer;

	new_buffer.reserve(_file_buffer.size() + image.size());
	new_buffer.insert( new_buffer.end(), _file_buffer.begin(), _file_buffer.end());
	new_buffer.insert( new_buffer.end(), image.begin(), image.end());

	_file_buffer = new_buffer;
	_el_cont++;
}

void WriteHDF5Node::write_to_disk(){

	// Set number of items
	int _dem = 1;
	for(int i = 1; i < _n_dim - 2; i++){
		_dem *= _dim[i];
	}
	_dim[0] = _el_cont / _dem;
	H5::DataSpace space(_n_dim, &_dim[0]);

	// Open file
	std::cout << "numb_items: " << std::to_string(_dim[0]) << std::endl;
    H5::H5File file(_fname + "_part_" + std::to_string(_f_count) + ".h5", H5F_ACC_TRUNC);

    // Create and write dataset
    H5::DataSet dataset = file.createDataSet(_dataset_name, H5::PredType::NATIVE_DOUBLE, space);
    dataset.write(&_file_buffer[0], H5::PredType::NATIVE_DOUBLE);

    // Initialize parameters
    _file_buffer.clear();
    _f_count++;
    _el_cont = 0;
    std::cout << "File size: " << _curr_size << std::endl; 
}