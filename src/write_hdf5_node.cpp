#include "write_hdf5_node.h"
#include "edge.h"
#define FILE_SIZE_LIMIT 2 * (unsigned long int)1000000000// ~2Gb
// NOTE: Here we are rounding down file size limit in order to avoid files with size over 2Gb

/* Consider dimensions as (numb_items, ..., channels, height, width) */
/* Min: 4D */
WriteHDF5Node::WriteHDF5Node(std::string id, std::string fname, std::vector<hsize_t> dim, std::string dataset_name, std::vector<double> labels): Node(id), _fname(fname), _dim(dim), _curr_size(0), _h(0), _w(0), _c(0), _file_buffer(), _f_count(0), _label_count(0), _dataset_name(dataset_name), _el_cont(0), _labels(labels){}

void *WriteHDF5Node::run(){

	// Initialize variables
	_h = *(_dim.rbegin());
	_w = *(_dim.rbegin()+1);

	_n_dim = _dim.size();
	_curr_size = 0;
	_f_count = 0;
	_label_count = 0;
	_file_buffer.clear();

	while(true){

		cv::Mat out;
		copy_from_buffer(out);
		if(!out.empty()){

			// If there is space in the buffer
			if(_curr_size + _w * _h * out.channels() * 2 *  sizeof(H5T_NATIVE_DOUBLE) <= FILE_SIZE_LIMIT){

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

			_curr_size += _w * _h * out.channels() * 2* sizeof(H5T_NATIVE_DOUBLE);
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

	// Check if mat is continuous
	if(!out.isContinuous()){
		out = out.clone();
	}
	image.assign(out.datastart, out.dataend);

	// Concatenate buffers
	std::vector<double> new_buffer;
	new_buffer.reserve(_file_buffer.size() + image.size());
	new_buffer.insert( new_buffer.end(), _file_buffer.begin(), _file_buffer.end());
	new_buffer.insert( new_buffer.end(), image.begin(), image.end());
	_file_buffer = new_buffer;
	_el_cont += out.channels();
}

void WriteHDF5Node::write_to_disk(){

	// Set number of items
	int _dem = 1;
	for(int i = 1; i < _n_dim - 2; i++){
		_dem *= _dim[i];
	}

	int numb_samples = _el_cont / _dem;
	_dim[0] = numb_samples;
	hid_t space = H5Screate_simple(_n_dim, &_dim[0], NULL);

	// Open file
	std::cout << "numb_items: " << std::to_string(_dim[0]) << std::endl;
    std::string full_fname = _fname + "_part_" + std::to_string(_f_count) + ".h5";
    hid_t file =  H5Fcreate(full_fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create and write dataset
    hid_t dataset = H5Dcreate2(file, _dataset_name.c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &_file_buffer[0]);

    // Write label (# Labels, 1)
    std::vector<hsize_t> label_dim;
    label_dim.push_back(numb_samples);
    //label_dim.push_back(1);

    // Create label space
    hid_t label_space = H5Screate_simple(1, &label_dim[0], NULL);

    // Get range of labels to write
    std::vector<double>::const_iterator first = _labels.begin() + _label_count;
	std::vector<double>::const_iterator last = _labels.begin() + _label_count + numb_samples;
	std::vector<double> sub_labels(first, last);

    // Create and write labels
    hid_t label_dataset = H5Dcreate2(file, "labels", H5T_NATIVE_DOUBLE, label_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(label_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &sub_labels[0]);

    // Update parameters
    _label_count += numb_samples;
    _file_buffer.clear();
    _f_count++;
    _el_cont = 0;
    std::cout << "File size: " << _curr_size << std::endl; 
}