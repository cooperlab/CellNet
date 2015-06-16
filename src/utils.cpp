#include "utils.h"

namespace utils{
	// Extract information from HDF5 file
	void get_data(std::string data_path, std::string dataset_name, std::vector<double> &data_out){

		//  Declare variables 
		const char *fname = data_path.c_str();
		hid_t  file_id , dataset_id , dataspace_id , file_dataspace_id;
		hsize_t* dims;
		hssize_t  num_elem;
		int rank;
		int ndims;

		// Open  existing  HDF5  file 
		file_id = H5Fopen(fname , H5F_ACC_RDONLY , H5P_DEFAULT);

		// Open  existing  dataset
		dataset_id = H5Dopen(file_id , dataset_name.c_str(), H5P_DEFAULT);

		//  Determine  dataset  parameters
		file_dataspace_id = H5Dget_space(dataset_id);
		rank = H5Sget_simple_extent_ndims(file_dataspace_id);
		dims = (hsize_t*)  malloc(rank *sizeof(hsize_t));
		ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, NULL);
		//std::cout << "rank: " << std::to_string(rank) << std::endl;

		// Allocate  matrix
		num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
		data_out.reserve(num_elem); // = (double*)malloc(num_elem *sizeof(double));
		//std::cout << "num_elem: " << std::to_string(num_elem) << std::endl;

		// Create  dataspace
		dataspace_id = H5Screate_simple(rank , dims , NULL);

		// Read  matrix  data  from  file 
		H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, file_dataspace_id , H5P_DEFAULT , &data_out[0]);

		//  Release  resources  and  close  file 
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);
		H5Sclose(file_dataspace_id);
		H5Fclose(file_id);
		free(dims);
	}
}