#include "utils.h"
#include <sys/time.h>

typedef struct 
{
	int var;
	int str_null_term;
	int set_ascii;
	char *c_str;
}H5string;

namespace utils{

	// Get current time
	double get_time(void)
	{
		double now_time;
		struct timeval  etstart;
		struct timezone tzp;

		if (gettimeofday(&etstart, &tzp) == -1)
			perror("Error: calling gettimeofday() not successful.\n");

	    now_time = ((double)etstart.tv_sec) +              /* in seconds */
	               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
		return now_time;
	}

	// Extract information from HDF5 file
	void get_data(std::string data_path, std::string dataset_name, std::vector<float> &data_out){

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
		std::vector<float> data_out_tmp(num_elem);
		// = (double*)malloc(num_elem *sizeof(double));
		//std::cout << "num_elem: " << std::to_string(num_elem) << std::endl;

		// Create  dataspace
		dataspace_id = H5Screate_simple(rank , dims , NULL);

		// Read  matrix  data  from  file 
		H5Dread(dataset_id, H5T_NATIVE_FLOAT, dataspace_id, file_dataspace_id , H5P_DEFAULT , &data_out_tmp[0]);
		data_out = data_out_tmp;
		
		//  Release  resources  and  close  file 
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);
		H5Sclose(file_dataspace_id);
		H5Fclose(file_id);
		free(dims);
	}

	// Extract information from HDF5 file
	void get_data(std::string data_path, std::string dataset_name, std::vector<int> &data_out){

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
		std::vector<int> data_out_tmp(num_elem);
		// = (double*)malloc(num_elem *sizeof(double));
		//std::cout << "num_elem: " << std::to_string(num_elem) << std::endl;

		// Create  dataspace
		dataspace_id = H5Screate_simple(rank , dims , NULL);

		// Read  matrix  data  from  file 
		H5Dread(dataset_id, H5T_NATIVE_INT, dataspace_id, file_dataspace_id , H5P_DEFAULT , &data_out_tmp[0]);
		data_out = data_out_tmp;
		
		//  Release  resources  and  close  file 
		H5Dclose(dataset_id);
		H5Sclose(dataspace_id);
		H5Sclose(file_dataspace_id);
		H5Fclose(file_id);
		free(dims);
	}

	// Extract information from HDF5 file
	void get_data(std::string data_path, std::string dataset_name, std::vector<std::string> &data_out){

		//  Declare variables 
		const char *fname = data_path.c_str();	
		hid_t dset, fileType, space, memType, fileId;
		herr_t status;
		hsize_t dims[2];
		char **slides;
		bool result = true;
		int numSlides;

		fileId = H5Fopen (fname, H5F_ACC_RDONLY, H5P_DEFAULT);
		dset = H5Dopen(fileId, dataset_name.c_str(), H5P_DEFAULT);
		fileType = H5Dget_type(dset);
		space = H5Dget_space(dset);
		H5Sget_simple_extent_dims(space, dims, NULL);

		slides = (char**)malloc(dims[0] * sizeof(char*));
		if( slides == NULL ) {
			result = false;
		}

		if( result ) {
			memType = H5Tcopy(H5T_C_S1);
			H5Tset_size(memType, H5T_VARIABLE);
			status = H5Dread(dset, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, slides);
			if( status < 0 ) {
				result = false;
			} else {
				numSlides = dims[0];
				H5Dclose(dset);
				H5Tclose(fileType);
			}
		}
		for (int i=0; i<dims[0]; i++){
			std::string slide(slides[i]);
			data_out.push_back(slide);
		}
	}
}