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

	std::vector<cv::Mat> merge_all(std::vector<cv::Mat> &layers){

		std::vector<cv::Mat> merged_layers_vector;
		cv::Mat merged_layers;

		cv::merge(layers, merged_layers);
		merged_layers_vector.push_back(merged_layers);

		return merged_layers_vector;
	}

	void resize_all(std::vector<cv::Mat> &layers, cv::Size size){

		for(std::vector<cv::Mat>::size_type i = 0; i < layers.size(); i++){
			cv::resize(layers.at(i), layers.at(i), size, 0, 0, CV_INTER_CUBIC);
		}
	}	

	void fill_data(int N, int num_elem, std::vector<std::vector<std::tuple<float, float>>> &cells_coordinates_set, std::vector<std::vector<int>> &shuffled_labels, std::vector<float> &x_centroid, std::vector<float> &y_centroid, std::vector<int> &labels, std::vector<float> &slide_idx){
	
		// Fill train dataset
		srand (time(NULL));
		int k = 0;
		for(int i=0; i < N; i++){

			// Get random sample
			k = rand() % num_elem;
			
			// Adjust label value to 0 or 1
			if(labels[k] == -1){
				labels[k] = 0;
			}

			// Append 
			cells_coordinates_set[slide_idx[k]].push_back(std::make_tuple(x_centroid[k], y_centroid[k]));
			shuffled_labels[slide_idx[k]].push_back(labels[k]);

			// Erase selected element
			x_centroid.erase(x_centroid.begin() + k);
			y_centroid.erase(y_centroid.begin() + k);
			slide_idx.erase(slide_idx.begin() + k);
			labels.erase(labels.begin() + k);
			num_elem--;
		}
	}

	bool has_prefix(const std::string& s, const std::string& prefix)
	{
	    return (s.size() >= prefix.size()) && equal(prefix.begin(), prefix.end(), s.begin());    
	}

	std::string get_image_name(std::string name, std::string image_path){

		DIR *dir = opendir(image_path.c_str());
	    dirent *entry;
	    std::string image_name;

	    while(entry = readdir(dir))
	    {
	        if(has_prefix(entry->d_name, name))
	        {

	        	std::string image(entry->d_name);
	     		image_name = image;
	        }
	    }

	    return image_name;
	}

	std::vector<std::string> get_images_path(std::string path){
		std::vector<std::string> images_path;
        DIR *pDIR;
        struct dirent *entry;

		std::cout << "Looking for images in " << path << std::endl;

        if( pDIR=opendir(path.c_str()) ){
                while(entry = readdir(pDIR)){
                        if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
                        	images_path.push_back(entry->d_name);
                }
                closedir(pDIR);
        }
        return images_path;
	}

	void remove_slides(std::vector<std::string> &file_paths, std::vector< std::vector<std::tuple<float, float>>> &cells_coordinates_set, std::vector< std::vector<int>> &labels, std::vector<int> slides){
		
		std::vector<std::string> new_file_paths;
		std::vector<std::vector<std::tuple<float, float>>> new_cells_coordinates_set;
		std::vector<std::vector<int>> new_labels;

		for(int i=0; i < slides.size(); i++){
			
			int k = slides[i];

			// Keep elements
			new_file_paths.push_back(file_paths[k]);
			new_cells_coordinates_set.push_back(cells_coordinates_set[k]);
			new_labels.push_back(labels[k]);
		}

		// Update vectors
		file_paths = new_file_paths;
		cells_coordinates_set = new_cells_coordinates_set; 
		labels = new_labels;
	}

	void remove_slides(std::vector<std::string> &file_paths, std::vector<int> slides){
		
		std::vector<std::string> new_file_paths;
		for(int i=0; i < slides.size(); i++){
			
			int k = slides[i];

			// Keep elements
			new_file_paths.push_back(file_paths[k]);
		}

		// Update vectors
		file_paths = new_file_paths;
	}
}
