#include "read_node.h"
#include "edge.h"
#include <iostream>
#include <sstream>
#include "utils.h"
#define SHIFT 25

ReadNode::ReadNode(std::string id, std::vector<std::string> image_paths, std::vector<std::vector<std::tuple<double, double>>> cells_coordinates_set, std::vector<double> labels, int mode): Node(id, mode), _image_paths(image_paths), _cells_coordinates_set(cells_coordinates_set), i_ptr(0){
	 _labels = labels;
	 std::cout << "total labels: " << std::to_string(_labels.size()) << std::endl;
	 runtime_total_first = utils::get_time();
}

void *ReadNode::run(){
	
	increment_threads();

	std::vector<cv::Mat> extracted_images;
	cv::Mat entire_image;

	// Execute
	int i = get_input();
	while(i >= 0){

		increment_counter();
		entire_image = open_image(_image_paths.at(i));
		extracted_images = crop_cells(entire_image, _cells_coordinates_set[i]);
		copy_to_buffer(extracted_images, _labels);

		// Execute
		i = get_input();
	}

	/****************** Debug ******************/
	//show_entire_image(entire_image);

	// Release memory
	extracted_images.clear();
	entire_image.release();
	
	//show_cropped_cells();
	if( check_finished() == true){

		std::cout << "******************" << std::endl << "ReadNode complete" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}
	return NULL;
}	

int ReadNode::get_input(){

	boost::mutex::scoped_lock lk(_mutex);
	if(i_ptr < _image_paths.size()){
		return i_ptr++;
	}
	else{
		return -1;
	}
}

// This method gets the 20 X magnification layer 
int ReadNode::get_layer(openslide_t *oslide){

	/*get # of levels*/
	int levels = openslide_get_level_count(oslide);
	
	/*loop through levels, get dimensions/downsample factor of each*/
	std::vector<double> magnification;
	int objective;
	std::stringstream str_objective;

	//std::cout << "num_levels: " << std::to_string(levels) << std::endl;
	for(int i = 0; i < levels; i++) {

		/*get level info*/
		str_objective << openslide_get_property_value(oslide, OPENSLIDE_PROPERTY_NAME_OBJECTIVE_POWER);
		str_objective >> objective; 
		magnification.push_back(objective / (double) openslide_get_level_downsample(oslide, (int32_t)i));
		//std::cout << "Magnification: " << magnification[i] << std::endl;
	}

	// Return the closest magnification to 20.0
	for(std::vector<double>::size_type i = 0; i < magnification.size(); i++){
		
		if(magnification[i] >= 20.0){
			std::cout << "Picked mag: " << magnification[i] << std::endl;
			return i;
		}
	}

	// Return the second highest value
	std::cout << "Magnification: " << magnification[magnification.size()-1] << std::endl;
	return magnification.size()-1;
}

// This method opens an image using openslide and removes the alpha channel
cv::Mat ReadNode::open_image(std::string image_path){

	std::cout << image_path << std::endl;
	cv::Mat entire_image;

	openslide_t *oslide = openslide_open(image_path.c_str());
	if(oslide != NULL){

		int layer_i = get_layer(oslide);
		std::cout << "layer: " << std::to_string(layer_i) << std::endl;
		
		// Declare variables
		int64_t w, h;
		openslide_get_level_dimensions(oslide, layer_i, &w, &h);
		uint32_t *buf = g_new(uint32_t, w * h);
		uint32_t *out = g_new(uint32_t, w * h);

		// Read region
		openslide_read_region(oslide, buf, 0, 0, layer_i, w, h);

		// Convert to RGBX
		for (int64_t i = 0; i < w * h; i++) {

			uint32_t pixel = buf[i];
			uint8_t a = pixel >> 24;

			if (a == 255) {
	     	   // Common case.  Compiles to a shift and a BSWAP.
				out[i] = GUINT32_TO_BE(pixel << 8);
			} else if (a == 0) {
	        	// Less common case.  Hardcode white pixel; could also
	        	// use value from openslide.background-color property
	        	// if it exists
				out[i] = GUINT32_TO_BE(0xffffff00u);
			} else {
	        	// Unusual case.
				uint8_t r = 255 * ((pixel >> 16) & 0xff) / a;
				uint8_t g = 255 * ((pixel >> 8) & 0xff) / a;
				uint8_t b = 255 * (pixel & 0xff) / a;
				out[i] = GUINT32_TO_BE(r << 24 | g << 16 | b << 8);
			}
		}
		
		// Convert to opencv
		// Get XBGR channels
		cv::Mat int_XBGR = cv::Mat(h, w, CV_8UC4, out);
		std::vector<cv::Mat> XBRG_channels;
		cv::split(int_XBGR, XBRG_channels);

		// Pop X channels
		XBRG_channels.pop_back();

		// Merge channels BGR back to image
		cv::merge(XBRG_channels, entire_image);

		// Close openslide object
		free(buf);
		free(out);
		openslide_close(oslide);
	}
	return entire_image;
}

std::vector<cv::Mat> ReadNode::crop_cells(cv::Mat entire_image, std::vector<std::tuple<double, double>> _cells_coordinates){

	std::vector<cv::Mat> extracted_images;

	if(!entire_image.empty()){

		cv::Size s = entire_image.size();
		int _entire_image_height = s.height;
		int _entire_image_width = s.width;

		for (std::vector<cv::Mat>::size_type i= 0; i != _cells_coordinates.size(); i++) {

			int x = std::get<0>(_cells_coordinates[i]);
			int y = std::get<1>(_cells_coordinates[i]);
			

			if((x-SHIFT >=0) && (y-SHIFT >=0) && (x+SHIFT < _entire_image_width) && (y+SHIFT < _entire_image_height)){
				cv::Point tl(x-SHIFT, y-SHIFT);
				cv::Point br(x+SHIFT, y+SHIFT);

				// Setup a rectangle to define region of interest
				cv::Rect cellROI(tl, br);

				// Crop image
				cv::Mat croppedImage = entire_image(cellROI);

				// Store image
				extracted_images.push_back(croppedImage);
			}
			else{
				std::cout << "Crop out of bound" << std::endl;
			}
		}
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
	return extracted_images;
}

void ReadNode::show_entire_image(cv::Mat entire_image){
	if(!entire_image.empty()){
		//std::cout << "Showing Entire Image" << std::endl;
		//cv::imshow("img", entire_image);
		//cv::waitKey(0);

		cv::imwrite("/home/nelson/CellNet/entire_image.png", entire_image);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}

void ReadNode::show_cropped_cells(std::vector<cv::Mat> extracted_images){
	if(!extracted_images.empty()){
		for(std::vector<cv::Mat>::size_type i =0; i < extracted_images.size(); i++){

			std::cout << "Showing Cropped Images" << std::endl;
			imshow("img" + std::to_string(i), extracted_images.at(i));
		}
		cv::waitKey(0);
	}
	else{
		std::cout << "Image is empty" << std::endl;
	}
}