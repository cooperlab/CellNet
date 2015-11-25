//
//	Copyright (c) 2015, Emory University
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
#include <iostream>
#include <deque>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <ctime>

#include "utils.h"
#include "read_hdf5_node.h"
#include "base_config.h"






//
// Assume 3 bytes per pixel, we can change this later 
//	to support alpha channel or greyscale later
//
#define BYTES_PER_PIXEL		3


// Limit memory mallocs to 1GB, that's approximately 
//	143165 images at 50 x 50 
//
#define MAX_IMAGES_PER_READ		143165





ReadHDF5Node::ReadHDF5Node(string id, vector<string> fileNames, int mode, bool labels) :
Node(id, mode),
_fileNames(fileNames),
_hasLabels(labels)
{
	runtime_total_first = utils::get_time();

}





void *ReadHDF5Node::run()
{
	vector<string>::iterator	fileIt;
	
	increment_threads();
	double 	start = utils::get_time();

	for(fileIt = _fileNames.begin(); fileIt != _fileNames.end(); fileIt++) {
		cout << "." << flush;		
		if( !ReadImages(*fileIt) ) {
			cerr << "Unable to read images from " << *fileIt << endl;
		} else {

			copy_to_buffer(_input_data, _labels);
			_input_data.clear();
			_labels.clear();
		}
	}		

	if( check_finished() == true ) {
		vector<Edge*>::iterator	edgeIt;

		cout << "******************" << endl
			 << "ReadHDF5Node" << endl 
			 << "Run time: " << to_string(utils::get_time() - start) << endl 
			 << "# of elements: " << to_string(_counter) << endl 
			 << "******************" << endl;

		for(edgeIt = _out_edges.begin(); edgeIt != _out_edges.end(); edgeIt++) {
			(*edgeIt)->set_in_node_done();
		}
	}
}





bool ReadHDF5Node::ReadLabels(hid_t fileId)
{
	bool	result = true;
	herr_t	status;

	_labels.resize(_numImages);
	status = H5LTread_dataset_int(fileId, "/labels", _labels.data());
	if( status < 0 ) {
		cerr << "Unable to read labels" << endl;
		result = false;
	}
	
	return result;
}





bool ReadHDF5Node::ReadImages(string filename)
{
	bool	result = true;
	hid_t	fileId, datasetId, dataspaceId;
	hsize_t	dims[3], blockOffset[3] = {0, 0, 0}, blockSize[3];
	herr_t	status;
	uint8_t	*ptr = NULL;


	fileId = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if( fileId < 0 ) {
		cerr << "Unable to open " << filename << endl;
		result = false;
	}

	if( result ) {
		status = H5LTget_dataset_info(fileId, "/images", dims, NULL, NULL);
		if( status < 0 ) {
			cerr << "Unable to read dataset info" << endl;
			result = false;
		} else {
			_numImages = dims[0];
			_imageHeight = dims[1];
			_imageWidth = dims[2]; 		// Actualy img width * BYTES_PER_PIXEL
		}
	}	

	if( result && _hasLabels ) {
		result = ReadLabels(fileId);
	}

	if( result ) {

		datasetId = H5Dopen(fileId, "/images", H5P_DEFAULT);
		if( datasetId < 0 ) {
			cerr << "Unable to open dataset" << endl;
			result = false;
		} else {

			dataspaceId = H5Dget_space(datasetId);
			if( dataspaceId < 0 ) {
				cerr << "Unable to get dataspace" << endl;
				result = false;
			}
		}
	}

	if( result ) {

		int	   imagesRead = 0, imagesToRead;
		tuple<uint8_t*, int>	curBuffer;

		// Spin up formatting thread
		thread	formatter(&ReadHDF5Node::FormatImages, this);

		// Image width & height are constatnt, just set them once.
		blockSize[1] = _imageHeight;
		blockSize[2] = _imageWidth;

		while( imagesRead < _numImages ) {
			imagesToRead = min(_numImages - imagesRead, MAX_IMAGES_PER_READ);

			ptr = (uint8_t*)malloc(imagesToRead * _imageHeight * _imageWidth);
			if( ptr == NULL ) {
				cerr << "Unable to allocate buffer for images" << endl;
				result = false;
				break;
			}

			get<0>(curBuffer) = ptr;
			get<1>(curBuffer) = imagesToRead;

			// Set hyperslab size for the number of images to read
			blockSize[0] = imagesToRead;
			status = H5Sselect_hyperslab(dataspaceId, H5S_SELECT_SET, blockOffset, NULL, blockSize, NULL);
			if( status < 0 ) {
				cerr << "Unable to select hyperslab" << endl;
				result = false;
				break;
			}
		
			status = H5Dread(datasetId, H5T_NATIVE_UCHAR, H5S_ALL, dataspaceId, H5P_DEFAULT, ptr);
			if( status < 0 ) {
				cerr << "Unable to read block" << endl;
				result = false;
				break;
			}
			_imagePipe.push_back(curBuffer);
			_imageSem.Increment();

			imagesRead += imagesToRead;
			ptr = NULL;				// Buffer is now on the imagePipe queue			
		} 
		
		formatter.join();
	}
 
	if( ptr != NULL ) {
		free(ptr);
	}
	if( dataspaceId >= 0 ) {
		H5Sclose(dataspaceId);
	}
	if( datasetId >= 0 ) {
		H5Dclose(datasetId);
	}
	if( fileId >= 0 ) {
		H5Fclose(fileId);
	}
	return result;
}





void ReadHDF5Node::FormatImages(void)
{
	int	imagesFormatted = 0;
	tuple<uint8_t*, int>	curBuffer;
	uint8_t		*ptr;

	cv::Mat img = cv::Mat(_imageWidth / BYTES_PER_PIXEL, _imageHeight, CV_8UC3);
	int	 bufferOffset = 0, stride = _imageWidth * _imageHeight;

	while( imagesFormatted < _numImages ) {
		_imageSem.Decrement();	// Wait for next block

		curBuffer = _imagePipe.front();
		_imagePipe.pop_front();
		ptr = get<0>(curBuffer);

		for(int i = 0; i < get<1>(curBuffer); i++) {
			memcpy(img.ptr(), &ptr[bufferOffset], stride);
			_input_data.push_back(img.clone());
			if( !_hasLabels ) {
				// Use index into dataset as an id
				_labels.push_back(imagesFormatted + i);
			}
			bufferOffset += stride;
			_counter++;
		}

		imagesFormatted += get<1>(curBuffer);

		bufferOffset = 0;
		free(ptr);
	}
}

