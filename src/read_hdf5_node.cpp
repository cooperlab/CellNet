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


// Limit memory mallocs to 1GB
//
#define MAX_BUFFER	(1024 * 1024 * 1024)





ReadHDF5Node::ReadHDF5Node(string id, vector<string> fileNames, int mode, int deconvChannels, bool labels) :
Node(id, mode),
_fileNames(fileNames),
_hasLabels(labels),
_deconvChannels(deconvChannels)
{

}





void *ReadHDF5Node::run()
{
	vector<string>::iterator	fileIt;
	
	increment_threads();
	_runtimeStart = utils::get_time();


	for(fileIt = _fileNames.begin(); fileIt != _fileNames.end(); fileIt++) {

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
			 << "Run time: " << to_string(utils::get_time() - _runtimeStart) << endl 
			 << "# of elements: " << to_string(_counter) << endl 
			 << "******************" << endl;

		for(edgeIt = _out_edges.begin(); edgeIt != _out_edges.end(); edgeIt++) {
			(*edgeIt)->set_in_node_done();
		}
	}
	return NULL;
}





bool ReadHDF5Node::ReadLabels(hid_t fileId)
{
	bool	result = true;
	herr_t	status;

	cout << "Reading labels" << endl;
	_labels.resize(_numImages);
	status = H5LTread_dataset_int(fileId, "/labels", _labels.data());
	if( status < 0 ) {
		cerr << "Unable to read labels" << endl;
		result = false;
	} else {
		vector<int>::iterator it;
		for(it = _labels.begin(); it != _labels.end(); it++) {
			if( *it == -1 ) {
				*it = 0;
			}
		}
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
			imagesToRead = min(_numImages - imagesRead, MAX_BUFFER / (_imageHeight * _imageWidth));
			
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
		
		if( result == false ) {
			// An error occured somewhere in the previous loop. Send a flag to
			// the formatting thread so it knows to stop.
			get<0>(curBuffer) = NULL;
			get<1>(curBuffer) = -1;
			_imagePipe.push_back(curBuffer);
			_imageSem.Increment();
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
		if( ptr == NULL ) {
			// An error occured, stop
			break;
		} else {
			for(int i = 0; i < get<1>(curBuffer); i++) {
				memcpy(img.ptr(), &ptr[bufferOffset], stride);

				// An assumption has been made that the deconvoluted images are
				// H&E stained images, with the first channel being Hematoxylin,
				// the second Eosin and the third the compliment (cross-product).
				//
				if( _deconvChannels > 0 && _deconvChannels < 3 ) {
					// Deconoluted image, drop last channel
					//
					vector<cv::Mat> channels;
					cv::Mat			deconv;
					
					cv::split(img, channels);

					// Drop compliment
					channels.pop_back();

					if( _deconvChannels == 1 ) {
						// Drop the Eosin channel 
						channels.pop_back();
					}
					cv::merge(channels, deconv);
					_input_data.push_back(deconv.clone());

				} else {
					_input_data.push_back(img.clone());
				}

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
}

