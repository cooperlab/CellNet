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
#include <fstream>
#include <openslide/openslide.h>
#include <opencv2/opencv.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <sys/utsname.h>
#include <sys/time.h>
#include <ctime>
#include <glob.h>
#include <endian.h>

#include "utils.h"
#include "base_config.h"
#include "data-conv-cmd.h"


using namespace std;



#define BYTES_PER_PIXEL		3



struct Cent {
	int		x;
	int		y;
};






int GetSlideCells(vector<Cent> centroids, uint8_t *images, int64_t&	offset, float reqPower, 
				  char *filename, int imgSize)
{
	int		result = 0;
	openslide_t	*img = NULL;
	int64_t		chunkSize = imgSize * imgSize * 3;

	if( images == NULL ) {
		result = -10;
	}
	
	if( result == 0 ) {
		img = openslide_open(filename);
		if( img == NULL ) {
			cerr << "Unable to open " << filename << endl;
			result = -20;
		}
	}

	if( result == 0 ) {
		uint8_t		alpha;
		float		objPower, scaleFactor;
		int			sampleSize; 

		objPower = stof(openslide_get_property_value(img, OPENSLIDE_PROPERTY_NAME_OBJECTIVE_POWER));	

		scaleFactor = objPower / reqPower;
		sampleSize = imgSize * scaleFactor;

		cout << "Slide objective power: " << objPower
			 << ", Requested power: " << reqPower
			 << ", Scale factor: " << scaleFactor 
			 << ", sample size: " << sampleSize << endl;

		uint32_t	*buffer = (uint32_t*)malloc(sampleSize * sampleSize * sizeof(uint32_t));

		if( buffer != NULL ) {

			for(int cell = 0; cell < centroids.size(); cell++) {

				openslide_read_region(img, 
									  buffer, 
									  centroids[cell].x - (sampleSize / 2), 
									  centroids[cell].y - (sampleSize / 2), 
									  0, 
									  sampleSize, 
									  sampleSize);

				// Some scanners format pixels in paculiar ways. Need
				// to handle them here. See Openslide docs on premultiplied RGB.
				// Note! The OpenSlide example code has a byte swap, this seems
				// to not be needed. (Images written to jpeg are correct without it)
				//
				for(int64_t i = 0; i < sampleSize * sampleSize; i++) {
					alpha = (buffer[i] >> 24);

					if( alpha == 255 ) {
						buffer[i] = be32toh(buffer[i] << 8);
					} else if( alpha == 0 ) {
						buffer[i] = 0xFFFFFF;
					} else {
						uint8_t r = 255 * ((buffer[i] >> 16) & 0xff) / alpha;
						uint8_t g = 255 * ((buffer[i] >> 8) & 0xff) / alpha;
						uint8_t b = 255 * (buffer[i] & 0xff) / alpha;

						buffer[i] = r << 16 | g << 8 | b;
					}
				}

				cv::Mat		outImg, abgrImg = cv::Mat(sampleSize, sampleSize, CV_8UC4, buffer);
				vector<cv::Mat> channels;
				cv::split(abgrImg, channels);

				// Pop X channels
				channels.pop_back();

				// Merge RGB channels back to image Mat
				cv::merge(channels, outImg);
				
				if( scaleFactor == 1.0f ) {
					if( outImg.isContinuous() ) {
						uint8_t	*data = outImg.ptr();
					
						memcpy(&images[offset], data, chunkSize);
						offset += chunkSize;
					} 
				} else {
					cv::Mat		reSized;

					cv::resize(outImg, reSized, cv::Size(), 1.0f / scaleFactor, 
								1.0f / scaleFactor, cv::INTER_CUBIC);

					uint8_t	*data = reSized.ptr();
					
					memcpy(&images[offset], data, chunkSize);
					offset += chunkSize;
				}
			}
			free(buffer);
		}
	}

	if( img != NULL ) {
		openslide_close(img);
	}

	return result;
}





int CropCells(uint8_t *images, vector<Cent>& centroids, vector<int>& labels, 
			  vector<int>& slideIdx, char **slideNames, int numSlides, string imageDir,
			  float reqPower, int imgSize)
{
	int		result = 0;
	glob_t	globBuff;
	string	path;
	vector< vector<Cent> > slideCentroids;
	vector< vector<int> > slideLabels;
	int64_t	offset = 0;

	// Collect centroids belonging to the same slide. This way
	// we can go slide by slide to extract the images. We will 
	// reorder the labels and centroids afterwards.
	//
	slideCentroids.resize(numSlides);
	slideLabels.resize(numSlides);
	for(int i = 0; i < centroids.size(); i++) {
		slideCentroids[slideIdx[i]].push_back(centroids[i]);
		slideLabels[slideIdx[i]].push_back(labels[i]);
	}


	for(int i = 0; i < numSlides; i++) {
		path = imageDir + slideNames[i] + "*";
		glob(path.c_str(), GLOB_TILDE, NULL, &globBuff);
		if( globBuff.gl_pathc == 0 ) {
			cerr << "Unable to find image for " << slideNames[i] << endl;
		} else {
			cout << "Extracting " << slideCentroids[i].size() << " cell images from " << globBuff.gl_pathv[0] << endl;
			result = GetSlideCells(slideCentroids[i], images, offset, reqPower, globBuff.gl_pathv[0], imgSize);
		}
	}

	// Reorder slideIdx & labels to match the order of the images in the buffer.
	//
	offset = 0;
	for(int slide = 0; slide < numSlides; slide++) {
		for(int obj = 0; obj < slideCentroids[slide].size(); obj++) {

			centroids[offset] = slideCentroids[slide][obj];
			slideIdx[offset] = slide;
			labels[offset] = slideLabels[slide][obj];

			offset++;
		}
	}
	return result;
}





int SaveProvenance(hid_t fileId, string commandLine, float reqPower, int imgSize)
{
	int		result = 0;
	hsize_t	dims[2];
	herr_t	status;
	
	struct utsname	hostInfo;

	if( uname(&hostInfo) ) {
		cerr << "Unable to get host info" << endl;
		result = -40;
 	} else {
		
		string sysInfo = hostInfo.nodename;
		sysInfo += ", ";
		sysInfo += hostInfo.sysname;
		sysInfo += " (";
		sysInfo += hostInfo.release;
		sysInfo += " ";
		sysInfo += hostInfo.machine;
		sysInfo += ")";

		status = H5LTset_attribute_string(fileId, "/", "host_info", sysInfo.c_str());
		if( status < 0 ) {
			cerr << "Unable to save host info" << endl;
			result = -41;
		}
	}

	if( result == 0 ) {
		int	ver[2] = {TISSUE_NET_VERSION_MAJOR, TISSUE_NET_VERSION_MINOR};

		status = H5LTset_attribute_int(fileId, "/", "version", ver, 2);
		if( status < 0 ) {
			cerr << "Unabel to save version info" << endl;
			result = -42;
		}
	}

	if( result == 0 ) {
		time_t t = time(0);
		struct tm *now = localtime(&t);
		char curTime[100];

		snprintf(curTime, 100, "%02d-%02d-%4d, %2d:%2d",
	    		now->tm_mon + 1, now->tm_mday, now->tm_year + 1900,
	    		now->tm_hour, now->tm_min);

		status = H5LTset_attribute_string(fileId, "/", "creation date", curTime);
		if( status < 0 ) {
			result = false;
		}
	}

	if( result == 0 ) {
		status = H5LTset_attribute_string(fileId, "/", "command_line", commandLine.c_str());
		if( status < 0 ) {
			cerr << "Unable to save command line info" << endl;
			result = -41;
		}
	}

	if( result == 0 ) {
		string mag = to_string(reqPower) + "x";
		
		status = H5LTset_attribute_string(fileId, "/", "magnification", mag.c_str());
		if( status < 0 ) {
			cerr << "Unable to save magnification" << endl;
			result = -42;
		}
	}

	if( result == 0 ) {
		string 	size = to_string(imgSize);
		
		status = H5LTset_attribute_string(fileId, "/", "image_dims", size.c_str());
		if( status < 0 ) {
			cerr << "Unable to save image size" << endl;
			result = -43;
		}
	}
	return result;
}


// HDF5 has a default cache size of 1M. 
//
#define CACHE_SIZE   (1024 * 1024)




int SaveImageDataset(hid_t fileId, uint8_t *images, int numImages, int imgSize)
{
	int		result = 0, chunkSize = CACHE_SIZE / (imgSize * imgSize * 3);
	hsize_t	dims[3] = {(hsize_t)numImages, (hsize_t)imgSize, (hsize_t)imgSize * 3L}, 
			chunkDims[3] = {(hsize_t)chunkSize, (hsize_t)imgSize, (hsize_t)imgSize * 3L}, 
			start[3] = {0, 0, 0}, size[3] = {0, (hsize_t)imgSize, (hsize_t)imgSize * 3L};
	hid_t	datasetId, fileSpaceId, pList, memSpaceId;
	herr_t	status;

	// Need to save image data using chunks so we can compress it.
	//
	fileSpaceId = H5Screate_simple(3, dims, NULL);		// Max size same as current
	if( fileSpaceId < 0 ) {
		cerr << "Unable to create fileSpace" << endl; 
		result = -50;
	}

	if( result == 0 ) {
		if( numImages < chunkSize ) {
			chunkDims[0] = numImages;
		}

		pList = H5Pcreate(H5P_DATASET_CREATE);
		H5Pset_layout(pList, H5D_CHUNKED);
		H5Pset_chunk(pList, 3, chunkDims);

		datasetId = H5Dcreate(fileId, "/images", H5T_NATIVE_UCHAR, fileSpaceId, H5P_DEFAULT,
								pList, H5P_DEFAULT);
		if( datasetId < 0 ) {
			cerr << "Unable to create image dataset" << endl;
			result = -51;
		}
		H5Pclose(pList);
	}

	if( result == 0 ) {
		size[0] = numImages;
		status = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, start, NULL, size, NULL);
		if( status < 0 ) {
			cerr << "Unable to select hyperslab" << endl;
			result = -52;
		} else { 

			status = H5Dwrite(datasetId, H5T_NATIVE_UCHAR, H5S_ALL, fileSpaceId, H5P_DEFAULT, images);
			if( status < 0 ) {
				cerr << "Unable to write image data" << endl;
				result = -53;
			}	
		}
		H5Dclose(datasetId);
		H5Sclose(fileSpaceId);
	}
	return result;
}





int SaveSlidenames(hid_t fileId, char **slideNames, int numSlides)
{
	int		result = 0;
	hsize_t	dims[2], size = numSlides;
	herr_t	status;
	hid_t	dset, dataType, slideSpace, slideMemType;

	slideSpace = H5Screate_simple(1, &size, NULL);
	if( slideSpace < 0 ) {
		cerr << "Unable to create slide name dataspace" << endl;
		result = -60;
	}

	if( result == 0 ) {
		dataType = H5Tcopy(H5T_C_S1);
		H5Tset_size(dataType, H5T_VARIABLE);
		dset = H5Dcreate(fileId, "/slides", dataType, slideSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if( dset < 0 ) {
			cerr << "Unable to create slides dataset" << endl;
			result = -61;
		} else {

			H5Dwrite(dset, dataType, slideSpace, H5S_ALL, H5P_DEFAULT, slideNames);
			H5Dclose(dset);
			H5Sclose(slideSpace);
			H5Tclose(dataType);
		}
 	}
	return result;
}




int SaveData(vector<Cent>& centroids, vector<int> labels, vector<int> slideIdx, 
			char **slideNames, int numSlides, uint8_t *images, string filename, 
			string commandLine, float reqPower, int imgSize)
{
	int		result = 0;
	hid_t	fileId;
	hsize_t	dims[2];
	herr_t	status;

	fileId = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if( fileId < 0 ) {
		cerr << "Unable to create HDF5 file " << filename << endl;
		result = -30;
	}

	if( result == 0 ) {
		result = SaveImageDataset(fileId, images, centroids.size(), imgSize);
	}

	if( result == 0 ) {
		Cent	*centroidData = centroids.data();

		dims[0] = centroids.size();
		dims[1] = 2;
		status = H5LTmake_dataset(fileId, "/centroids", 2, dims, H5T_NATIVE_INT, centroidData);
		if( status < 0 ) {
			cerr << "Unable to create centroid dataset" << endl;
			result = -32;
		}
	}

	if( result == 0 ) {

		dims[0] = labels.size();
		dims[1] = 1;
		status = H5LTmake_dataset(fileId, "/labels", 2, dims, H5T_NATIVE_INT, labels.data());
		if( status < 0 ) {
			cerr << "Unable to create labels dataset" << endl;
			result = -33;
		}
	}

	if( result == 0 ) {

		dims[0] = slideIdx.size();
		dims[1] = 1;
		status = H5LTmake_dataset(fileId, "/slideIdx", 2, dims, H5T_NATIVE_INT, slideIdx.data());
		if( status < 0 ) {
			cerr << "Unable to create slideIdx dataset" << endl;
			result = -34;
		}
	}

	if( result == 0 ) {
		result = SaveSlidenames(fileId, slideNames, numSlides);
	}

	if( result == 0 ) {
		result = SaveProvenance(fileId, commandLine, reqPower, imgSize);
	}

	if( fileId >= 0 ) {
		H5Fclose(fileId);
	}	

	return result;
}





int	ReadCentroidData(hid_t fileId, vector<Cent>& centroids)
{
	int 		result = 0;
	hsize_t		dims[2];
	herr_t		status;
	float		*centX = NULL, *centY = NULL;

	
	status = H5LTget_dataset_info(fileId, "/x_centroid", dims, NULL, NULL);
	if( status < 0 ) {
		cerr << "Unable to get dataset info" << endl;
		result = -20;
	} else {

		centX = (float*)malloc(dims[0] * sizeof(float));
		centY = (float*)malloc(dims[0] * sizeof(float));

		if( centX == NULL || centY == NULL ) {
			cerr << "Unable to allocate centroid buffer" << endl;
			result = -21;
		}
	}
	
	if( result == 0 ) {

		status = H5LTread_dataset_float(fileId, "/x_centroid", centX);
		if( status < 0 ) {
			cerr << "Unable ro read X centroids" << endl;
			result = -22;
		}
	}

	if( result == 0 ) {

		status = H5LTread_dataset_float(fileId, "/y_centroid", centY);
		if( status < 0 ) {
			cerr << "Unable ro read Y centroids" << endl;
			result = -23;
		}
	}

	if( result == 0 ) {
		Cent	sample;
		
		for(int i = 0; i < dims[0]; i++) {
			sample.x = centX[i];
			sample.y = centY[i];

			centroids.push_back(sample);
		}
	}
	
	if( centX ) {
		free(centX);
	}
	if( centY ) {
		free(centY);
	}
	return result;
}





int ReadSlideNames(hid_t fileId, char **&slideNames, int& numSlides)
{	
	int			result = 0;
	hid_t		dset, fileType, slideSpace, slideMemType;
	hsize_t		dims[2];
	herr_t		status;

	dset = H5Dopen(fileId, "/slides", H5P_DEFAULT);
	if( dset < 0 ) {
		cerr << "Unable to open slides dataset" << endl;
		result = -30;
	}

	if( result == 0 ) {
		fileType = H5Dget_type(dset);
		slideSpace = H5Dget_space(dset);
		H5Sget_simple_extent_dims(slideSpace, dims, NULL);

		slideNames = (char**)malloc(dims[0] * sizeof(char*));
		if( slideNames == NULL ) {
			cerr << "Unable to allocate slide name buffer" << endl;
			result = -31;
		} else {
			slideMemType = H5Tcopy(H5T_C_S1);
			H5Tset_size(slideMemType, H5T_VARIABLE);
			status = H5Dread(dset, slideMemType, H5S_ALL, H5S_ALL, H5P_DEFAULT, slideNames);
			if( status < 0 ) {
				cerr << "Unable to read slide names" << endl;
				result = -32;
			} else { 
				numSlides = dims[0];
				H5Dclose(dset);
				H5Tclose(fileType);
			}
		} 
	}
	return result;
}





int ReadDatafile(string filename, vector<Cent>& centroids, vector<int>& labels, 
			 vector<int>&slideIdx, char **&slideNames, int& numSlides)
{
	int		result = 0;
	hid_t	fileId;
	hsize_t	dims[2];
	herr_t	status;
	float	*centX = NULL, *centY = NULL;

	fileId = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if( fileId < 0 ) {
		cerr << "Unable to open " << filename << endl;
		result = -10;
	}

	if( result == 0 ) {
		result = ReadCentroidData(fileId, centroids);
	}

	if( result == 0 ) {
		labels.resize(centroids.size());
		status = H5LTread_dataset_int(fileId, "/labels", labels.data());
		if( status < 0 ) {
			cerr << "Unable to read labels" << endl;
			result = -11;
		}
	}

	if( result == 0 ) {
		slideIdx.resize(centroids.size());
		status = H5LTread_dataset_int(fileId, "/slideIdx", slideIdx.data());
		if( status < 0 ) {
			cerr << "Unable to read slide indices" << endl;
			result = -12;
		}
	}

	if( result == 0 ) {
		result = ReadSlideNames(fileId, slideNames, numSlides);
	}

	if( fileId >= 0 ) {
		H5Fclose(fileId);
	} 

	return result;
}





int main(int argc, char *argv[])
{
	int 				result = 0;
	double				startTime;
	gengetopt_args_info	args;

	if( cmdline_parser(argc, argv, &args) != 0 ) {
		exit(-1);
	}

	cout << "Using: " << endl;
	cout << "  OpenSlide    " << openslide_get_version() << endl;
	cout << "  OpenCV       " << CV_VERSION << endl;


	vector<int>		labels, slideIdx;
	vector<Cent>	centroids;
	char			**slideNames = NULL;
	uint8_t			*imagesBuffer = NULL;
	int				numSlides, imgSize = args.size_arg;


	result = ReadDatafile(args.dataset_arg, centroids, labels, slideIdx, slideNames, numSlides);

	if( result == 0 ) {
		cout << "Read " << centroids.size() << " centroids in " << numSlides << " slides" << endl;
	}

	if( result == 0 ) {
		imagesBuffer = (uint8_t*)malloc(centroids.size() * imgSize * imgSize * BYTES_PER_PIXEL);
		if( imagesBuffer == NULL ) {
			cerr << "Unable to allocate buffer for images" << endl;
			result = -2;
		}
	}

	if( result == 0 ) {
		cout << "Cropping cells..." << endl;
		startTime = utils::get_time();
		result = CropCells(imagesBuffer, centroids, labels, slideIdx, slideNames, 
						   numSlides, args.image_dir_arg, args.maginfication_arg, imgSize);
		cout << "CropCells took: " << utils::get_time() - startTime << endl;
	}

	if( result == 0 ) {
		string cmdline;

		for(int i = 0; i < argc; i++) {
			cmdline += argv[i];
			cmdline += " ";
		}
		cout << "Writing HDF5 file..." << endl;
		startTime = utils::get_time();
		result = SaveData(centroids, labels, slideIdx, slideNames, numSlides, imagesBuffer, 
							args.output_arg, cmdline, args.maginfication_arg, imgSize);
		cout << "SaveData took: " << utils::get_time() - startTime << endl;
	}

	if( imagesBuffer ) {
		free(imagesBuffer);
	}

	return result;
}
