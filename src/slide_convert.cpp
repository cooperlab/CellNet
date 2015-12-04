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
#include <fstream>
#include <openslide/openslide.h>
#include <opencv2/opencv.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <sys/utsname.h>
#include <sys/time.h>
#include <ctime>

#include "utils.h"
#include "base_config.h"
#include "slide-conv-cmd.h"


using namespace std;


#define SAMPLE_WIDTH  	50
#define SAMPLE_HEIGHT	50


struct Cent {
	int		x;
	int		y;
};







int ParseCentroids(vector<Cent>& centroids, char *filename)
{
	int 		result = 0;
	Cent		centroid;
	string		line;
	size_t		pos;

	ifstream inFile(filename, ios::in);
	
	if( !inFile.is_open() ) {
		cerr << "Unable to open " << filename << endl;
		result = -10;
	}

	if( result == 0 ) {
		while( inFile ) {
			getline(inFile, line);
			
			if( line.length() > 0 ) {
				pos = line.find_first_of(",");
			
				centroid.x = (int)stof(line.substr(0, pos));
				centroid.y = (int)stof(line.substr(pos + 1));	

				centroids.push_back(centroid);
			}
		}
	}
	
	if( inFile.is_open() ) {
		inFile.close();
	}
	return result;
}






int CropCells(vector<Cent>& centroids, uint8_t *images, char *filename, float reqPower)
{
	int		result = 0;
	openslide_t	*img = NULL;
	int64_t		offset = 0, chunkSize = SAMPLE_WIDTH * SAMPLE_HEIGHT * 3;

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
		float		objPower, scaleFactor;
		int			sampleSize; 
		uint8_t		alpha;

		objPower = stof(openslide_get_property_value(img, OPENSLIDE_PROPERTY_NAME_OBJECTIVE_POWER));
		// Scaling up not permitted.
		//
		if( objPower < reqPower ) {
			reqPower = objPower;
		}	
		scaleFactor = objPower / reqPower;
		sampleSize = SAMPLE_WIDTH * scaleFactor;

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
				for(int64_t i = 0; i < 2500; i++) {
					alpha = (buffer[i] >> 24);

					if( alpha == 255 ) {
						buffer[i] &= 0xFFFFFF;
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
				// Merge channels BGR back to image
				merge(channels, outImg);
				
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





int SaveProvenance(hid_t fileId, string commandLine, float reqPower)
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

		snprintf(curTime, 100, "%2d-%2d-%4d, %2d:%2d",
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
	return result;
}


// HDF5 has a default cache size of 1M. 139 images
// at 50 by 50 will fit in the cache.
//
#define CHUNK_SIZE   139




int SaveImageDataset(hid_t fileId, uint8_t *images, int numImages)
{
	int		result = 0;
	hsize_t	dims[3] = {(hsize_t)numImages, SAMPLE_HEIGHT, SAMPLE_WIDTH * 3L}, 
			chunkDims[3] = {CHUNK_SIZE, SAMPLE_HEIGHT, SAMPLE_WIDTH * 3L}, 
			start[3] = {0, 0, 0}, size[3] = {0, SAMPLE_HEIGHT, SAMPLE_WIDTH * 3L};
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
		if( numImages < CHUNK_SIZE ) {
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




int SaveData(vector<Cent>& centroids, uint8_t *images, string filename, string commandLine,
			 float reqPower)
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
		result = SaveImageDataset(fileId, images, centroids.size());
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
		result = SaveProvenance(fileId, commandLine, reqPower);
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


	vector<Cent>	centroids;
	uint8_t	*imagesBuffer = NULL;

	cout << "Parsing cell list..." << endl;
	startTime = utils::get_time();
	result = ParseCentroids(centroids, args.centroids_arg);
	cout << "ParseCentroids took: " << utils::get_time() - startTime << endl;

	if( result == 0 ) {
		imagesBuffer = (uint8_t*)malloc(centroids.size() * SAMPLE_WIDTH * SAMPLE_HEIGHT * 3);
		if( imagesBuffer == NULL ) {
			cerr << "Unable to allocate buffer for images" << endl;
			result = -2;
		}
	}

	if( result == 0 ) {
		cout << "Cropping cells..." << endl;
		startTime = utils::get_time();
		// TODO - add magnification  args.magnifiction_arg
		result = CropCells(centroids, imagesBuffer, args.image_arg, args.magnification_arg);
		cout << "CropCells took: " << utils::get_time() - startTime << endl;
	}


	if( result == 0 ) {
		string	imageFilename = args.image_arg, outFilename;
		size_t	pos = imageFilename.find_last_of("/");
		
		if( pos == string::npos ) {
			outFilename = args.image_arg;
		} else {
			outFilename = imageFilename.substr(pos + 1);
		}
		outFilename += ".h5";

		string cmdline;
		for(int i = 0; i < argc; i++) {
			cmdline += argv[i];
			cmdline += " ";
		}
		cout << "Writing HDF5 file..." << endl;
		startTime = utils::get_time();
		result = SaveData(centroids, imagesBuffer, outFilename, cmdline, args.magnification_arg);
		cout << "SaveData took: " << utils::get_time() - startTime << endl;
	}

	if( imagesBuffer ) {
		free(imagesBuffer);
	}

	return result;
}
