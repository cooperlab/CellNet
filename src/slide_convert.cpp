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
#include <ctime>


#include "base_config.h"



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






int CropCells(vector<Cent>& centroids, uint8_t *images, char *filename)
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
		int64_t		width, height, maxWidth, maxHeight;
		int			objectivePower;

		cout << "Extracting " << centroids.size() << " objects" << endl;


		openslide_get_level0_dimensions(img, &maxWidth, &maxHeight);
		objectivePower = stoi(openslide_get_property_value(img, OPENSLIDE_PROPERTY_NAME_OBJECTIVE_POWER));

		cout << "Max image dimensions: " << maxWidth << " x " << maxHeight << endl;
		cout << "Objective power: " << objectivePower << endl;
		
		int levels = openslide_get_level_count(img);
		double	downSample, mag;

		cout << "Has " << levels << " levles" << endl;
		for(int i = 0; i < levels; i++) {
			openslide_get_level_dimensions(img, i, &width, &height);
			downSample = openslide_get_level_downsample(img, i);
			mag = (double)objectivePower / downSample;

			cout << "level " << i << " dimensions: " << width << " x " << height << endl;
			cout << "Level down sample: " << downSample << endl;
			cout << "Level magnification: " << mag << endl;
		}

		uint32_t	*buffer = (uint32_t*)malloc(SAMPLE_WIDTH * SAMPLE_HEIGHT * sizeof(uint32_t));
		uint8_t		alpha;

		if( buffer != NULL ) {

			for(int cell = 0; cell < centroids.size(); cell++) {

				openslide_read_region(img, 
									  buffer, 
									  centroids[cell].x - (SAMPLE_WIDTH / 2), 
									  centroids[cell].y - (SAMPLE_HEIGHT / 2), 
									  0, 
									  SAMPLE_WIDTH, 
									  SAMPLE_HEIGHT);

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

				cv::Mat		outImg, abgrImg = cv::Mat(SAMPLE_WIDTH, SAMPLE_HEIGHT, CV_8UC4, buffer);
				vector<cv::Mat> channels;
				cv::split(abgrImg, channels);

				// Pop X channels
				channels.pop_back();

				// Merge channels BGR back to image
				merge(channels, outImg);
				if( outImg.isContinuous() ) {
					uint8_t	*data = outImg.ptr();
					
					memcpy(&images[offset], data, chunkSize);
					offset += chunkSize;
				}
						
#if 0
				string outName = "test" + to_string(cell) + ".jpg";
				imwrite(outName.c_str(), outImg);
#endif
			}
			free(buffer);
		}
	}

	if( img != NULL ) {
		openslide_close(img);
	}

	return result;
}





int SaveProvenance(hid_t fileId, string commandLine)
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
		int	ver[2] = {CELL_NET_VERSION_MAJOR, CELL_NET_VERSION_MINOR};

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
	return result;
}







int SaveData(vector<Cent>& centroids, uint8_t *images, string filename, string commandLine)
{
	int		result = 0;
	hid_t	fileId;
	hsize_t	dims[3];
	herr_t	status;

	fileId = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if( fileId < 0 ) {
		cerr << "Unable to create HDF5 file " << filename << endl;
		result = -30;
	}

	if( result == 0 ) {
		dims[0] = centroids.size();
		dims[1] = SAMPLE_HEIGHT;
		dims[2] = SAMPLE_WIDTH * 3;

		status = H5LTmake_dataset(fileId, "/images", 3, dims, H5T_NATIVE_UCHAR, images);
		if( status < 0 ) {
			cerr << "Unable to create images dataset" << endl;
			result = -31;
		}
	}

	if( result == 0 ) {
		Cent	*centroidData = centroids.data();

		dims[1] = 2;
		status = H5LTmake_dataset(fileId, "/centroids", 2, dims, H5T_NATIVE_INT, centroidData);
		if( status < 0 ) {
			cerr << "Unable to create centroid dataset" << endl;
			result = -32;
		}
	}

	if( result == 0 ) {
		result = SaveProvenance(fileId, commandLine);
	}

	if( fileId >= 0 ) {
		H5Fclose(fileId);
	}	
	
	return result;
}






int main(int argc, char *argv[])
{
	int result = 0;

	if( argc != 3 ) {

		cerr << "Usage: " << argv[0] << " <image file> <centroid list>" << endl;
		exit(-1);
	}
	cout << "Using: " << endl;
	cout << "  OpenSlide    " << openslide_get_version() << endl;
	cout << "  OpenCV       " << CV_VERSION << endl;


	vector<Cent>	centroids;
	uint8_t	*imagesBuffer = NULL;

	result = ParseCentroids(centroids, argv[2]);

	if( result == 0 ) {
		imagesBuffer = (uint8_t*)malloc(centroids.size() * SAMPLE_WIDTH * SAMPLE_HEIGHT * 3);
		if( imagesBuffer == NULL ) {
			cerr << "Unable to allocate buffer for images" << endl;
			result = -2;
		}
	}

	if( result == 0 ) {
		result = CropCells(centroids, imagesBuffer, argv[1]);
	}

	if( result == 0 ) {
		string	imageFilename = argv[1], outFilename;
		size_t	pos = imageFilename.find_last_of("/");
		
		if( pos == string::npos ) {
			outFilename = argv[1];
		} else {
			outFilename = imageFilename.substr(pos + 1);
		}
		outFilename += ".h5";

		string cmdline;
		for(int i = 0; i < argc; i++) {
			cmdline += argv[i];
			cmdline += " ";
		}

		result = SaveData(centroids, imagesBuffer, outFilename, cmdline);
	}

	if( imagesBuffer ) {
		free(imagesBuffer);
	}

	return result;
}
