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
#if !defined(_READ_HDF5_NODE_H_)
#define _READ_HDF5_NODE_H_

#include <thread>
#include <condition_variable>
#include <vector>
#include <deque>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <hdf5.h>
#include <hdf5_hl.h>

#include "node.h"



using namespace std;


class Semaphore {
public:
	Semaphore(void) : _count(0) {}
	void Increment(void) 
	{
		unique_lock<std::mutex> lck(_countMtx);
		_count++;
		_cv.notify_one();
	}
	void Decrement(void)
	{
		unique_lock<std::mutex> lck(_countMtx);
		while( _count == 0 )
			_cv.wait(lck);
	}
private:

	mutex					_countMtx;
	condition_variable		_cv;
	int						_count;

};






class ReadHDF5Node : public Node
{
public:

		ReadHDF5Node(string id, vector<string> fileNames, int mode, 
					 bool deconvImg, bool labels = false);
		void 	*run(void);
		void	init(void);

private:

		vector<string> 		_fileNames;
		bool				_hasLabels;
		bool				_deconvImg;
		int					_numImages;
		int					_imageWidth;
		int					_imageHeight;
		Semaphore			_imageSem;
		deque< tuple<uint8_t*, int> >  _imagePipe;
		vector<cv::Mat> 	_input_data;
		vector<int>			_labels;


		bool	ReadImages(string filename);
		bool	ReadLabels(hid_t fileId);
		void	FormatImages(void);
};



		
#endif
