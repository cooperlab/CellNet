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
#include <cstring>

#include "graph_net.h"


using namespace std;



WriteImageNode::WriteImageNode(std::string id, bool split, int transferSize, int mode, char *scoreFile) : 
Node(id, mode),
_transferSize(transferSize),
_split(split),
_scoreFile(scoreFile),
_scores(NULL)
{

}




WriteImageNode::~WriteImageNode()
{
	if( _scores != NULL ) {
		free(_scores);
	}
}





void *WriteImageNode::run(){

	std::vector<cv::Mat> out; 
	double	start = utils::get_time();

	if( _scoreFile != NULL ) {
		if( !ReadScores() ) {
			cerr << "Unable to read score file " << _scoreFile << endl;
			_scoreFile = NULL;
		}
	}


	while( true ) {

		copy_chunk_from_buffer(out, _labels);

		if( out.size() >= _transferSize ) {

			SaveImages(out);

			out.clear();
			_labels.clear();

		} else if( _in_edges.at(0)->is_in_node_done() ) {

			// Check for any data leftover
			if( out.size() > 0 ) {

				SaveImages(out);

				out.clear();
				_labels.clear();
			}
			break;
		}
	}

	
	// Notify it has finished
	vector<Edge *>::iterator	it;

	for(it = _out_edges.begin(); it != _out_edges.end(); it++) {
		(*it)->set_in_node_done();
	}

	cout << "******************" << endl 
		 << "WriteImageNode complete" << endl 
		 << "Run time: " << to_string(utils::get_time() - start) << endl 
		 << "# of elements: " << to_string(_counter) << endl 
		 << "******************" << endl;

	return NULL;
}





void WriteImageNode::SaveImages(vector<cv::Mat> images)
{
	string name; 
	vector<cv::Mat>		layers;
	int	scoreIdx = 0;	
		
	for(int i = 0; i < images.size(); i++) {

		if( _split ) {
			cv::split(images[i], layers);


			for(int l = 0; l < layers.size(); l++) {

				name = "img" + to_string(_counter) + "_" + to_string(_labels[i]) 
						+ "_" + to_string(l) + ".jpg";
			
				cv::imwrite(name.c_str(), layers[l]);
			}
		} else {

			if( _scores == NULL ) {
				name = "img" + to_string(_counter) + "_" + to_string(_labels[i]) 
					 + ".jpg";
			} else {
				if( _scores[scoreIdx].idx - 1 == _counter ) {
					name = "img_" + to_string(_scores[scoreIdx++].score) + "_" + to_string(_labels[i]) 
						 + ".jpg";
				} else {
					name = "img_-1_" + to_string(_counter) + ".jpg";
				}	
			}

			if( images[i].channels() == 3 ) {
				cv::cvtColor(images[i], images[i], CV_BGR2RGB);
			}
			cv::imwrite(name.c_str(), images[i]);
		}
		increment_counter();
	}
}





bool WriteImageNode::ReadScores(void)
{
	bool result = true;
	ifstream inFile(_scoreFile, ios::in);

	_scoreCnt = 0;
	
	if( inFile.is_open() ) {
		string line, idx, score;
		Scores	*newBuff = NULL;
		int		count = 0;

		while( getline(inFile, line) ) {
			size_t	pos;
			
			newBuff = (Scores*)realloc(_scores, ++count * sizeof(Scores));
			if( newBuff == NULL ) {
				cerr << "Unable to update scores buffer" << endl;
				result = false;
				break;
			} else {
				_scores = newBuff;
 			}

			pos = line.find_first_of(",");
			idx = line.substr(0, pos);
			score = line.substr(pos + 1);

			_scores[count - 1].idx = stoi(idx);
			_scores[count - 1].score = stof(score);

		}
		inFile.close();
	} else {
		cerr << "Unable to open " << _scoreFile << endl;
		result = false;
	}

	return result;
}

