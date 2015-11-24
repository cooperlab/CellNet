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
#include "node.h"

using namespace std;



Node::Node(string id, int mode) : 
_id(id), 
_mode(mode), 
_in_edges(), 
_out_edges(), 
_counter(0), 
runtime_total_first(0), 
_counter_threads(0), 
ctrl(0), 
_labels()
{

}





Node::~Node(){

    _in_edges.clear();
    _out_edges.clear();
}





void Node::insert_in_edge(Edge *edge_ptr)
{
	_in_edges.push_back(edge_ptr);
}





void Node::insert_out_edge(Edge *edge_ptr)
{
	_out_edges.push_back(edge_ptr);
}





void Node::copy_to_buffer(vector<cv::Mat> out, vector<int> &labels)
{
// Repeat Node
// Copy everything to all output edges, repeating the images 
	if( _mode == Node::Repeat ) {
		for(vector<int>::size_type i=0; i < _out_edges.size(); i++) {

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer

			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			vector<cv::Mat> new_buffer;
			new_buffer.reserve(curr_buffer->size() + out.size());
			new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
			new_buffer.insert( new_buffer.end(), out.begin(), out.end());

			vector<int> new_buffer_labels;
			new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
			new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
			new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
			labels.erase(labels.begin(), labels.begin() + out.size());
			
			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer, new_buffer_labels);

			/******* Restricted Access ********/

		}
	}
// Alternate mode
// Copy everyting to a buffer alternating the outputs buffers
	else if( _mode == Node::Alternate ) {

			// This code considers only one thread
			int i = ctrl;
			int l_size = labels.size();

			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			vector<cv::Mat> new_buffer;
			new_buffer.reserve(curr_buffer->size() + out.size());
			new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
			new_buffer.insert( new_buffer.end(), out.begin(), out.end());

			vector<int> new_buffer_labels;

			new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
			new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
			new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
			labels.erase(labels.begin(), labels.begin() + out.size());
 
			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer, new_buffer_labels);

			// Update control
			ctrl++;
			if(ctrl >= _out_edges.size()){
				ctrl = 0;
			}
			/******* Restricted Access ********/
			
	}
// Chunk mode
// Split the input images into N chunks and tranfer them to the output buffer
// Where N is equal to the number of output nodes
	else if( _mode == Node::Chunk ){

		// This code considers only one thread
		int block_size = out.size()/_out_edges.size();
		for(vector<int>::size_type i=0; i < _out_edges.size(); i++) {

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			vector<cv::Mat> new_buffer;
			vector<int> new_buffer_labels;
			if( i < _out_edges.size()-1 ) {

				if( block_size > 0 ) {

					new_buffer.reserve(curr_buffer->size() + block_size);
					new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
					new_buffer.insert( new_buffer.end(), out.begin(), out.begin() + block_size);
				
					new_buffer_labels.reserve(curr_buffer_labels->size() + block_size);
					new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
					new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + block_size);
					
					labels.erase(labels.begin(), labels.begin() + block_size);
					out.erase(out.begin(), out.begin() + block_size);
				}
			} else {

				new_buffer.reserve(curr_buffer->size() + out.size());
				new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
				new_buffer.insert( new_buffer.end(), out.begin(), out.end());

				new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
				new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
				new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
				
				labels.erase(labels.begin(), labels.begin() + out.size());
				out.clear();
			}

			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer, new_buffer_labels);
			/******* Restricted Access ********/
		}
	}
}





void Node::copy_to_edge(vector<cv::Mat>& out, vector<int>& labels, int edge)
{

	if( edge < _out_edges.size() ) {

		// Lock access to buffer
		boost::mutex::scoped_lock lk(_out_edges.at(edge)->_mutex);

		/******* Restricted Access ********/
		// Get current buffer
		vector<cv::Mat> 	*curr_buffer = _out_edges.at(edge)->get_buffer();
		vector<int> 		*curr_buffer_labels = _out_edges.at(edge)->get_buffer_labels();

		// Concatenate buffers
		vector<cv::Mat> new_buffer;
		new_buffer.reserve(curr_buffer->size() + out.size());
		new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
		new_buffer.insert( new_buffer.end(), out.begin(), out.end());

		vector<int> new_buffer_labels;
		new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
		new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
		new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
		labels.erase(labels.begin(), labels.begin() + out.size());

		// Set new buffer in edge			
		_out_edges.at(edge)->set_buffer(new_buffer, new_buffer_labels);

		/******* Restricted Access ********/

	} else {
		
		cerr << "Requesting copy to an non-existing edge: " 
			 << edge << "(" << _out_edges.size() << ")" << endl;
	}
}






void Node::copy_from_buffer(vector<cv::Mat> &out, vector<int> &label)
{
	
	// Lock access to buffer
	boost::mutex::scoped_lock lk(_in_edges.at(0)->_mutex);

	/******* Restricted Access ********/
	// Get buffer
	vector<cv::Mat> *_buffer = _in_edges.at(0)->get_buffer();
	vector<int> *_buffer_labels = _in_edges.at(0)->get_buffer_labels();

	// Remove first element from buffer
	if( !_buffer->empty() ) {

		out.push_back(_buffer->at(0));
		label.push_back(_buffer_labels->at(0));

		_buffer->erase(_buffer->begin());	
		_buffer_labels->erase(_buffer_labels->begin());	
	}
	/******* Restricted Access ********/
	//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
}






// This function copy chunks of data from the input edges of a given node.
// Opposite to the function copy_from_buffer, this function supports multiple input edges for a node.
void Node::copy_chunk_from_buffer(vector<cv::Mat> &out, vector<int> &labels)
{
	
	// Check all in nodes
	for(vector<int>::size_type i=0; i < _in_edges.size(); i++){

		// Lock access to buffer
		boost::mutex::scoped_lock lk(_in_edges.at(i)->_mutex);

		/******* Restricted Access ********/
		// Get buffer
		vector<cv::Mat> *_buffer = _in_edges.at(i)->get_buffer();
		vector<int> *_buffer_labels = _in_edges.at(i)->get_buffer_labels();

		// Remove first element from buffer
		if( !_buffer->empty() ) {

			vector<cv::Mat> new_block;
			new_block.reserve(out.size() + _buffer->size());
			new_block.insert( new_block.end(), out.begin(), out.end());
			new_block.insert( new_block.end(), _buffer->begin(), _buffer->end());
			out = new_block;
			_buffer->clear();	 


			vector<int> new_block_labels;
			new_block_labels.reserve(labels.size() + _buffer_labels->size());
			new_block_labels.insert( new_block_labels.end(), labels.begin(), labels.end());
			new_block_labels.insert( new_block_labels.end(), _buffer_labels->begin(), _buffer_labels->end());	
			labels = new_block_labels;
			_buffer_labels->clear();
		}
		/******* Restricted Access ********/
		//cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << endl; 
	}
}





void Node::increment_counter()
{
	
	boost::mutex::scoped_lock lk(_mutex_counter);
	_counter++;
}





void Node::increment_threads()
{

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads++;
}





bool Node::check_finished()
{

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads--;
	if( _counter_threads == 0 ) { 
		return true;
	} else {
		return false;
	}
}
