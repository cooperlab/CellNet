#include "node.h"

Node::Node(std::string id, int mode): _id(id), _mode(mode), _in_edges(), _out_edges(), _counter(0), runtime_total_first(0), _counter_threads(0), ctrl(0), _labels(){}
std::string Node::get_id(){return _id;}
void Node::insert_in_edge(Edge *edge_ptr){_in_edges.push_back(edge_ptr);}
void Node::insert_out_edge(Edge *edge_ptr){_out_edges.push_back(edge_ptr);}
void Node::copy_to_buffer(std::vector<cv::Mat> out, std::vector<int> &labels){

	if(_mode == 0){	
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			std::vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			std::vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			std::vector<cv::Mat> new_buffer;
			new_buffer.reserve(curr_buffer->size() + out.size());
			new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
			new_buffer.insert( new_buffer.end(), out.begin(), out.end());

			std::vector<int> new_buffer_labels;
			new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
			new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
			new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
			labels.erase(labels.begin(), labels.begin() + out.size());

			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer, new_buffer_labels);

			/******* Restricted Access ********/

		}
	}
	else if(_mode == 1){

			// This code considers only one thread
			int i = ctrl;
			int l_size = labels.size();

			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			std::vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			std::vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			std::vector<cv::Mat> new_buffer;
			new_buffer.reserve(curr_buffer->size() + out.size());
			new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
			new_buffer.insert( new_buffer.end(), out.begin(), out.end());

			std::vector<int> new_buffer_labels;

			new_buffer_labels.reserve(curr_buffer_labels->size() + out.size());
			new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
			new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + out.size());
			labels.erase(labels.begin(), labels.begin() + out.size());
 
			// Set new buffer
			_out_edges.at(i)->set_buffer(new_buffer, new_buffer_labels);

			// Update control
			if(++ctrl >= _out_edges.size()){
				ctrl = 0;
			}
			/******* Restricted Access ********/
			
	}
	else if(_mode == 2){

		// This code considers only one thread
		int block_size = out.size()/_out_edges.size();
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){

			//std::cout << "To Buffer" << std::endl; 
			// Lock access to buffer
			boost::mutex::scoped_lock lk(_out_edges.at(i)->_mutex);

			/******* Restricted Access ********/
			// Get current buffer
			std::vector<cv::Mat> *curr_buffer = _out_edges.at(i)->get_buffer();
			std::vector<int> *curr_buffer_labels = _out_edges.at(i)->get_buffer_labels();

			// Concatenate buffers
			std::vector<cv::Mat> new_buffer;
			std::vector<int> new_buffer_labels;
			if(i < _out_edges.size()-1){

				if(block_size > 0){

					new_buffer.reserve(curr_buffer->size() + block_size);
					new_buffer.insert( new_buffer.end(), curr_buffer->begin(), curr_buffer->end());
					new_buffer.insert( new_buffer.end(), out.begin(), out.begin() + block_size);
				
					new_buffer_labels.reserve(curr_buffer_labels->size() + block_size);
					new_buffer_labels.insert( new_buffer_labels.end(), curr_buffer_labels->begin(), curr_buffer_labels->end());
					new_buffer_labels.insert( new_buffer_labels.end(), labels.begin(), labels.begin() + block_size);
					
					labels.erase(labels.begin(), labels.begin() + block_size);
					out.erase(out.begin(), out.begin() + block_size);
				}
			}
			else{

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

Node::~Node(){

    _in_edges.clear();
    _out_edges.clear();
}

void Node::copy_from_buffer(cv::Mat &out, int &label){
	
	// Lock access to buffer
	boost::mutex::scoped_lock lk(_in_edges.at(0)->_mutex);

	/******* Restricted Access ********/
	// Get buffer
	std::vector<cv::Mat> *_buffer = _in_edges.at(0)->get_buffer();
	std::vector<int> *_buffer_labels = _in_edges.at(0)->get_buffer_labels();

	// Remove first element from buffer
	if(!_buffer->empty()){

		out = _buffer->at(0);
		label = _buffer_labels->at(0);

		_buffer->erase(_buffer->begin());	
		_buffer_labels->erase(_buffer_labels->begin());	
	}
	/******* Restricted Access ********/
	//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
}

// This function copy chunks of data from the input edges of a given node.
// Opposite to the function copy_from_buffer, this function supports multiple input edges for a node.
void Node::copy_chunk_from_buffer(std::vector<cv::Mat> &out, std::vector<int> &labels){
	
	// Check all in nodes
	for(std::vector<int>::size_type i=0; i < _in_edges.size(); i++){

		// Lock access to buffer
		boost::mutex::scoped_lock lk(_in_edges.at(i)->_mutex);

		/******* Restricted Access ********/
		// Get buffer
		std::vector<cv::Mat> *_buffer = _in_edges.at(i)->get_buffer();
		std::vector<int> *_buffer_labels = _in_edges.at(i)->get_buffer_labels();

		// Remove first element from buffer
		if(!_buffer->empty()){

			std::vector<cv::Mat> new_block;
			new_block.reserve(out.size() + _buffer->size());
			new_block.insert( new_block.end(), out.begin(), out.end());
			new_block.insert( new_block.end(), _buffer->begin(), _buffer->end());
			out = new_block;
			_buffer->clear();	 

			std::vector<int> new_block_labels;
			new_block_labels.reserve(labels.size() + _buffer_labels->size());
			new_block_labels.insert( new_block_labels.end(), labels.begin(), labels.end());
			new_block_labels.insert( new_block_labels.end(), _buffer_labels->begin(), _buffer_labels->end());	
			labels = new_block_labels;
			_buffer_labels->clear();
		}
		/******* Restricted Access ********/
		//std::cout << "Node: " << _id << " unlocking	 buffer " << _in_edges.at(0)->_id << std::endl; 
	}
}

void Node::increment_counter(){
	
	boost::mutex::scoped_lock lk(_mutex_counter);
	_counter++;
}

void Node::increment_threads(){

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads++;
}

bool Node::check_finished(){

	boost::mutex::scoped_lock lk(_mutex_ctrl);
	_counter_threads--;
	if(_counter_threads == 0){
		return true;
	}
	else{
		return false;
	}
}