#include "prediction_node.h"

PredictionNode::PredictionNode(std::string id, int mode, int batch_size, std::string model_path): Node(id, mode), _batch_size(batch_size), _model_path(model_path), _data_buffer(), _labels_buffer(), _net(new caffe::Net<float>(model_path.c_str(), caffe::TEST)), _predictions(){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	init_model();
}

void PredictionNode::init_model(){

    caffe::Caffe::SetDevice(0);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    // Initialize the history
  	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = this->_net->params();
}

void *PredictionNode::run(){

	increment_threads();
	while(true){

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(_data_buffer.size() >= _batch_size){

			// Track of number of elements
			for(std::vector<cv::Mat>::size_type i=0; i < _data_buffer.size(); i++){
				increment_counter();
			}
			
			// For each epoch feed the model with a mini-batch of samples
			int epochs = _data_buffer.size()/_batch_size;
			for(int i = 0; i < epochs; i++){	

				// Split batch
				std::vector<cv::Mat> batch;
				std::vector<int> batch_labels;

				// Reseve space
				batch.reserve(_batch_size);
				batch.insert(batch.end(), _data_buffer.begin(), _data_buffer.begin() + _batch_size);
				batch_labels.reserve(_batch_size);
				batch_labels.insert(batch_labels.end(), _labels_buffer.begin(), _labels_buffer.begin() + _batch_size);

				// Remove batch from buffer
				_data_buffer.erase(_data_buffer.begin(), _data_buffer.begin() + _batch_size);
				_labels_buffer.erase(_labels_buffer.begin(), _labels_buffer.begin() + _batch_size);

				// Get memory layer from net
				const boost::shared_ptr<caffe::MemoryDataLayer<float>> data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
				
				// Add matrices
				data_layer->AddMatVector(batch, batch_labels);
				std::cout << "# of Inputed Images: " << std::to_string(batch.size()) << std::endl;

				// Foward
				float loss;
				_net->ForwardPrefilled(&loss);

				// Append outputs
				const boost::shared_ptr<caffe::InnerProduct<float>> out_layer = boost::static_pointer_cast<caffe::InnerProduct<float>>(_net->layer_by_name("fc2"));
				out_layer
			}	
		}
		else{

			// Check if all input nodes have already finished
			bool is_all_done = true;
			for(std::vector<int>::size_type i=0; i < _in_edges.size(); i++){

				if(!_in_edges.at(i)->is_in_node_done()){
					is_all_done = false;
				}
			}

			// All input nodes have finished
			if(is_all_done){
				break;
			}
		}
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "TrainNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_counter) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}
