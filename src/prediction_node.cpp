#include "prediction_node.h"

PredictionNode::PredictionNode(std::string id, int mode, int batch_size, std::string test_model_path, std::string params_file): Node(id, mode), _batch_size(batch_size), _data_buffer(), _labels_buffer(), _predictions(), _net(new caffe::Net<float>(test_model_path.c_str(), caffe::TEST)), _params_file(params_file){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	init_model();
}

void PredictionNode::init_model(){

    caffe::Caffe::set_mode(caffe::Caffe::CPU);

    // Initialize the history
  	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = _net->params();
  	_net->CopyTrainedLayersFrom(_params_file.c_str());
  	std::cout << "Model loaded" << std::endl;
}

void *PredictionNode::run(){

	increment_threads();
	int first_idx = 0;
	while(true){

		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(first_idx + _batch_size <= _data_buffer.size()){

			// For each epoch feed the model with a mini-batch of samples
			int epochs = (_data_buffer.size() - first_idx)/_batch_size;
			for(int i = 0; i < epochs; i++){	

				// Split batch
				std::vector<cv::Mat> batch;
				std::vector<int> batch_labels;

				// Reserve space
				batch.reserve(_batch_size);
				batch.insert(batch.end(), _data_buffer.begin() + first_idx, _data_buffer.begin() + first_idx + _batch_size);
				batch_labels.reserve(_batch_size);
				batch_labels.insert(batch_labels.end(), _labels_buffer.begin() + first_idx, _labels_buffer.begin() + first_idx + _batch_size);
				
				// Get memory layer from net
				const boost::shared_ptr<caffe::MemoryDataLayer<float>> data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
				
				// Add matrices
				data_layer->AddMatVector(batch, batch_labels);

				// Foward
				float loss;
				_net->ForwardPrefilled(&loss);
				const boost::shared_ptr<caffe::Blob<float> >& out_layer = _net->blob_by_name("ip2");

				const float* results = out_layer->cpu_data();

				// Append outputs
				for(int j=0; j < out_layer->shape(0); j++){
					
					int idx_max = 0;
					float max = results[j*out_layer->shape(1) + 0];

					// Argmax
					for(int k=0; k < out_layer->shape(1); k++){

						if(results[j*out_layer->shape(1) + k] > max){

							max = results[j*out_layer->shape(1) + k];
							idx_max = k;
						}
					}

					_predictions.push_back(idx_max);
				}
				first_idx += _batch_size;
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

				// Print results
				compute_accuracy();
				print_out_labels();
				break;
			}
		}
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "TrainNode" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_labels_buffer.size()) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}

void PredictionNode::compute_accuracy(){

	int hit = 0;
	for(int i=0; i < _predictions.size(); i++){

		if(_labels_buffer[i] == _predictions[i]){

			hit++;
		}
	}

	float acc = (float)hit/_predictions.size();
	std:: cout << "Accuracy: " << acc <<  " Hits: " << hit << std::endl;
}

void PredictionNode::print_out_labels(){

	for(int i=0; i < _predictions.size(); i++){

		std::cout << "out: " << _predictions[i] << " target: " << _labels_buffer[i] << std::endl;
		//std::vector<cv::Mat> input;
		//split(_data_buffer[i], input);
		//for(int k = 0; k < input.size(); k++){

		//	cv::imshow("channel#" + std::to_string(k), input[k]);
		//}
		//cv::waitKey(0);
	}
}
