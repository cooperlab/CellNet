#include "prediction_pipe_node.h"

PredictionPipeNode::PredictionPipeNode(std::string id, int mode, int batch_size, std::string test_model_path, std::string params_file, int device_id, std::string pipe_name): Node(id, mode), _batch_size(batch_size), _data_buffer(), _labels_buffer(), _predictions(), _test_model_path(test_model_path),  _net(), _params_file(params_file), _device_id(device_id), _pipe(0), _pipe_name(pipe_name), _stored_labels(){
	runtime_total_first = utils::get_time();
	_data_buffer.clear();
	_pipe = open(_pipe_name.c_str(), O_RDONLY);
	if(_pipe == -1){
		std::cout << "Fail to open pipe" << std::endl;
		return;
	}
	init_model();
}

void PredictionPipeNode::init_model(){

	// Set gpu
	caffe::Caffe::SetDevice(_device_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::DeviceQuery();
    
    // Create Net
    _net = boost::make_shared<caffe::Net<float>>(_test_model_path.c_str(), caffe::TEST);

    // Initialize the history
  	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = _net->params();
  	_net->CopyTrainedLayersFrom(_params_file.c_str());

  	std::cout << "Model loaded" << std::endl;
}

void *PredictionPipeNode::run(){

	increment_threads();
	int first_idx = 0;

	while(true){

		// Request data
		int res = read_from_pipe(_data_buffer, _labels_buffer);
		if(res){
			if(_batch_size <= _data_buffer.size()){

				// For each epoch feed the model with a mini-batch of samples
				int epochs = _data_buffer.size()/_batch_size;
				for(int i = 0; i < epochs; i++){	

					// Perform prediction
					step(0, _batch_size);

					// Store labels	
					_stored_labels.insert(_stored_labels.end(), _labels_buffer.begin(), _labels_buffer.begin() + _batch_size);
			
					// Erase data
					_data_buffer.erase(_data_buffer.begin(), _data_buffer.begin()+_batch_size);
					_labels_buffer.erase(_labels_buffer.begin(), _labels_buffer.begin()+_batch_size);
				}	
			}
		}
		else{

			// Handle non-multiples
			if(_data_buffer.size() > 0){
				
				std::cout << "Remaining samples: " << std::to_string(_data_buffer.size()) << std::endl; 
				step(0, _data_buffer.size());

				// Store labels	
				_stored_labels.insert(_stored_labels.end(), _labels_buffer.begin(), _labels_buffer.begin() + _data_buffer.size());

				// Clean
				_data_buffer.clear();
				_labels_buffer.clear();
			}

			// Print results
			compute_accuracy();
			print_out_labels();
			break;
		}
	}

	if(check_finished() == true){

		std::cout << "******************" << std::endl << "Prediction" << std::endl << "Total_time_first: " << std::to_string(utils::get_time() - runtime_total_first) << std::endl << "# of elements: " << std::to_string(_labels_buffer.size()) << std::endl << "******************" << std::endl;

		// Notify it has finished
		for(std::vector<int>::size_type i=0; i < _out_edges.size(); i++){
			_out_edges.at(i)->set_in_node_done();
		}
	}

	return NULL;
}

int PredictionPipeNode::step(int first_idx, int batch_size){

	// Split batch
	std::vector<cv::Mat> batch;
	std::vector<int> batch_labels;

	// Reserve space
	batch.reserve(batch_size);
	batch.insert(batch.end(), _data_buffer.begin() + first_idx, _data_buffer.begin() + first_idx + batch_size);
	batch_labels.reserve(batch_size);
	batch_labels.insert(batch_labels.end(), _labels_buffer.begin() + first_idx, _labels_buffer.begin() + first_idx + batch_size);
	
	// Get memory layer from net
	const boost::shared_ptr<caffe::MemoryDataLayer<float>> data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
	
	// Set batch size
	data_layer->set_batch_size(batch_size);

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
	first_idx += batch_size;

	// Clean variables
	batch.clear();
	batch_labels.clear();
	return first_idx;
}

int PredictionPipeNode::read_from_pipe(std::vector<cv::Mat> &outs, std::vector<int> &labels){
	
	// Create buffer
	std::vector<uint8_t> buffer(4);
	
	// Format <height, width, channels, label, data>
	// Read header
	int res = read(_pipe, &buffer[0], buffer.size());
	while(res == 0){
		res = read(_pipe, &buffer[0], buffer.size());
	}

	int height = (int)buffer[0];
	int width = (int)buffer[1];
	int channels = (int)buffer[2];
	int label = (int)buffer[3];

	if((height > 0) && (width > 0) && (channels > 0)){

		// Increment
		_counter++;

		// Read 
		std::vector<uint8_t> buffer_data(height * width * channels);
        res = read(_pipe, &buffer_data[0], buffer_data.size());
		while(res == 0){
            res = read(_pipe, &buffer_data[0], buffer_data.size());
        }

		cv::Mat img(height, width, CV_8UC(channels));
		memcpy(img.data, &buffer_data[0], height * width * channels * sizeof(uint8_t));
		std::vector<cv::Mat> vec_img;
		vec_img.push_back(img);

		// Append new samples
		std::vector<cv::Mat> new_buffer;
		new_buffer.reserve(outs.size() + vec_img.size());
		new_buffer.insert(new_buffer.end(), outs.begin(), outs.end());
		new_buffer.insert(new_buffer.end(), vec_img.begin(), vec_img.end());

		// Update buffer
		outs = new_buffer;
		labels.push_back(label);
		vec_img.clear();
		return 1;
	}
	return 0;
}

void PredictionPipeNode::compute_accuracy(){

	int hit = 0;
	for(int i=0; i < _predictions.size(); i++){

		if(_stored_labels[i] == _predictions[i]){

			hit++;
		}
	}

	float acc = (float)hit/_predictions.size();
	std:: cout << "Accuracy: " << acc <<  " Hits: " << hit << std::endl;
}

void PredictionPipeNode::print_out_labels(){

	for(int i=0; i < _predictions.size(); i++){

		std::cout << "out: " << _predictions[i] << " target: " << _stored_labels[i] << std::endl;
		//std::vector<cv::Mat> input;
		//split(_data_buffer[i], input);
		//for(int k = 0; k < input.size(); k++){

		//	cv::imwrite("/home/nelson/CellNet/src/teste/img" + std::to_string(i)+std::to_string(k) + ".jpg", input[k]);
		//}
	}
}
