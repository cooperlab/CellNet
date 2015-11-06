#include "prediction_node.h"

PredictionNode::PredictionNode(std::string id, int mode, int batch_size, std::string test_model_path, 
							   std::string params_file, int device_id, std::string outFilename) : 
Node(id, mode), 
_batch_size(batch_size), 
_data_buffer(), 
_labels_buffer(), 
_predictions(), 
_test_model_path(test_model_path),  
_net(), 
_params_file(params_file), 
_device_id(device_id),
_outFilename(outFilename),
_out_layer(),
_data_layer()
{

	std::cout << "In PredictionNode constructor" << std::endl;

	runtime_total_first = utils::get_time();
	_data_buffer.clear();
}




void PredictionNode::init_model(){

	std::cout << "In PredictionNode::init_model" << std::endl;

	// Set gpu
	caffe::Caffe::SetDevice(_device_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::DeviceQuery();
    
    // Create Net
    _net = boost::make_shared<caffe::Net<float>>(_test_model_path.c_str(), caffe::TEST);

    // Initialize the history
  	const std::vector<boost::shared_ptr<caffe::Blob<float> > >& net_params = _net->params();
  	_net->CopyTrainedLayersFrom(_params_file.c_str());

  	_data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layer_by_name("data"));
	_out_layer = _net->blob_by_name("prob");

  	std::cout << "Model loaded" << std::endl;
}





void *PredictionNode::run(){

	increment_threads();
	int first_idx = 0;

	init_model();


	while(true){
		copy_chunk_from_buffer(_data_buffer, _labels_buffer);
		if(first_idx + _batch_size <= _data_buffer.size()){

			// For each epoch feed the model with a mini-batch of samples
			int epochs = (_data_buffer.size() - first_idx)/_batch_size;
			for(int i = 0; i < epochs; i++){	

				first_idx = step(first_idx, _batch_size);
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

				// Handle non-multiples
				if(first_idx != _data_buffer.size()-1){
					
					std::cout << "Remaining samples: " << std::to_string(_data_buffer.size()-first_idx) << std::endl; 
					step(first_idx, _data_buffer.size()-first_idx);
				}

				// Print results
				compute_accuracy();
				print_out_labels();
				write_to_file();
				break;
			}
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

int PredictionNode::step(int first_idx, int batch_size){

	// Split batch
	std::vector<cv::Mat> batch;
	std::vector<int> batch_labels;

	// Reserve space
	batch.reserve(batch_size);
	batch.insert(batch.end(), _data_buffer.begin() + first_idx, _data_buffer.begin() + first_idx + batch_size);
	batch_labels.reserve(batch_size);
	batch_labels.insert(batch_labels.end(), _labels_buffer.begin() + first_idx, _labels_buffer.begin() + first_idx + batch_size);
	
	// Set batch size
	_data_layer->set_batch_size(batch_size);

	// Add matrices
	_data_layer->AddMatVector(batch, batch_labels);

	// Foward
	float loss;
	_net->ForwardPrefilled(&loss);

	const float* results = _out_layer->cpu_data();

	// Append outputs
	for(int j=0; j < _out_layer->shape(0); j++){
		
		int idx_max = 0;
		float max = results[j*_out_layer->shape(1) + 0];

		// Argmax
		for(int k=0; k < _out_layer->shape(1); k++){

			if(results[j*_out_layer->shape(1) + k] > max){

				max = results[j*_out_layer->shape(1) + k];
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

/*
caffe::Blob<Dtype> *PredictionNode::convert_to_blob(std::vector<cv::Mat> batch){

	// Set size
	cv::Size s = batch[0].size();
	int batch_size = batch.size();
	int channels = batch[0].channels();
	int width = s.width;
	int height = s.height;

	// convert the image to a caffe::Blob
	caffe::Blob<Dtype> *blob = new caffe::Blob<Dtype>(batch_size, c, width, height);
	for(int k=0; k < batch_size; k++){
		for (int c = 0; c < channels; ++c) {
		    for (int h = 0; h < height; ++h) {
		        for (int w = 0; w < width; ++w) {

		            blob->mutable_cpu_data()[blob->offset(k, c, h, w)] = batch[k].at<cv::Vec3b>(h, w)[c];
		        }
		    }
		}
	}
	return blob;
}
*/

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





void PredictionNode::write_to_file(){

	std::ofstream out;
	out.open(_outFilename, std::ofstream::out  | std::ios::app);

	for(int i=0; i < _predictions.size(); i++){

		 out <<  _predictions[i] << ";" << _labels_buffer[i] << std::endl;
	}
	out.close();
}





void PredictionNode::print_out_labels(){

	for(int i=0; i < _predictions.size(); i++){

		std::cout << "out: " << _predictions[i] << " target: " << _labels_buffer[i] << std::endl;
		std::vector<cv::Mat> input;
		split(_data_buffer[i], input);
//		for(int k = 0; k < input.size(); k++){

//			cv::imwrite("/home/nelson/CellNet/src/teste/img" + std::to_string(i)+std::to_string(k) + ".jpg", input[k]);
//		}
	}
}
