#include <iostream>
#include <spawn.h>
#include <sys/wait.h>	
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <ctime>
#include <utils.h>
#define NUMB_PIPES 4

extern char **environ;
const static std::string LOCAL_HOME = "/home/nelson";

int main(){
	
	// Start clock
	double begin_time = utils::get_time();	

	std::vector<pid_t> pids;
	std::vector<int> status_vec;
	std::string batch_size = "1500";
	pid_t pid;

	// Create pipes
	for(int k=0; k < NUMB_PIPES; k++){
		std::string pipe_name(LOCAL_HOME +"/CellNet/app/pipe"+std::to_string(k));
		int res = mkfifo(pipe_name.c_str(), 0777);
	    if (res == 0){
	        printf("FIFO created\n");
	    }
	}

	// Start preprocessing
	std::string preprocess_path = LOCAL_HOME + "/CellNet/app/preprocess_stream";
	std::string prediction_path = LOCAL_HOME + "/CellNet/app/prediction_stream";

	char* argV1[] = {&preprocess_path[0]};
	int status = posix_spawn(&pid, preprocess_path.c_str(), NULL, NULL, argV1, environ);
	pids.push_back(pid);
	status_vec.push_back(status);

	// Start prediction
	for(int k=1; k <= NUMB_PIPES; k++){
		std::string device_id = std::to_string(k-1);
		char *argV2[3];
		argV2[0] = &prediction_path[0]; 
		argV2[1] = &device_id[0];
		argV2[2] = &batch_size[0];

		int status = posix_spawn(&pid, prediction_path.c_str(), NULL, NULL, argV2, environ);
		pids.push_back(pid);
		status_vec.push_back(status);

	}
	// Wait processes
	for(int k=0; k < NUMB_PIPES+1; k++){

		waitpid(pids[k], &status_vec[k], 0);
	}

	// Exit
	for(int k=0; k < NUMB_PIPES+1; k++){

		std::cout << "Process: " << std::to_string(k) << " status: " << std::to_string(status_vec[k]) << std::endl;
	}
    
    // Stop clock
	std::cout << "Elapsed Time: " << double( utils::get_time() - begin_time )  << std::endl;
    return 0;
}
