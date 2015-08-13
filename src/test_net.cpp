#include <iostream>
#include <spawn.h>
#include <sys/wait.h>	
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#define NUMB_PIPES 4
#define BATCH_SIZE 10
extern char **environ;

int main(){

	std::vector<pid_t> pids;
	std::vector<int> status_vec;

	// Create pipes
	for(int k=0; k < NUMB_PIPES; k++){

		int res = mkfifo("pipe"+std::to_string(k), 0777);
	    if (res == 0){
	        printf("FIFO created\n");
	    }
	}

	// Start preprocessing
	char *argV1[] = {"/home/nnauata/CellNet/app/preprocess_stream"};
	int status = posix_spawn(&pids[0], "/home/nnauata/CellNet/app/preprocess_stream", NULL, NULL, argV1, environ);
	status_vec.push_back(status);

	// Start prediction
	for(int k=1; k <= NUMB_PIPES; k++){

		char *argV2[] = {k-1, BATCH_SIZE, "/home/nnauata/CellNet/app/prediction_stream"};
		int status = posix_spawn(&pids[k], "/home/nnauata/CellNet/app/prediction_stream", NULL, NULL, argV2, environ);
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