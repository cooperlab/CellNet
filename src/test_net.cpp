#include <iostream>
#include <spawn.h>
#include <sys/wait.h>	
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

extern char **environ;

int main(){

	std::cout << "Started Running Preprocess" << std::endl;
	pid_t pid1;
	pid_t pid2;
	char *argV1[] = {"/home/nelson/CellNet/app/preprocess_stream"};
	char *argV2[] = {"/home/nelson/CellNet/app/prediction_stream"};
	std::string pipe_name = "pipe0";

	// Create pipe
	int res = mkfifo(pipe_name.c_str(), 0777);
    if (res == 0){
        printf("FIFO created\n");
    }

	int status = posix_spawn(&pid1, "/home/nelson/CellNet/app/preprocess_stream", NULL, NULL, argV1, environ);
	if (status == 0) {

        std::cout << "Preprocess pid: " << pid1 << std::endl;
        int status2 = posix_spawn(&pid2, "/home/nelson/CellNet/app/prediction_stream", NULL, NULL, argV2, environ);
        if(status2 == 0) {

	    	std::cout << "Prediction pid: " << pid2 << std::endl;
	        if (waitpid(pid1, &status, 0) != -1) {

	        	std::cout << "Prediction exited with status " << status2 << std::endl;
		        if (waitpid(pid2, &status2, 0) != -1) {

		            std::cout << "Preprocess exited with status " << status << std::endl;
		        }
		    	
	    	}
	    }
    	else {
        	std::cout << "posix_spawn: " << status2 << std::endl;
    	}
    } else {
        std::cout << "posix_spawn: " << status << std::endl;
    }
	
}