LOCAL_HOME = /usr
SDK_HOME = /home/nelson/CellNet
CAFFE_HOME = /home/nelson/caffe
OBJ_DIR = $(SDK_HOME)/obj
SRC_DIR = $(SDK_HOME)/src

# Define C++ compiler
CCC	          = /usr/bin/g++

# Define C compiler options
CFLAGS        = -Wall -c -O2

# Define C++ compiler options
CCCFLAGS      = -Wall -c -O2 -D__STDC_CONSTANT_MACROS 

# Define Includes
CPPFLAGS =  -std=c++11
CPPFLAGS += -I$(SDK_HOME)/include									# Include graph nodes and algorithms 
CPPFLAGS += -I$(LOCAL_HOME)/local/include/opencv  					# Include OpenCV ...
CPPFLAGS += -I$(LOCAL_HOME)/local/include/opencv2  					# Include OpenCV ...
CPPFLAGS += -I$(LOCAL_HOME)/local/include/openslide   				# Include OpenSlide
CPPFLAGS += -I$(LOCAL_HOME)/include/glib-2.0						# Include glib			
CPPFLAGS += -I$(LOCAL_HOME)/lib/x86_64-linux-gnu/glib-2.0/include	# Include glib		
CPPFLAGS += -I$(CAFFE_HOME)/include/		   						# Include Caffe
CPPFLAGS += -I$(CAFFE_HOME)/build/include/		   					# Include Caffe
CPPFLAGS += -I$(LOCAL_HOME)/local/cuda/include						# Include Cuda

# Define location of OpenCV
CV_LIBS       = -L$(LOCAL_HOME)/lib/ -L$(LOCAL_HOME)/local/lib/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lpthread -L$(LOCAL_HOME)/lib/x86_64-linux-gnu -lboost_system -lboost_thread -lhdf5 -lhdf5_cpp -lglog -L$(CAFFE_HOME)/build/lib -lcaffe

# Define location of X Windows libraries and the X11 library names
XLIBS         = -L/usr/X11R6/lib -L/usr/lib -lX11

# Define location of OpenSlide
SLIDE_LIBS =    -L$(LOCAL_HOME)/lib/ -lglib-2.0 -lopenslide

# Define libraries to be linked with
LIBS	      =  -lm $(XLIBS) $(CV_LIBS) $(SLIDE_LIBS) 

# Define linker
LINKER	      = g++

# Define all object files to be the same as CPPSRCS but with all the .cpp and .c suffixes replaced with .o
OBJ           =  $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPPSRCS))
OBJ2           =  $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPPSRCS2))

# Define name of target executable
PROGRAM	      = train
PROGRAM2      = test

# Define all C++ source files here
CPPSRCS       = $(SRC_DIR)/train_model.cpp $(SRC_DIR)/graph_net.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/node.cpp $(SRC_DIR)/edge.cpp $(SRC_DIR)/laplacian_pyramid_node.cpp $(SRC_DIR)/read_node.cpp $(SRC_DIR)/debug_node.cpp $(SRC_DIR)/write_png_node.cpp $(SRC_DIR)/write_hdf5_node.cpp $(SRC_DIR)/grayscale_node.cpp $(SRC_DIR)/train_node.cpp $(SRC_DIR)/prediction_node.cpp
CPPSRCS2       = $(SRC_DIR)/test_model.cpp $(SRC_DIR)/graph_net.cpp $(SRC_DIR)/utils.cpp $(SRC_DIR)/node.cpp $(SRC_DIR)/edge.cpp $(SRC_DIR)/laplacian_pyramid_node.cpp $(SRC_DIR)/read_node.cpp $(SRC_DIR)/debug_node.cpp $(SRC_DIR)/write_png_node.cpp $(SRC_DIR)/write_hdf5_node.cpp $(SRC_DIR)/grayscale_node.cpp $(SRC_DIR)/train_node.cpp $(SRC_DIR)/prediction_node.cpp

##############################################################################
# Define additional rules that make should know about in order to compile our
# files.                                        
##############################################################################

# Define default rule if Make is run without arguments
all : $(PROGRAM) $(PROGRAM2) 

# Define rule for compiling all C++ files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CCC) $(CCCFLAGS) $(CPPFLAGS) -c $< -o $@

	
# Define rule for creating executable
$(PROGRAM) :	$(OBJ)
		@echo -n "Loading $(PROGRAM) ... "
		$(LINKER) -o $(PROGRAM) $(OBJ) $(LIBS) 
		@echo "done"

$(PROGRAM2) :	$(OBJ2)
		@echo -n "Loading $(PROGRAM2) ... "
		$(LINKER) -o $(PROGRAM2) $(OBJ2) $(LIBS) 
		@echo "done"

# Define rule to clean up directory by removing all object, temp and core
# files along with the executable
clean :
	@rm -f $(OBJ) *~ $(PROGRAM)
	@rm -f $(OBJ) *~ $(PROGRAM2)

