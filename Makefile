# Makefile for qbot

# Use current conda env for python
CONDA_PREFIX = $(shell python -c "import sys; print(sys.prefix)")
LDFLAGS = $(shell python3-config --ldflags)
CONDINC = $(shell python3-config --includes)
CONDLIB = $(shell python3-config --libs)

# libtorch install location
# maybe LIBTORCH should be conda_prefix/lib/python3.12/site-packages/torch as it seems to have approx same files but matches with python torch versions? idk

# LIBTORCH = /usr/local/libtorch
LIBTORCH=$(CONDA_PREFIX)
LIBTORCH_INCLUDE = $(LIBTORCH)/include
LIBTORCH_INCLUDE_CUDA = $(LIBTORCH)/include/torch/csrc/api/include
LIBTORCH_LIB = $(LIBTORCH)/lib
# -Wl,--no-as-needed is required to include cuda libtorch libs even when not directly referenced
TORCH_LIBS = -Wl,--no-as-needed -L$(LIBTORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

BUILD_DIR = build

# Compiler options
CXX = g++ # use g++ compiler
#FLAGS = -I/usr/include/python3.8 -I/usr/include/python3.8 -fdebug-prefix-map=/build/python3.8-4wuY7n/python3.8-3.8.10=. -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector  -DNDEBUG -fwrapv
FLAGS = -I$(CONDA_PREFIX)/include/python3.12 -fstack-protector -DNDEBUG -fwrapv -L$(CONDA_PREFIX)/lib -I$(LIBTORCH_INCLUDE) -I$(LIBTORCH_INCLUDE_CUDA)
HOLD=-specs=/usr/share/dpkg/no-pie-compile.specs 
CXXFLAGS = $(FLAGS) -lpthread -pthread -std=c++17 -g -D_GNU_SOURCE -DWITHOUT_NUMPY -no-pie #-Xlinker -export-dynamic # openmp and pthread, g for debugging

#flags for python 3.6 compatibility
WLDLIBS = -lpython3.6m
WFLAGS = -I/usr/include/python3.6m -I/usr/lib/python3.6 -fstack-protector -Wformat -DNDEBUG -fwrapv
WCXXFLAGS = $(WFLAGS) -lpthread -pthread -lstdc++fs -std=c++17 -g -D_GNU_SOURCE -DWITHOUT_NUMPY -no-pie #-Xlinker -export-dynamic # openmp and pthread, g for debugging

#flags to compile without python integration and all c++ visualization/input
VFLAGS = -fstack-protector -Wall -Wformat -Werror=format-security -DNDEBUG -fwrapv -DNOVIZ
VCXXFLAGS = $(VFLAGS) -lpthread -pthread -std=c++17 -g -D_GNU_SOURCE -no-pie

FCXXFLAGS = $(FLAGS) -lpthread -pthread -std=c++17 -D_GNU_SOURCE -DWITHOUT_NUMPY -no-pie -O3

# .SUFFIXES: .o .cpp
OFILES = $(BUILD_DIR)/QuoridorMain.o $(BUILD_DIR)/Tree.o $(BUILD_DIR)/utility.o $(BUILD_DIR)/Game.o $(BUILD_DIR)/storage.o $(BUILD_DIR)/inference.o
LEOPARD_OFILES = $(BUILD_DIR)/leopard.o $(BUILD_DIR)/Tree.o $(BUILD_DIR)/utility.o $(BUILD_DIR)/storage.o
INFERENCE_OFILES = $(BUILD_DIR)/inference.o $(BUILD_DIR)/Tree.o $(BUILD_DIR)/utility.o $(BUILD_DIR)/storage.o

qbot: $(OFILES)
	#$(CXX) $(CXXFLAGS) $(OFILES) -lpython3.12 -lcrypt -lpthread -ldl  -lutil -lm -lm -o qbot
	$(CXX) $(CONDINC) $(LDFLAGS) $(CXXFLAGS) $(OFILES) -lpython3.12 -lcrypt -lpthread -ldl  -lutil -lm -lm -o qbot
	@echo Produced qbot executable 

work: FLAGS = $(WFLAGS)
work: CXXFLAGS = $(WCXXFLAGS)
work: LDLIBS = $(WLDLIBS)
work: $(OFILES)
	$(CXX) $(WCXXFLAGS) $(OFILES) -lpython3.6m -lcrypt -pthread -ldl -lutil -lm -o qbot
	@echo Produced python3.6 executable

noviz: FLAGS = $(VFLAGS)
noviz: CXXFLAGS = $(VCXXFLAGS)
noviz: $(OFILES)
	$(CXX) $(VCXXFLAGS) $(OFILES) -lcrypt -ldl -lutil -lm -o qbot
	@echo Produced non-visualizing executable

fast: CXXFLAGS = $(FCXXFLAGS)
fast: $(OFILES)
	$(CXX) $(FCXXFLAGS) $(TORCH_LIBS) $(OFILES) -lpython3.12 -lcrypt -lpthread -ldl  -lutil -lm -lm -o qbot
	@echo Produced fast qbot executable 

leopard: $(LEOPARD_OFILES)
	$(CXX) $(FCXXFLAGS) $(LEOPARD_OFILES) -lpython3.12 -o leopard

inference: $(BUILD_DIR)/Tree.o $(BUILD_DIR)/utility.o $(BUILD_DIR)/storage.o $(BUILD_DIR)/inference_main.o
	$(CXX) $(CXXFLAGS) $^ $(TORCH_LIBS) -lpython3.12 -o inference
	@echo Produced inference test executable with LibTorch

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/inference.o: src/inference.cpp
	$(CXX) $(CXXFLAGS) $(TORCH_LIBS) -c $< -o $@

# Special rule for inference main version of the file
$(BUILD_DIR)/inference_main.o: src/inference.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TORCH_LIBS) -DINFERENCE_MAIN -c $< -o $@

clean: 
	$(RM) $(OFILES) $(LEOPARD_OFILES) $(INFERENCE_OFILES)

# Dependency rules for *.o files
src/Tree.o: src/Tree.cpp src/Tree.h src/Global.h
src/QuoridorMain.o: src/QuoridorMain.cpp src/Global.h
src/utility.o: src/utility.cpp src/utility.h src/Global.h 
src/Game.o: src/Game.cpp src/Game.h src/Global.h
src/storage.o: src/storage.cpp src/storage.h
