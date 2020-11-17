# Makefile for qbot

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -g -D_GNU_SOURCE -no-pie# openmp and pthread, g for debugging

# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = src/QuoridorMain.o src/Tree.o src/utility.o src/Game.o src/storage.o

qbot: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o qbot
	@echo Produced qbot executable 

clean: 
	$(RM) *.o *~


# Dependency rules for *.o files
src/Tree.o: src/Tree.cpp src/Tree.h src/utility.cpp src/Global.h
src/QuoridorMain.o: src/QuoridorMain.cpp src/Tree.cpp src/Game.cpp src/Global.h
src/utility.o: src/utility.h src/utility.cpp src/Global.h 
src/Game.o: src/Game.cpp src/Game.h src/Global.h
src/storage.o: src/storage.cpp src/storage.h