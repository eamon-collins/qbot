# Makefile for qbot

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -g -D_GNU_SOURCE -no-pie# openmp and pthread, g for debugging


.SUFFIXES: .o .cpp
OFILES = src/QuoridorMain.o src/Tree.o src/utility.o src/Game.o src/storage.o

qbot: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o qbot
	@echo Produced qbot executable 

clean: 
	$(RM) $(OFILES)

# Dependency rules for *.o files
src/Tree.o: src/Tree.cpp src/Tree.h src/Global.h
src/QuoridorMain.o: src/QuoridorMain.cpp src/Global.h
src/utility.o: src/utility.cpp src/utility.h src/Global.h 
src/Game.o: src/Game.cpp src/Game.h src/Global.h
src/storage.o: src/storage.cpp src/storage.h