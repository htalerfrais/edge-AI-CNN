CXX = g++
CC = gcc
TARGET = app

# Flags
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
CFLAGS = -O3 -Wall
LDFLAGS = $(shell pkg-config --libs opencv4) -lm

# Objets
OBJS = main.o neural_network.o contour_detection.o contour_selector.o digit_extractor.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c

contour_detection.o: contour_detection.cpp contour_detection.h
	$(CXX) $(CXXFLAGS) -c contour_detection.cpp

contour_selector.o: contour_selector.cpp contour_selector.h
	$(CXX) $(CXXFLAGS) -c contour_selector.cpp

digit_extractor.o: digit_extractor.cpp digit_extractor.h
	$(CXX) $(CXXFLAGS) -c digit_extractor.cpp

clean:
	rm -f *.o $(TARGET)