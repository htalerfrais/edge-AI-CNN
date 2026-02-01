CXX = g++
CC = gcc
TARGET = app

# Flags
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
CFLAGS = -O3 -Wall
LDFLAGS = $(shell pkg-config --libs opencv4) -lm

# Objets
COMMON_OBJS = neural_network.o contour_detection.o contour_selector.o digit_extractor.o process_frame.o
TARGET_OBJS = main.o $(COMMON_OBJS)
TEST_OBJS = main_images.o $(COMMON_OBJS)

all: app

app: $(TARGET_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

app_images: $(TEST_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

main.o: main.cpp process_frame.h
	$(CXX) $(CXXFLAGS) -c main.cpp

main_images.o: main_images.cpp process_frame.h
	$(CXX) $(CXXFLAGS) -c main_images.cpp

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c

contour_detection.o: contour_detection.cpp contour_detection.h
	$(CXX) $(CXXFLAGS) -c contour_detection.cpp

contour_selector.o: contour_selector.cpp contour_selector.h
	$(CXX) $(CXXFLAGS) -c contour_selector.cpp

digit_extractor.o: digit_extractor.cpp digit_extractor.h
	$(CXX) $(CXXFLAGS) -c digit_extractor.cpp

process_frame.o: process_frame.cpp process_frame.h contour_detection.h contour_selector.h digit_extractor.h
	$(CXX) $(CXXFLAGS) -c process_frame.cpp

clean:
	rm -f *.o $(TARGET) app_images