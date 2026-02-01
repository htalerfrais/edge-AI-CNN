CXX = g++
CC = gcc
TARGET = app

# Flags
CXXFLAGS = -O3 -Wall -std=c++17 $(shell pkg-config --cflags opencv4)
CFLAGS = -O3 -Wall
LDFLAGS = $(shell pkg-config --libs opencv4) -lm

# Objets
OBJS = main.o neural_network.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c

clean:
	rm -f *.o $(TARGET)