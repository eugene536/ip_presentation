all: main.cpp
	g++-4.9 -std=c++1y -o main.o main.cpp -DDEBUG -g3 `pkg-config opencv --libs --cflags`

run: main.o 
	./main.o

clean:
	rm -f main.o ./results/*
