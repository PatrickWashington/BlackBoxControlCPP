CC = g++
CFLAGS = -std=c++17
INC = -I ../include

# PY = -L /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7

default: run

run: main.o System.o ILQR.o InvertedPendulum.o NetworkSystem.o NeuralNetwork.o
	$(CC) $(CFLAGS) $(INC) -o run main.o System.o ILQR.o InvertedPendulum.o NetworkSystem.o NeuralNetwork.o

main.o: main.cpp System.hpp InvertedPendulum.hpp NeuralNetwork.hpp NetworkSystem.hpp ILQR.hpp
	$(CC) $(CFLAGS) $(INC) -c main.cpp

System.o: System.cpp System.hpp
	$(CC) $(CFLAGS) $(INC) -c System.cpp

ILQR.o: ILQR.cpp ILQR.hpp
	$(CC) $(CFLAGS) $(INC) -c ILQR.cpp

InvertedPendulum.o: InvertedPendulum.cpp InvertedPendulum.hpp
	$(CC) $(CFLAGS) $(INC) -c InvertedPendulum.cpp

NetworkSystem.o: NetworkSystem.cpp NetworkSystem.hpp
	$(CC) $(CFLAGS) $(INC) -c NetworkSystem.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.hpp
	$(CC) $(CFLAGS) $(INC) -c NeuralNetwork.cpp

clean: 
	$(RM) run *.o

debug: CFLAGS += -g -Wall -Wextra -Werror -pedantic
debug: default

release: CFLAGS += -O2
release: default