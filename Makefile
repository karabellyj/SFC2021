CPP = g++
CPPFLAGS = -std=c++11 -Wall -Werror -pedantic
PROG = rbm
TRAINFILE = mnist/train-images-idx3-ubyte
TESTFILE = mnist/t10k-images-idx3-ubyte

all: $(PROG)

$(PROG): $(PROG).o main.cpp
	$(CPP) $(CPPFLAGS) $^ -o $@

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $^ -o $@

clean:
	rm -fv *.o *.zip $(PROG)

zip:
	zip xkarab03.zip *.cpp *.h *.pdf Makefile