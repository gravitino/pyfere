CXX=g++
CXXFLAGS= -O3 -march=native -fopenmp -Wall

INCLUDE= -I /usr/include/python2.7/
LINK= -lboost_python-py27

# Python 3 support
#INCLUDE= -I /usr/include/python3.6/
#LINK= -lboost_python3-py36

all: pyfere.so

pyfere.so: pyfere.cpp
	$(CXX) $(CXXFLAGS) -shared $(INCLUDE) pyfere.cpp $(LINK) -fpic -o pyfere.so

clean:
	rm -rf pyfere.so
