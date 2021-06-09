libQuantGen.a: QuantBit.o QuantIndivid.o QuantGenAlg.o
	ar r libQuantGen.a QuantBit.o QuantIndivid.o QuantGenAlg.o

QuantBit.o: QuantBit.cpp QuantBit.h
	mpic++ -std=c++11 -fopenmp -lgomp -O2 -c QuantBit.cpp

QuantIndivid.o: QuantIndivid.cpp QuantIndivid.h
	mpic++ -std=c++11 -fopenmp -lgomp -O2 -c QuantIndivid.cpp

QuantGenAlg.o: QuantGenAlg.cpp QuantGenAlg.h
	mpic++ -std=c++11 -fopenmp -lgomp -O2 -c QuantGenAlg.cpp

clean:
	rm *.o libQuantGen.a