libClassicGen.a: ClassicIndivid.o ClassicGenAlg.o
	ar r libClassicGen.a ClassicIndivid.o ClassicGenAlg.o

ClassicIndivid.o: ClassicIndivid.cpp ClassicIndivid.h
	mpic++ -std=c++11 -fopenmp -lgomp -O2 -c ClassicIndivid.cpp

ClassicGenAlg.o: ClassicGenAlg.cpp ClassicGenAlg.h
	mpic++ -std=c++11 -fopenmp -lgomp -O2 -c ClassicGenAlg.cpp

clean:
	rm *.o libClassicGen.a