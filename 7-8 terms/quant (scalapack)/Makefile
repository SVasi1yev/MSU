all: main ro.dat H.dat
	mpirun -np 4 ./main 2 ro.dat 1 H.dat 5

ro.dat: gen_ro
	./gen_ro 2
	
H.dat: gen_h
	./gen_h 2 H.dat "1|0" "1|0" "1|0"
	
main: libscalapack.a scalapack.h main.cpp
	mpicxx -o main main.cpp -L. -lscalapack -lpthread -llapack -lm -lgfortran -lblas
	
gen_ro: gen_ro.cpp
	g++ -o gen_ro gen_ro.cpp
	
gen_h: gen_h.cpp
	g++ -o gen_h gen_h.cpp
	
clean:
	rm *.dat main gen_ro gen_h 
