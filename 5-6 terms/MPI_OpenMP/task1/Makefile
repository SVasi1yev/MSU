all: report

test: main.o equal.o tests/A_test.dat tests
	./main.o tests/A_test.dat tests/B_test.dat tests/C.dat 0
	./equal.o tests/C.dat tests/C_test.dat

report: rand_matr_gen.o time_file_gen.o main.o format_time_file.o
	./rand_matr_gen.o f 1000 1000 A.dat
	./rand_matr_gen.o f 1000 1000 B.dat
	./time_file_gen.o
	./main.o A.dat B.dat C.dat 0
	./main.o A.dat B.dat C.dat 0
	./main.o A.dat B.dat C.dat 0
	./main.o A.dat B.dat C.dat 0
	./main.o A.dat B.dat C.dat 1
	./main.o A.dat B.dat C.dat 1
	./main.o A.dat B.dat C.dat 1
	./main.o A.dat B.dat C.dat 1
	./main.o A.dat B.dat C.dat 2
	./main.o A.dat B.dat C.dat 2
	./main.o A.dat B.dat C.dat 2
	./main.o A.dat B.dat C.dat 2
	./main.o A.dat B.dat C.dat 3
	./main.o A.dat B.dat C.dat 3
	./main.o A.dat B.dat C.dat 3
	./main.o A.dat B.dat C.dat 3
	./main.o A.dat B.dat C.dat 4
	./main.o A.dat B.dat C.dat 4
	./main.o A.dat B.dat C.dat 4
	./main.o A.dat B.dat C.dat 4
	./main.o A.dat B.dat C.dat 5
	./main.o A.dat B.dat C.dat 5
	./main.o A.dat B.dat C.dat 5
	./main.o A.dat B.dat C.dat 5
	./format_time_file.o
	gnuplot "GNUPlotScript.plt"

clean:
	rm -rf *.o *.dat tests/C.dat

main.o: main.cpp
	g++ -std=c++11 main.cpp -o main.o

equal.o: equal.cpp
	g++ -std=c++11 equal.cpp -o equal.o

rand_matr_gen.o: rand_matr_gen.cpp
	g++ -std=c++11 rand_matr_gen.cpp -o rand_matr_gen.o

time_file_gen.o: time_file_gen.cpp
	g++ -std=c++11 time_file_gen.cpp -o time_file_gen.o

format_time_file.o: format_time_file.cpp
	g++ -std=c++11 format_time_file.cpp -o format_time_file.o

