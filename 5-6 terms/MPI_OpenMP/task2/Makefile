all: rand_matr_gen main format_file
	./rand_matr_gen f 500 500 A.dat
	./rand_matr_gen f 500 500 B.dat
	./main A.dat B.dat output.dat
	./rand_matr_gen f 1000 1000 A.dat
	./rand_matr_gen f 1000 1000 B.dat
	./main A.dat B.dat output.dat
	./rand_matr_gen f 1500 1500 A.dat
	./rand_matr_gen f 1500 1500 B.dat
	./main A.dat B.dat output.dat
	./rand_matr_gen f 2000 2000 A.dat
	./rand_matr_gen f 2000 2000 B.dat
	./main A.dat B.dat output.dat
	./rand_matr_gen f 2500 2500 A.dat
	./rand_matr_gen f 2500 2500 B.dat
	./main A.dat B.dat output.dat
	./format_file output.dat format_output.dat
	gnuplot GNU_plot_script.plt

clean:
	rm -rf main rand_matr_gen format_file *.dat

rand_matr_gen: rand_matr_gen.cpp
	g++ -std=c++11 rand_matr_gen.cpp -o rand_matr_gen

main: main.cpp
	g++ -std=c++11 -I /usr/local/include/ -o main main.cpp /usr/lib64/libpapi.so

format_file: format_file.cpp
	g++ -std=c++11 format_file.cpp -o format_file
