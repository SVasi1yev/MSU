all: main.cpp
	mpic++ -fopenmp -O2 -o main main.cpp

clean:
	rm main rand_gen_complexd print_complexd *.dat

rand_gen_complexd: rand_gen_complexd.cpp
	g++ -o rand_gen_complexd rand_gen_complexd.cpp

print_complexd: print_complexd.cpp
	g++ -o print_complexd print_complexd.cpp