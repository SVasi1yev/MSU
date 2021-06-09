#include "train.cpp"
#include "QuantGenAlg.cpp"
#include "ClassicGenAlg.cpp"

#include <torch/torch.h>
#include <ctime>
#include <functional>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int seed = 0;
    const int train_size = 50000;
    const int test_size = 10000;
    const int epoch_num = 30;
    const int batch_size = 200;
    const int gpu_count = torch::cuda::device_count();

    const char alg_type = argv[1][0];
    const int iter_num = std::atoi(argv[7]);
    const int global_pop_size = std::atoi(argv[2]);
    const int individ_size = 12 + 8 * 14 + 2 + 7;
    const bool need_mutation = std::atoi(argv[3]);
    const double mutation_probability = std::atof(argv[4]);
    const bool need_crossover = std::atoi(argv[5]);
    const double crossover_probability = std::atof(argv[6]);
    const int mpi_size = size;
    const int mpi_rank = rank;
    const int omp_size = std::atoi(argv[8]);;

    if (rank == 0) {

        std::cout << "seed = " << seed << "\n"
                  << "train_size = " << train_size << "\n"
                  << "test_size = " << test_size << "\n"
                  << "epoch_num = " << epoch_num << "\n"
                  << "batch_size = " << batch_size << "\n"
                  << "alg_type = " << alg_type << "\n"
                  << "iter_num = " << iter_num << "\n"
                  << "global_pop_size = " << global_pop_size << "\n"
                  << "individ_size = " << individ_size << "\n"
                  << "need_mutation = " << need_mutation << "\n"
                  << "mutation_prob = " << mutation_probability << "\n"
                  << "need_crossover = " << need_crossover << "\n"
                  << "crossover_prob = " << crossover_probability << "\n\n";

    }
    MPI_Barrier(MPI_COMM_WORLD);

    torch::manual_seed(seed);

    torch::Tensor data;
    torch::Tensor targets;

    unsigned char* c_data = new unsigned char[60000 * 3 * 32 * 32];
    unsigned char* c_targets = new unsigned char[60000];

    std::ifstream input;

    for (int i = 0; i < 60000; i++) {
        if (i == 0) {
            input.open("../cifar-10-binary/data_batch_1.bin", std::ios::binary | std::ios::in);
        } else if (i == 10000) {
            input.close();
            input.open("../cifar-10-binary/data_batch_2.bin", std::ios::binary | std::ios::in);
        } else if (i == 20000) {
            input.close();
            input.open("../cifar-10-binary/data_batch_3.bin", std::ios::binary | std::ios::in);
        } else if (i == 30000) {
            input.close();
            input.open("../cifar-10-binary/data_batch_4.bin", std::ios::binary | std::ios::in);
        } else if (i == 40000) {
            input.close();
            input.open("../cifar-10-binary/data_batch_5.bin", std::ios::binary | std::ios::in);
        } else if (i == 50000) {
            input.close();
            input.open("../cifar-10-binary/test_batch.bin", std::ios::binary | std::ios::in);
        }

        input.read((char*) (c_targets + i), 1);
        input.read((char*) (c_data + i * 3 * 32 * 32), 3 * 32 * 32);
    }

    float* norm_data = new float[60000 * 3 * 32 * 32];
    for (int i = 0; i < 60000 * 3 * 32 * 32; i++) {
        norm_data[i] = (float) c_data[i] / 255;
    }
    long* long_targets = new long[60000];
    for (int i = 0; i < 60000; i++) {
        long_targets[i] = c_targets[i];
    }

    data = torch::from_blob(norm_data, {60000, 3, 32, 32}, at::kFloat).clone();
    targets = torch::from_blob(long_targets, {60000}, at::kLong).clone();

    input.close();
    delete[] c_data;
    delete[] c_targets;
    delete[] norm_data;
    delete[] long_targets;

    int data_arr_size;
    if (gpu_count == 0) {
        data_arr_size = 1;
    } else {
        data_arr_size = gpu_count;
    }

    torch::Tensor* train_data = new torch::Tensor[data_arr_size];
    torch::Tensor* train_targets = new torch::Tensor[data_arr_size];
    torch::Tensor* test_data = new torch::Tensor[data_arr_size];
    torch::Tensor* test_targets = new torch::Tensor[data_arr_size];

    if (torch::cuda::is_available()) {
        std::cout << "Rank: " << rank
                  << ". CUDA is available. Training on GPU. "
                  << "GPU num: " << gpu_count << ".\n";

        for (int i = 0; i < gpu_count; i++) {
            train_data[i] = data.slice(
                    0, 0, train_size, 1
            ).to(torch::Device(torch::kCUDA, i));
            train_targets[i] = targets.slice(
                    0, 0, train_size, 1
            ).to(torch::Device(torch::kCUDA, i));

            test_data[i] = data.slice(
                    0, train_size, train_size + test_size, 1
            ).to(torch::Device(torch::kCUDA, i));
            test_targets[i] = targets.slice(
                    0, train_size, train_size + test_size, 1
            ).to(torch::Device(torch::kCUDA, i));
        }
    } else {
        std::cout << "Rank: " << rank
                  << ". CUDA is NOT available. Training on CPU.\n";

        train_data[0] = data.slice(
                0, 0, train_size, 1
        );
        train_targets[0] = targets.slice(
                0, 0, train_size, 1
        );

        test_data[0] = data.slice(
                0, train_size, train_size + test_size, 1
        );
        test_targets[0] = targets.slice(
                0, train_size, train_size + test_size, 1
        );
    }

    std::function<double(const char*, int)> fitness_function
            = std::bind(
                    train,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    train_data, train_targets,
                    test_data, test_targets,
                    epoch_num, batch_size, seed
            );

    MPI_Barrier(MPI_COMM_WORLD);
    double start;
    if (rank == 0) {
        start = MPI_Wtime();
    }

    if (alg_type == 'q') {
        QuantGen::QuantGenAlg alg(
                global_pop_size,
                individ_size,
                need_mutation,
                mutation_probability,
                need_crossover,
                crossover_probability,
                fitness_function,
                QuantGen::angle_function,
                mpi_size,
                mpi_rank,
                omp_size,
                seed
        );
        alg.startAlgorithm(iter_num);
    } else if (alg_type == 'c') {
        ClassicGen::ClassicGenAlg alg(
                global_pop_size,
                individ_size,
                mutation_probability,
                crossover_probability,
                fitness_function,
                mpi_size,
                mpi_rank,
                omp_size,
                seed
        );
        alg.startAlgorithm(iter_num);
    } else {
        std::cout << "Wrong parameter ALG_TYPE\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "TIME: " << MPI_Wtime() - start << "\n";
    }

    delete[] train_data;
    delete[] train_targets;
    delete[] test_data;
    delete[] test_targets;
}