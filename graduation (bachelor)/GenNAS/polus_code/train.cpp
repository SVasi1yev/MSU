#include <torch/torch.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include <mpi.h>
#include <iostream>

using namespace torch;

int bin2dec(const char* bin, int size) {
    int res = 0;
    int pow = 1;
    for (int i = size - 1; i >= 0; i--) {
        res += pow * bin[i];
        pow *= 2;
    }

    return res;
}

int gray2dec(const char* gray, int size) {
    char bin[size];
    char value = gray[0];
    bin[0] = value;
    for (int i = 1; i < size; i++) {
        if (gray[i] == 1) {
            value = (value + 1) % 2;
        }
        bin[i] = value;
    }
    return bin2dec(bin, size);
}

std::function<int(const char*, int)> decoding_function = gray2dec;

const int type_size = 2;
const int layers_num = 9;
const int layer_code_size = 14;
const int optim_size = 2;
const int lr_size = 7;

struct Info {
    int pic_channels = 0;
    int pic_size = 0;
    int out_size = 0;
    int out_classes = 0;

    int layers_count = 0;
    int layers[layers_num];
    int act_funcs_count = 0;
    int act_funcs[layers_num];

    int conv_count = 0;
    int maxpool_count = 0;
    int fc_count = 0;

    int only_fc = false;
};

torch::nn::Conv2d construct_conv(const char* code, Info& info) {
    const int kernal_size_size = 3;
    const int stride_stride = kernal_size_size;
    const int stride_size = 2;
    const int ch_out_stride = stride_stride + stride_size;
    const int ch_out_size = 5;
    const int act_func_stride = ch_out_stride + ch_out_size;
    const int act_func_size = 2;

    int kernel_size = decoding_function(
            code,
            kernal_size_size
    ) + 1;
    int stride = decoding_function(
            code + stride_stride,
            stride_size
    ) + 1;
    int ch_out = decoding_function(
            code + ch_out_stride,
            ch_out_size
    ) + 1;
    int act_func = decoding_function(
            code + act_func_stride,
            act_func_size
    );

    torch::nn::Conv2d res = torch::nn::Conv2d(
            nn::Conv2dOptions(
                    info.pic_channels, ch_out, kernel_size
            ).stride(stride).padding(0).with_bias(true)
    );
    info.conv_count++;

    info.pic_size = floor((double) info.pic_size - kernel_size) / stride + 1;
    info.pic_channels = ch_out;
    info.layers[info.layers_count++] = 0;

    info.act_funcs[info.act_funcs_count++] = act_func;

    return res;
}

torch::nn::MaxPool2d construct_maxpool(const char* code, Info& info) {
    const int kernel_size_size = 2;
    const int stride_stride = kernel_size_size;
    const int stride_size = 2;

    int kernel_size = decoding_function(
            code,
            kernel_size_size
    ) + 1;
    int stride = decoding_function(
            code + stride_stride,
            stride_size
    ) + 1;

    torch::nn::MaxPool2d res = torch::nn::MaxPool2d(
            nn::MaxPool2dOptions(kernel_size).stride(stride)
    );
    info.maxpool_count++;

    info.pic_size = floor((double) info.pic_size - kernel_size) / stride + 1;
    info.layers[info.layers_count++] = 1;

    return res;
}

torch::nn::Linear construct_fc(const char* code, Info& info) {
    const int size_size = 10;
    const int act_func_stride = size_size;
    const int act_func_size = 2;

    int size = decoding_function(
            code,
            size_size
    ) + 1;
    int act_func = decoding_function(
            code + act_func_stride,
            act_func_size
    );

    torch::nn::Linear res = nullptr;
    if (info.only_fc) {
        res = torch::nn::Linear(info.out_size, size);
    } else {
        res = torch::nn::Linear(
                info.pic_channels * info.pic_size * info.pic_size,
                size
        );
    }
    info.fc_count++;

    info.out_size = size;
    info.only_fc = true;
    info.layers[info.layers_count++] = 2;

    info.act_funcs[info.act_funcs_count++] = act_func;

    return res;
}

struct Net : torch::nn::Module {
    Net(const char* obs) {

        info.pic_size = 32;
        info.pic_channels = 3;
        info.out_classes = 10;

        convs.push_back(register_module(
                "c0", construct_conv(obs, info)
        ));

        char type_str[3];
        type_str[2] = '\0';
        int code_stride = layer_code_size - type_size;
        for (int i = 1; i < layers_num - 1; i++) {
            int type = decoding_function(obs + code_stride, 2);
            if ((type == 0) && !(info.only_fc)) {
                type_str[0] = 'c';
                type_str[1] = '0' + info.conv_count;
                convs.push_back(register_module(
                        type_str, construct_conv(obs + code_stride + type_size, info)));
            } else if ((type == 1) && !(info.only_fc)) {
                type_str[0] = 'm';
                type_str[1] = '0' + info.maxpool_count;
                maxpools.push_back(register_module(
                        type_str, construct_maxpool(obs + code_stride + type_size, info)));
            } else if (((type == 2) || info.only_fc) && (type != 3)) {
                type_str[0] = 'f';
                type_str[1] = '0' + info.fc_count;
                fcs.push_back(register_module(
                        type_str, construct_fc(obs + code_stride + type_size, info)));
            }
            code_stride += layer_code_size;
            if (info.pic_size <= 0) {
                incorrect = true;
                return;
            }
        }

        type_str[0] = 'f';
        type_str[1] = '0' + info.fc_count;
        torch::nn::Linear res = nullptr;
        if (info.only_fc) {
            res = torch::nn::Linear(info.out_size, info.out_classes);
        } else {
            res = torch::nn::Linear(
                    info.pic_channels * info.pic_size * info.pic_size,
                    info.out_classes
            );
        }
        info.out_size = info.out_classes;
        info.layers[info.layers_count++] = 2;

        info.act_funcs[info.act_funcs_count++] = -1;

        fcs.push_back(register_module(type_str, res));
    }

    torch::Tensor forward(torch::Tensor x) {
        int convs_count = 0;
        int maxpools_count = 0;
        int fcs_count = 0;
        int act_funcs_count = 0;
        for (int i = 0; i < info.layers_count; i++) {
            if (info.layers[i] == 0) {
                x = convs[convs_count++]->forward(x);
                int act_func = info.act_funcs[act_funcs_count++];
                if (act_func == 0) {
                    x = torch::sigmoid(x);
                } else if(act_func == 1) {
                    x = torch::relu(x);
                } else if(act_func == 2) {
                    x = torch::tanh(x);
                } else if (act_func == 3) {
                    x = torch::softplus(x);
                }
            } else if (info.layers[i] == 1) {
                x = maxpools[maxpools_count++]->forward(x);
            } else if (info.layers[i] == 2) {
                if (fcs_count == 0) {
                    x = x.reshape({x.size(0), -1});
                }
                x = fcs[fcs_count++]->forward(x);
                int act_func = info.act_funcs[act_funcs_count++];
                if (act_func == 0) {
                    x = torch::sigmoid(x);
                } else if(act_func == 1) {
                    x = torch::relu(x);
                } else if(act_func == 2) {
                    x = torch::tanh(x);
                } else if (act_func == 3) {
                    x = torch::softplus(x);
                }
            }
        }
        return torch::log_softmax(x, 1);
    }

    Info info;

    std::vector<torch::nn::Conv2d> convs;
    std::vector<torch::nn::MaxPool2d> maxpools;
    std::vector<torch::nn::Linear> fcs;

    bool incorrect = false;
};

double train(const char* observation, int obs_size,
             torch::Tensor* train_data, torch::Tensor* train_targets,
             torch::Tensor* test_data, torch::Tensor* test_targets,
             int epoch_num, int batch_size, int seed
)
{
    int idx = 0;

    torch::manual_seed(seed);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        idx = omp_get_thread_num();
        device = torch::Device(torch::kCUDA, idx);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto net = std::make_shared<Net>(observation);
    if (net->incorrect) {
        return 0;
    }
    net->to(device);

    int optim_code = decoding_function(observation + layer_code_size * (layers_num - 1) - type_size, optim_size);
    double lr = 1e-4 * (decoding_function(observation + layer_code_size * (layers_num - 1) - type_size + optim_size, lr_size) + 1);

    torch::optim::SGD sgd = torch::optim::SGD(
            net->parameters(),
            torch::optim::SGDOptions(lr)
    );
    torch::optim::Adagrad adagrad = torch::optim::Adagrad(
            net->parameters(),
            torch::optim::AdagradOptions(lr)
    );
    torch::optim::RMSprop rmsprop = torch::optim::RMSprop(
            net->parameters(),
            torch::optim::RMSpropOptions(lr)
    );
    torch::optim::Adam adam = torch::optim::Adam(
            net->parameters(),
            torch::optim::AdamOptions(lr)
    );

    int log_interval = 10;
    double last_accs[5];
    double max_accuracy = 0;
    for (size_t epoch = 1; epoch <= epoch_num; epoch++) {
        double accuracy = 0;
        for (int batch_index = 0; batch_index * batch_size < train_data[idx].size(0); batch_index++) {
            net->zero_grad();

            torch::Tensor X = train_data[idx].slice(
                    0,
                    batch_index * batch_size,
                    (batch_index + 1) * batch_size < train_data[idx].size(0) ? (batch_index + 1) * batch_size
                                                                             : train_data[idx].size(0),
                    1
            );

            torch::Tensor y = train_targets[idx].slice(
                    0,
                    batch_index * batch_size,
                    (batch_index + 1) * batch_size < train_data[idx].size(0) ? (batch_index + 1) * batch_size
                                                                             : train_data[idx].size(0),
                    1
            );

            torch::Tensor pred = net->forward(X);
            torch::Tensor loss = torch::nll_loss(pred, y);
            loss.backward();

            if (optim_code == 0) {
                sgd.step();
            } else if (optim_code == 1) {
                adagrad.step();
            } else if (optim_code == 2) {
                rmsprop.step();
            } else if (optim_code == 3) {
                adam.step();
            }

            accuracy += (pred.argmax(1, true).view(-1) == y).sum().item<int>();
        }

        torch::Tensor pred = net->forward(test_data[idx]);
        torch::Tensor loss = torch::nll_loss(pred, test_targets[idx]);
        accuracy = (double) (pred.argmax(1, true).view(-1) == test_targets[idx]).sum().item<int>()
                   / test_data[idx].size(0);
        if (epoch_num <= 5) {
            last_accs[epoch_num - 1] = accuracy;
        } else {
            double min = 101.0;
            for (int i = 0; i < 5; i++) {
                if (last_accs[i] < min) {
                    min = last_accs[i];
                }
            }
            if (((accuracy - min) / accuracy) < 0.02) {
                return max_accuracy;
            }
        }
        if (max_accuracy < accuracy) {
            max_accuracy = accuracy;
        }
    }

    return max_accuracy;
}
