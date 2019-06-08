#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>
#include <string>

size_t const BLOCK_SIZE = 256;

void prefix_sum(std::vector<double> &arr, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * arr.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * arr.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

    cl::Kernel kernel(program, "prefix_sum");
    kernel.setArg(0, dev_input);
    kernel.setArg(1, dev_output);
    kernel.setArg(2, cl::__local(sizeof(double) * BLOCK_SIZE));
    kernel.setArg(3, cl::__local(sizeof(double) * BLOCK_SIZE));
    kernel.setArg(4, arr.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange((((arr.size() + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE),
                               cl::NDRange(BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

    if (arr.size() > BLOCK_SIZE) {
        std::vector<double> prefixes((arr.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cl::Buffer dev_input_c(context, CL_MEM_READ_ONLY, sizeof(double) * arr.size());
        cl::Buffer dev_output_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * prefixes.size());

        queue.enqueueWriteBuffer(dev_input_c, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

        cl::Kernel kernel_c(program, "copy");
        kernel_c.setArg(0, dev_input_c);
        kernel_c.setArg(1, dev_output_c);
        kernel_c.setArg(2, arr.size());
        kernel_c.setArg(3, prefixes.size());
        queue.enqueueNDRangeKernel(kernel_c, cl::NullRange,
                                   cl::NDRange((((arr.size() + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE),
                                   cl::NDRange(BLOCK_SIZE));
        queue.enqueueReadBuffer(dev_output_c, CL_TRUE, 0, sizeof(double) * prefixes.size(), &prefixes[0]);
        prefixes[0] = 0;
        prefix_sum(prefixes, context, program, queue);
        cl::Buffer dev_input_partial(context, CL_MEM_READ_ONLY, sizeof(double) * prefixes.size());
        cl::Buffer dev_input_p(context, CL_MEM_READ_ONLY, sizeof(double) * arr.size());
        cl::Buffer dev_output_p(context, CL_MEM_WRITE_ONLY, sizeof(double) * arr.size());

        queue.enqueueWriteBuffer(dev_input_partial, CL_TRUE, 0, sizeof(double) * prefixes.size(), &prefixes[0]);
        queue.enqueueWriteBuffer(dev_input_p, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);

        cl::Kernel kernel_p(program, "add");
        kernel_p.setArg(0, dev_input_partial);
        kernel_p.setArg(1, dev_input_p);
        kernel_p.setArg(2, dev_output_p);
        kernel_p.setArg(3, arr.size());
        queue.enqueueNDRangeKernel(kernel_p, cl::NullRange,
                                   cl::NDRange((((prefixes.size() + BLOCK_SIZE - 1) / BLOCK_SIZE)) * BLOCK_SIZE),
                                   cl::NDRange(BLOCK_SIZE));
        queue.enqueueReadBuffer(dev_output_p, CL_TRUE, 0, sizeof(double) * arr.size(), &arr[0]);
    }
}

int main() {
    std::freopen("input.txt", "r", stdin);
    std::freopen("output.txt", "w", stdout);

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // create a message to send to kernel
        size_t N;
        std::vector<double> input;
        std::cin >> N;
        for (size_t i = 0; i < N; ++i) {
            double x;
            std::cin >> x;
            input.push_back(x);
        }

        prefix_sum(input, context, program, queue);

        for (auto &elem: input)
            std::cout << std::setprecision(3) << elem << " ";

    }
    catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
