#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include "cl.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>


using namespace std;

int main() {
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        ifstream cl_file("convolution.cl");
        string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
        cl::Program::Sources source(1, make_pair(cl_string.c_str(),
                                                 cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        }
        catch (cl::Error const &e) {
            string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            cout << endl << e.what() << " : " << e.err() << endl;
            cout << log_str;
            return 0;
        }

        ifstream in("input.txt");

        // create a message to send to kernel
        size_t const block_size = 256;
        size_t n, m;
        in >> n >> m;
        vector<double> a;
        vector<double> b;
        vector<double> c;
        a.resize(n * n);
        b.resize(m * m);
        c.resize(n * n);
        for (size_t i = 0; i < n * n; i++) {
            in >> a[i];
        }
        for (size_t i = 0; i < m * m; i++) {
            in >> b[i];
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * n * n);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * m * m);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * n * n);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * n * n, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * m * m, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_gmem(program, "convolution");
        kernel_gmem.setArg(0, dev_a);
        kernel_gmem.setArg(1, dev_b);
        kernel_gmem.setArg(2, dev_c);
        kernel_gmem.setArg(3, static_cast<int>(n));
        kernel_gmem.setArg(4, static_cast<int>(m));
        auto thr = static_cast<size_t>(ceil(((double) n * n) / block_size) * block_size);
        queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange, cl::NDRange(thr),
                                   cl::NDRange(block_size));
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * n * n, &c[0]);
        ofstream out("output.txt");
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                out << c[i * n + j] << " ";
            }
            out << endl;
        }
    }
    catch (cl::Error const &e) {
        cout << endl << e.what() << " : " << e.err() << endl;
    }

    return 0;
}

