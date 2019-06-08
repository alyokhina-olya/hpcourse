#define SWAP(a, b) {__local int * tmp=a; a=b; b=tmp;}

__kernel void prefix_sum(__global double *input, __global double *output, __local double *arr, __local double *brr,
                                 int size) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint local_size = get_local_size(0);
    if (global_id < size)
        arr[local_id] = brr[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = 1; s < local_size; s <<= 1) {
        if (local_id <= (s - 1)) {
            brr[local_id] = arr[local_id];
        }
        else {
            brr[local_id] = arr[local_id] + arr[local_id - s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(arr, brr);
    }
    if (global_id < size)
        output[global_id] = arr[local_id];
}


__kernel void copy(__global double *input, __global double *output, int input_size, int output_size) {
    uint global_id = get_global_id(0);
    uint local_size = get_local_size(0);
    uint ind = global_id / local_size + 1;
    if (ind < output_size && global_id < input_size &&  1 + global_id == ind * local_size)
        output[ind] = input[global_id];
}

__kernel void add(__global double *part_input, __global double *input, __global double *output, int size) {
    uint global_id = get_global_id(0);
    uint local_size = get_local_size(0);
    if (global_id < size)
        output[global_id] = input[global_id] + part_input[global_id / local_size];
}
