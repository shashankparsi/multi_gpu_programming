#include<vector>
#include<hip/hip_runtime.h>
#include<stdio.h>
#include<stdlib.h>
__global__ void vec_add(float *d_a,float *d_b,float *d_c,int n)
{
        int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if(i>=n)
                return;
        d_c[i] = d_a[i] + d_b[i];
}

int main()
{
        hipError_t err;
        int device_count;
        hipDeviceProp_t props;
        int deviceId;
        int devicenum=-1;
        hipGetDeviceProperties(&props,devicenum);
        hipGetDevice(&devicenum);
        //printf("info: running on device #%d %s\n", devicenum, props.name);
        hipGetDeviceCount(&device_count);
        printf("No.of GPU's available are: %d\n",device_count);
        uint64_t length=1024000000;
        int block_size = 256;
        int num_gpus=device_count;
        int length_per_gpu = length/num_gpus;
        int num_blocks_per_gpu = (length_per_gpu-1)/block_size+1;

        std::vector<float>a,b,c;
        for(int i=0;i<length;i++) {
        a.push_back((float)rand()/(float)RAND_MAX);
        b.push_back((float)rand()/(float)RAND_MAX);
        }

        std::vector<float*>d_a,d_b,d_c;
        std::vector<hipStream_t>streams;

        float gpu_elapsed_time_ms;
        hipEvent_t start,stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        for(int i=0;i<num_gpus;i++) //create stream and allocate memory
        {
                hipSetDevice(i);
                hipStream_t stream;
                hipStreamCreate(&stream);
                streams.push_back(stream);

                float *a_ptr,*b_ptr,*c_ptr;
                hipMalloc((void**)&a_ptr,length_per_gpu*sizeof(float));
                hipMalloc((void**)&b_ptr,length_per_gpu*sizeof(float));
                hipMalloc((void**)&c_ptr,length_per_gpu*sizeof(float));
                d_a.push_back(a_ptr);
                d_b.push_back(b_ptr);
                d_c.push_back(c_ptr);
               hipMemcpyAsync(a_ptr,a.data()+i*length_per_gpu,length_per_gpu*sizeof(float),hipMemcpyHostToDevice);
                hipMemcpyAsync(b_ptr,b.data()+i*length_per_gpu,length_per_gpu*sizeof(float),hipMemcpyHostToDevice);
        }
        for(int i=0;i<num_gpus;i++)
        {
                hipSetDevice(i);
                hipEventRecord(start,0);
                vec_add<<<num_blocks_per_gpu,block_size,0,streams[i]>>>(d_a[i],d_b[i],d_c[i],length_per_gpu);
                hipEventRecord(stop,0);
                hipEventSynchronize(stop);
                hipEventElapsedTime(&gpu_elapsed_time_ms,start,stop);
                printf("\nTime elapsed on vector addition on GPU[%d]:%f ms.\n\n",i,gpu_elapsed_time_ms);
        }
        for(int i=0;i<num_gpus;i++)
        {
                hipStreamSynchronize(streams[i]);
        }
        c.resize(length);
        for(int i=0;i<num_gpus;i++)
        {
                hipSetDevice(i);
                hipMemcpyAsync(c.data()+i*length_per_gpu,d_c[i],length_per_gpu*sizeof(float),hipMemcpyDeviceToHost);
        }
      for(int i=0;i<num_gpus;i++)
        {
                hipFree(d_a[i]);
                hipFree(d_b[i]);
                hipFree(d_c[i]);
        }
        return 0;
}
   
