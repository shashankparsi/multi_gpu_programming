# multi_gpu_programming
stream based multi gpu programming
compilation:
hipcc vec_add.cpp -o vec_add

execution:
./vec_add

output on MI300A:
No.of GPU's available are: 4

Time elapsed on vector addition on GPU[0]:1.813379 ms.


Time elapsed on vector addition on GPU[1]:1.813379 ms.


Time elapsed on vector addition on GPU[2]:1.813379 ms.


Time elapsed on vector addition on GPU[3]:1.813379 ms.
