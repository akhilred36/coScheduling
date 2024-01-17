#include <iostream>
#include <random>

#include <mpi.h>
#include <unistd.h>

void fillArray(double* array, int size)
{
    double lower_bound = 0;
    double upper_bound = 1000000;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;
    for (int i = 0; i < size; i++)
    {
        array[i] = unif(re);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    long lengthPerProcess = 500;
    int iterations = 1000;

    int flags, opt;
    while ((opt = getopt(argc, argv, "n:i:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                lengthPerProcess = atol(optarg);
                break;
            case 'i':
                iterations = atol(optarg);
            default:
                std::cerr << "Usage: " << argv[0]
                     << " -n <num elements per process> -i <num iterations>" << std::endl;
                return -1;
        }
    }

    double globalSum;
    double* localElements = (double*) malloc(sizeof(double)*lengthPerProcess);

    fillArray(localElements, lengthPerProcess);

    for (int i = 0; i < iterations; i++)
    {
        MPI_Allreduce(localElements, &globalSum, lengthPerProcess, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    std::cout << "Global sum is: " << globalSum << std::endl;

    free(localElements);
    MPI_Finalize();
    return 0;
}
