#include <iostream>
#include <random>

#include <mpi.h>
#include <unistd.h>

void fillArray(double* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = i + 1;
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
                break;
            default:
                std::cerr << "Usage: " << argv[0]
                     << " -n <num elements per process> -i <num iterations>" << std::endl;
                return -1;
        }
    }

    double globalSum = 0;
    double localSum = 0;
    double* localElements = (double*) malloc(sizeof(double)*lengthPerProcess);

    fillArray(localElements, lengthPerProcess);

    for (int i = 0; i < iterations; i++)
    {
        globalSum = 0;
        localSum = 0;
        for (int j = 0; j < lengthPerProcess; j++)
        {
            localSum += localElements[j];
        }
        MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    std::cout << "Rank " << rank << " local sum: " << localSum << std::endl;
    std::cout << "Rank " << rank << " global sum: " << globalSum << std::endl;
    free(localElements);
    MPI_Finalize();
    return 0;
}
