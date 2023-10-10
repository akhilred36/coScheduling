#include <iostream>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    long waitTime, msgSize;
    waitTime = 100;
    msgSize = 1000;

    int flags, opt;
    while ((opt = getopt(argc, argv, "m:w:")) != -1) {
        switch (opt) {
            case 'm':
                msgSize = atol(optarg);
                break;
            case 'w':
                waitTime = atol(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -m <message size (kb)> -w <wait time (ms)>" << std::endl;
                return -1;
        }
    }
    if (rank == 0)
    {
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "Using: \n\tMessage Size (kb): " << msgSize << ". Wait time (ms): " << waitTime <<
                     "\n\tNumber of Processes: " << numProcs << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
    }
    return 0;
}
