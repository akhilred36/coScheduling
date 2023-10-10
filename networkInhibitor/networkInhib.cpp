#include <mpi.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#define ITERS 1000

using namespace std;

void fillArray(shared_ptr<double[]> array, int size)
{
    double lower_bound = 0;
    double upper_bound = 1000000;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;
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

    long waitTime, msgSize;
    waitTime = 100;
    msgSize = 1000;

    int flags, opt;
    while ((opt = getopt(argc, argv, "m:w:")) != -1)
    {
        switch (opt)
        {
            case 'm':
                msgSize = atol(optarg);
                break;
            case 'w':
                waitTime = atol(optarg);
                break;
            default:
                cerr << "Usage: " << argv[0]
                     << " -m <message size (kb)> -w <wait time (ms)>" << endl;
                return -1;
        }
    }
    if (rank == 0)
    {
        cout << "------------------------------------------------------------"
             << endl;
        cout << "Using: \n\tMessage Size (kb): " << msgSize
             << ". Wait time (ms): " << waitTime
             << "\n\tNumber of Processes: " << numProcs << endl;
        cout << "------------------------------------------------------------"
             << endl;
    }
    vector<shared_ptr<double[]>> sendBuffers;
    vector<shared_ptr<double[]>> recvBuffers;

    for (int i = 0; i < numProcs; i++)
    {
        shared_ptr<double[]> sb(new double[msgSize * 1000]);
        shared_ptr<double[]> rb(new double[msgSize * 1000]);
        sendBuffers.push_back(sb);
        recvBuffers.push_back(rb);
        fillArray(sendBuffers.at(i), msgSize * 1000);
    }

    shared_ptr<MPI_Request[]> sendReqs(new MPI_Request[numProcs]);
    shared_ptr<MPI_Request[]> recvReqs(new MPI_Request[numProcs]);

    for (int i = 0; i < ITERS; i++)
    {
        for (int j = 0; j < numProcs; j++)
        {
            MPI_Isend(&(sendBuffers.at(j)), msgSize * 1000, MPI_DOUBLE, j, 0,
                      MPI_COMM_WORLD, &(sendReqs[j]));
            MPI_Irecv(&(recvBuffers.at(j)), msgSize * 1000, MPI_DOUBLE, j, 0,
                      MPI_COMM_WORLD, &(recvReqs[j]));
        }
        MPI_Waitall(numProcs, &(sendReqs[0]), MPI_STATUSES_IGNORE);
        MPI_Waitall(numProcs, &(recvReqs[0]), MPI_STATUSES_IGNORE);
        // Background network soaking job
        // Run app
        // Kill network soaking job when app finishes
    }
    MPI_Finalize();
    return 0;
}