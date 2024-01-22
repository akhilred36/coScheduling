#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#define ITERS 1000

using namespace std;

void fillArray(double* array, int size)
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
    vector<double*> sendBuffers;
    vector<double*> recvBuffers;

    for (int i = 0; i < numProcs; i++)
    {
        double* sb = new double[msgSize * 1000];
        double* rb = new double[msgSize * 1000];
        sendBuffers.push_back(sb);
        recvBuffers.push_back(rb);
        fillArray(sendBuffers.at(i), msgSize * 1000);
    }

    MPI_Request* sendReqs = new MPI_Request[numProcs];
    MPI_Request* recvReqs = new MPI_Request[numProcs];
    int count = 0;
    while (1)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int j = 0; j < numProcs; j++)
        {
            MPI_Isend(sendBuffers.at(j), msgSize * 1000, MPI_DOUBLE, j, 0,
                      MPI_COMM_WORLD, &(sendReqs[j]));
            MPI_Irecv(recvBuffers.at(j), msgSize * 1000, MPI_DOUBLE, j, 0,
                      MPI_COMM_WORLD, &(recvReqs[j]));
        }
        MPI_Waitall(numProcs, sendReqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(numProcs, recvReqs, MPI_STATUSES_IGNORE);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
        if (duration < waitTime)
        {
            auto remainingTime = waitTime - duration;
            this_thread::sleep_for(chrono::milliseconds(remainingTime));
        }
        count++;
    }

    for (auto pointer : sendBuffers)
    {
        delete pointer;
    }
    for (auto pointer : recvBuffers)
    {
        delete pointer;
    }
    delete sendReqs;
    delete recvReqs;
    MPI_Finalize();
    return 0;
}
