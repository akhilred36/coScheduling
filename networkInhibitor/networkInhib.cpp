#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define ITERS 1000

using namespace std;


static uint64_t seed = 12345; // default seed
int mpiRank;
int numProcs;
long waitTime;
long msgSize;
float commSparsity;
char commMode;
unsigned long iters;
vector<double*> sendBuffers;
vector<double*> recvBuffers;
MPI_Request* sendReqs;
MPI_Request* recvReqs;

void srand_lcg(uint64_t s) {
    seed = s;
}

// Returns a pseudo-random number in [0, 2^31 - 1]
uint32_t rand_lcg(void) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(seed >> 33);
}

void fillArray(double* array, int size)
{
    double lower_bound = 0;
    double upper_bound = 1000000;
    for (int i = 0; i < size; i++)
    {
        array[i] = lower_bound + (rand_lcg() / (double)(1ULL << 31)) * (upper_bound - lower_bound);
    }
}

void inhib()
{
    int count = 0;
    // while (1)
    for (unsigned long i = 0; i < iters; i++)
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
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    long waitTime, msgSize;
    float commSparsity;
    char commMode;
    waitTime = 100;
    msgSize = 1000;
    commSparsity = 1.0;
    commMode = 'd'; // -d: Deterministic, -r: Random
    iters = 1000;

    int flags, opt;
    while ((opt = getopt(argc, argv, "m:w:s:i:")) != -1)
    {
        switch (opt)
        {
            case 'm':
                msgSize = atol(optarg);
                break;
            case 'w':
                waitTime = atol(optarg);
                break;
            case 's':
                commSparsity = atof(optarg);
                if ((commSparsity >= 0) && (commSparsity <= 1))
                    break;
            case 'i':
                iters = (unsigned long) atol(optarg);
                break;
            default:
                cerr << "Usage: " << argv[0]
                     << " -m <message size (kb)> -w <wait time (ms)> -s <communication sparsity (0<=s<=1)>" << endl;
                return -1;
        }
    }
    if (mpiRank == 0)
    {
        cout << "------------------------------------------------------------"
             << endl;
        cout << "Using: \n\tMessage Size (kb): " << msgSize
             << ". Wait time (ms): " << waitTime
             << "\n\tNumber of Processes: " << numProcs 
             << "\n\tCommunication Sparsity: " << commSparsity << endl;
        cout << "------------------------------------------------------------"
             << endl;
    }

    for (int i = 0; i < numProcs; i++)
    {
        double* sb = new double[msgSize * 1000];
        double* rb = new double[msgSize * 1000];
        sendBuffers.push_back(sb);
        recvBuffers.push_back(rb);
        fillArray(sendBuffers.at(i), msgSize * 1000);
    }

    sendReqs = new MPI_Request[numProcs];
    recvReqs = new MPI_Request[numProcs];
    
    inhib();

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
