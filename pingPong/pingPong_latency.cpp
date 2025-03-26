#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <cmath>
#include <numeric>

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

// Perform linear regression to extract tstartup
pair<double, double> linearFit(const vector<double>& sizes, const vector<double>& times)
{
    int N = sizes.size();
    double sumX = accumulate(sizes.begin(), sizes.end(), 0.0);
    double sumY = accumulate(times.begin(), times.end(), 0.0);
    double sumXY = 0.0;
    double sumX2 = 0.0;

    for (int i = 0; i < N; i++)
    {
        sumXY += sizes[i] * times[i];
        sumX2 += sizes[i] * sizes[i];
    }

    double slope = (N * sumXY - sumX * sumY) / (N * sumX2 - sumX * sumX);
    double intercept = (sumY - slope * sumX) / N;

    return {intercept, slope}; // intercept = tstartup, slope = 1/bandwidth
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs != 2)
    {
        if (rank == 0)
            cerr << "Error: Number of processes need to be 2" << endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int n = 12;
    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1)
    {
        switch (opt)
        {
            case 'n':
                n = atoi(optarg);
                break;
            default:
                cerr << "Usage: " << argv[0]
                     << " -n <n> (for a max message size of 2^n. Default = 12)>"
                     << endl;
                return -1;
        }
    }

    int max_size = pow(2, n);
    double* sendData = (double*)calloc(max_size, sizeof(double));
    double* recvData = (double*)calloc(max_size, sizeof(double));
    fillArray(sendData, max_size);

    vector<double> times;
    vector<double> sizes;

    for (int i = 0; i < n; i++)
    {
        int arraySize = pow(2, i);
        if (rank == 0)
        {
            int ready;
            MPI_Recv(&ready, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
        else
        {
            int ready = 1;
            MPI_Send(&ready, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        chrono::time_point<std::chrono::system_clock> start, end;
        vector<double> tempTimes;

        for (int j = 0; j < 10000; j++)
        {
            if (rank == 0)
            {
                start = chrono::system_clock::now();
                MPI_Recv(recvData, arraySize, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            else
                MPI_Send(sendData, arraySize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            if (rank == 1)
                MPI_Recv(recvData, arraySize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            else
                MPI_Send(sendData, arraySize, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);

            if (rank == 0)
            {
                end = chrono::system_clock::now();
                chrono::duration<double> elapsed_seconds = end - start;
                tempTimes.push_back(elapsed_seconds.count() / 2);
            }
        }

        if (rank == 0)
        {
            double total = accumulate(tempTimes.begin(), tempTimes.end(), 0.0);
            double avg_time = total / 10000;
            times.push_back(avg_time);
            sizes.push_back(arraySize);
            cout << "Message Size (2^" << i << "): " << avg_time << " seconds" << endl;
        }
    }

    if (rank == 0)
    {
        auto [tstartup, inv_bw] = linearFit(sizes, times);
        cout << "Estimated tstartup: " << tstartup << " seconds" << endl;
        cout << "Estimated Bandwidth: " << (1.0 / inv_bw) << " bytes/second" << endl;
    }

    free(sendData);
    free(recvData);
    MPI_Finalize();
    return 0;
}
