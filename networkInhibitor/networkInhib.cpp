#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define ITERS 1000

using namespace std;

static uint64_t seed = 12345;  // default seed
int mpiRank;
int numProcs;
long waitTime;
long msgSize;
float commSparsity;
char commMode;
unsigned long iters;
vector<char*> sendBuffers;
vector<char*> recvBuffers;
MPI_Request* sendReqs;
MPI_Request* recvReqs;
vector<vector<int>>
    deterministicTargets;  // For deterministic mode: stores which processes
                           // each process sends to

void srand_lcg(uint64_t s) { seed = s; }

// Returns a pseudo-random number in [0, 2^31 - 1]
uint32_t rand_lcg(void)
{
  seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(seed >> 33);
}

void fillArray(char* array, int size)
{
  for (int i = 0; i < size; i++)
  {
    array[i] = static_cast<char>(rand_lcg() % 256);
  }
}

void generateDeterministicTargets()
{
  int targetsPerProcess = round(commSparsity * numProcs);
  deterministicTargets.resize(numProcs);

  for (int sender = 0; sender < numProcs; sender++)
  {
    // Use hash-based selection for deterministic targets
    uint64_t hashSeed = seed + sender;
    srand_lcg(hashSeed);

    vector<int> available(numProcs);
    for (int i = 0; i < numProcs; i++) available[i] = i;

    for (int t = 0; t < targetsPerProcess; t++)
    {
      int idx = rand_lcg() % available.size();
      deterministicTargets[sender].push_back(available[idx]);
      available[idx] = available.back();
      available.pop_back();
    }
  }
}

void inhib()
{
  int count = 0;
  // while (1)
  for (unsigned long i = 0; i < iters; i++)
  {
    // Generate send targets based on mode
    vector<vector<int>> sendTargets;
    if (commMode == 'd')
    {
      sendTargets = deterministicTargets;
    }
    else
    {
      // Random mode: generate new pattern each iteration
      int targetsPerProcess = round(commSparsity * numProcs);
      sendTargets.resize(numProcs);
      for (int sender = 0; sender < numProcs; sender++)
      {
        for (int t = 0; t < targetsPerProcess; t++)
        {
          int dest = rand_lcg() % numProcs;
          // Avoid duplicates and self
          bool found = false;
          for (int existing : sendTargets[sender])
          {
            if (existing == dest)
            {
              found = true;
              break;
            }
          }
          if (!found && dest != sender)
          {
            sendTargets[sender].push_back(dest);
          }
          else
          {
            t--;  // Retry
          }
        }
      }
    }

    // Count total sends for this iteration
    int totalSends = 0;
    for (const auto& targets : sendTargets)
    {
      totalSends += targets.size();
    }

    // Resize requests if needed
    if (totalSends > numProcs)
    {
      delete[] sendReqs;
      delete[] recvReqs;
      sendReqs = new MPI_Request[totalSends];
      recvReqs = new MPI_Request[totalSends];
    }

    int reqIdx = 0;
    for (int j = 0; j < numProcs; j++)
    {
      for (int dest : sendTargets[j])
      {
        MPI_Isend(sendBuffers.at(j), msgSize, MPI_BYTE, dest, 0, MPI_COMM_WORLD,
                  &(sendReqs[reqIdx]));
        reqIdx++;
      }
    }

    // Reset reqIdx for receives
    reqIdx = 0;
    for (int j = 0; j < numProcs; j++)
    {
      for (int src : sendTargets[j])
      {
        MPI_Irecv(recvBuffers.at(j), msgSize, MPI_BYTE, src, 0, MPI_COMM_WORLD,
                  &(recvReqs[reqIdx]));
        reqIdx++;
      }
    }

    MPI_Waitall(totalSends, sendReqs, MPI_STATUSES_IGNORE);
    MPI_Waitall(totalSends, recvReqs, MPI_STATUSES_IGNORE);
    this_thread::sleep_for(chrono::microseconds(waitTime));
    count++;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  long waitTime;
  unsigned long msgSize;  // message size in bytes
  float commSparsity;
  char commMode;
  waitTime = 100;
  msgSize = 1000;  // default 1000 bytes
  commSparsity = 1.0;
  commMode = 'd';  // -d: Deterministic, -r: Random
  iters = 1000;

  int flags, opt;
  while ((opt = getopt(argc, argv, "m:w:s:c:i:")) != -1)
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
        if ((commSparsity >= 0) && (commSparsity <= 1)) break;
      case 'c':
        commMode = optarg[0];
        if (commMode != 'd' && commMode != 'r')
        {
          cerr << "Error: commMode must be 'd' (deterministic) or 'r' (random)"
               << endl;
          return -1;
        }
        break;
      case 'i':
        iters = (unsigned long)atol(optarg);
        break;
      default:
        cerr << "Usage: " << argv[0]
             << " -m <message size (bytes)> -w <wait time (ms)> -s "
                "<communication "
                "sparsity (0<=s<=1)> -c <mode: d|r>"
             << endl;
        return -1;
    }
  }
  if (mpiRank == 0)
  {
    cout << "------------------------------------------------------------"
         << endl;
    cout << "Using: \n\tMessage Size (bytes): " << msgSize
         << ". Wait time (us): " << waitTime
         << "\n\tNumber of Processes: " << numProcs
         << "\n\tCommunication Sparsity: " << commSparsity
         << "\n\tCommunication Mode: "
         << (commMode == 'd' ? "Deterministic" : "Random") << endl;
    cout << "------------------------------------------------------------"
         << endl;
  }

  // Generate deterministic targets if in deterministic mode
  if (commMode == 'd')
  {
    generateDeterministicTargets();
  }

  for (int i = 0; i < numProcs; i++)
  {
    char* sb = new char[msgSize];
    char* rb = new char[msgSize];
    sendBuffers.push_back(sb);
    recvBuffers.push_back(rb);
    fillArray(sendBuffers.at(i), msgSize);
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
