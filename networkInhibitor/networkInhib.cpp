#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#define ITERS 1000

using namespace std;

static uint64_t seed = 12345;
int mpiRank;
int numProcs;
long waitTime;
unsigned long msgSize;
float commSparsity;
char commMode;
unsigned long iters;
char* outputFile = nullptr;

// ---- New globals for runtime limit ----
double maxRuntimeSec = 0.0;
bool runtimeLimitActive = false;
unsigned long actualIters = 0;  // counts actual completed iterations
std::chrono::high_resolution_clock::time_point
    g_startTime;  // used both for limit and final measurement
// -------------------------------------

// Buffers and request arrays (unchanged)
char* sendBuffer;
char* recvBuffer;  // size: (numProcs - 1) * msgSize; indexed by recv slot
MPI_Request* sendReqs;
MPI_Request* recvReqs;

// Deterministic communication targets
vector<vector<int>> deterministicTargets;

void srand_lcg(uint64_t s) { seed = s; }

uint32_t rand_lcg(void)
{
  seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(seed >> 33);
}

void fillArray(char* array, unsigned long size)
{
  for (unsigned long i = 0; i < size; i++)
  {
    array[i] = static_cast<char>(rand_lcg() % 256);
  }
}

void generateDeterministicTargets()
{
  int targetsPerProcess = (int)round(commSparsity * (numProcs - 1));
  deterministicTargets.resize(numProcs);

  for (int sender = 0; sender < numProcs; sender++)
  {
    uint64_t hashSeed = seed + sender;
    srand_lcg(hashSeed);

    vector<int> available;
    for (int i = 0; i < numProcs; i++)
    {
      if (i != sender) available.push_back(i);
    }

    for (int t = 0; t < targetsPerProcess && !available.empty(); t++)
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
  for (unsigned long i = 0; iters == ULONG_MAX || i < iters; i++)
  {
    // --- Build send targets for this iteration ---
    vector<vector<int>> sendTargets;

    if (commMode == 'd')
    {
      sendTargets = deterministicTargets;
    }
    else
    {
      // Random mode: rank 0 generates the graph and broadcasts it
      int targetsPerProcess = (int)round(commSparsity * (numProcs - 1));
      sendTargets.resize(numProcs);

      vector<int> flatTable(numProcs * targetsPerProcess, -1);

      if (mpiRank == 0)
      {
        for (int sender = 0; sender < numProcs; sender++)
        {
          vector<int> available;
          for (int k = 0; k < numProcs; k++)
          {
            if (k != sender) available.push_back(k);
          }
          for (int t = 0; t < targetsPerProcess && !available.empty(); t++)
          {
            int idx = rand_lcg() % available.size();
            flatTable[sender * targetsPerProcess + t] = available[idx];
            available[idx] = available.back();
            available.pop_back();
          }
        }
      }

      MPI_Bcast(flatTable.data(), numProcs * targetsPerProcess, MPI_INT, 0,
                MPI_COMM_WORLD);

      for (int sender = 0; sender < numProcs; sender++)
      {
        for (int t = 0; t < targetsPerProcess; t++)
        {
          int dest = flatTable[sender * targetsPerProcess + t];
          if (dest != -1) sendTargets[sender].push_back(dest);
        }
      }
    }

    // --- Determine this rank's send destinations and receive sources ---
    const vector<int>& myDests = sendTargets[mpiRank];
    vector<int> mySources;
    for (int r = 0; r < numProcs; r++)
    {
      for (int dest : sendTargets[r])
      {
        if (dest == mpiRank)
        {
          mySources.push_back(r);
          break;
        }
      }
    }

    int numSends = (int)myDests.size();
    int numRecvs = (int)mySources.size();

    // Resize request arrays if needed
    if (numSends > 0)
    {
      delete[] sendReqs;
      sendReqs = new MPI_Request[numSends];
    }
    if (numRecvs > 0)
    {
      delete[] recvReqs;
      recvReqs = new MPI_Request[numRecvs];
    }

    // Post non-blocking sends
    for (int s = 0; s < numSends; s++)
    {
      MPI_Isend(sendBuffer, msgSize, MPI_BYTE, myDests[s], 0, MPI_COMM_WORLD,
                &sendReqs[s]);
    }

    // Post non-blocking receives
    for (int r = 0; r < numRecvs; r++)
    {
      MPI_Irecv(recvBuffer + r * msgSize, msgSize, MPI_BYTE, mySources[r], 0,
                MPI_COMM_WORLD, &recvReqs[r]);
    }

    if (numSends > 0) MPI_Waitall(numSends, sendReqs, MPI_STATUSES_IGNORE);
    if (numRecvs > 0) MPI_Waitall(numRecvs, recvReqs, MPI_STATUSES_IGNORE);

    this_thread::sleep_for(chrono::microseconds(waitTime));

    MPI_Barrier(MPI_COMM_WORLD);

    // Increment iteration counter *after* the iteration completes
    ++actualIters;

    // Runtime limit check (if active)
    if (runtimeLimitActive)
    {
      auto now = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = now - g_startTime;
      if (elapsed.count() >= maxRuntimeSec)
      {
        break;  // exit the loop cleanly
      }
    }
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  waitTime = 100;
  msgSize = 1000;
  commSparsity = 1.0f;
  commMode = 'd';
  iters = ITERS;

  // Flags for mutual exclusion
  bool itersSet = false;
  bool runtimeSet = false;

  int opt;
  while ((opt = getopt(argc, argv, "m:w:s:c:i:o:r:")) != -1)
  {
    switch (opt)
    {
      case 'm':
        msgSize = (unsigned long)atol(optarg);
        break;
      case 'w':
        waitTime = atol(optarg);
        break;
      case 's':
        commSparsity = atof(optarg);
        if (commSparsity < 0.0f || commSparsity > 1.0f)
        {
          if (mpiRank == 0) cerr << "Error: sparsity must be in [0, 1]" << endl;
          MPI_Finalize();
          return -1;
        }
        break;
      case 'c':
        commMode = optarg[0];
        if (commMode != 'd' && commMode != 'r')
        {
          if (mpiRank == 0)
            cerr
                << "Error: commMode must be 'd' (deterministic) or 'r' (random)"
                << endl;
          MPI_Finalize();
          return -1;
        }
        break;
      case 'i':
        itersSet = true;
        {
          long raw = atol(optarg);
          iters = (raw == -1) ? ULONG_MAX : (unsigned long)raw;
        }
        break;
      case 'r':
        runtimeSet = true;
        maxRuntimeSec = atof(optarg);
        if (maxRuntimeSec <= 0.0)
        {
          if (mpiRank == 0)
            cerr << "Error: runtime must be > 0 seconds" << endl;
          MPI_Finalize();
          return -1;
        }
        break;
      case 'o':
        outputFile = optarg;
        break;
      default:
        if (mpiRank == 0)
          cerr << "Usage: " << argv[0]
               << " -m <message size (bytes)> -w <wait time (us)> -s "
                  "<communication sparsity (0<=s<=1)> -c <mode: d|r> "
                  "-i <iterations (-1 for infinite)> -r <max runtime (s)> "
                  "-o <output json file>\n"
               << "Note: -i and -r are mutually exclusive.\n";
        MPI_Finalize();
        return -1;
    }
  }

  // Enforce mutual exclusion between -i and -r
  if (itersSet && runtimeSet)
  {
    if (mpiRank == 0)
      cerr << "Error: -i and -r cannot be used together." << endl;
    MPI_Finalize();
    return -1;
  }

  // If runtime limit is active, set iterations to "infinite" and let the time
  // check stop the loop
  if (runtimeSet)
  {
    runtimeLimitActive = true;
    iters = ULONG_MAX;
  }

  if (mpiRank == 0)
  {
    cout << "------------------------------------------------------------\n";
    cout << "Using:"
         << "\n\tMessage Size (bytes): " << msgSize
         << "\n\tWait time (us): " << waitTime
         << "\n\tNumber of Processes: " << numProcs
         << "\n\tCommunication Sparsity: " << commSparsity
         << "\n\tCommunication Mode: "
         << (commMode == 'd' ? "Deterministic" : "Random")
         << "\n\tIterations: " << (iters == ULONG_MAX ? -1L : (long)iters);
    if (runtimeLimitActive) cout << "\n\tMax Runtime (s): " << maxRuntimeSec;
    cout << "\n";
    cout << "------------------------------------------------------------\n";
  }

  if (commMode == 'd')
  {
    generateDeterministicTargets();
  }

  sendBuffer = new char[msgSize];
  recvBuffer = new char[msgSize * (numProcs - 1)];
  fillArray(sendBuffer, msgSize);

  sendReqs = new MPI_Request[numProcs];
  recvReqs = new MPI_Request[numProcs];

  // Start the clock (used for runtime limit and final timing)
  g_startTime = std::chrono::high_resolution_clock::now();
  actualIters = 0;  // reset counter

  inhib();

  auto endTime = std::chrono::high_resolution_clock::now();

  delete[] sendBuffer;
  delete[] recvBuffer;
  delete[] sendReqs;
  delete[] recvReqs;

  if (mpiRank == 0)
  {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        endTime - g_startTime)
                        .count();
    double timeSeconds = duration / 1e6;

    int targetsPerProcess = (int)round(commSparsity * (numProcs - 1));
    double totalBytesPerIter =
        (double)numProcs * (double)targetsPerProcess * msgSize * 2.0;

    // Use actual completed iterations for bandwidth calculation
    double totalIterations = (double)actualIters;
    double totalBytes = totalBytesPerIter * totalIterations;

    double effectiveBandwidth =
        (timeSeconds > 0.0) ? (totalBytes / timeSeconds) / (1e9) : 0.0;

    cout << "------------------------------------------------------------\n";
    cout << "Results:\n";
    cout << "\tCompleted iterations: " << actualIters << "\n";
    cout << "\tTime (us): " << duration
         << "\n\tEffective Bandwidth (GB/s): " << std::fixed
         << std::setprecision(3) << effectiveBandwidth << "\n";
    cout << "------------------------------------------------------------\n";

    if (outputFile != nullptr)
    {
      std::ofstream outFile(outputFile);
      if (outFile.is_open())
      {
        long iterPrint = (iters == ULONG_MAX) ? -1L : (long)iters;

        outFile << "{\n";
        outFile << "  \"numRanks\": " << numProcs << ",\n";
        outFile << "  \"msgSize\": " << msgSize << ",\n";
        outFile << "  \"waitTime\": " << waitTime << ",\n";
        outFile << "  \"commSparsity\": " << commSparsity << ",\n";
        outFile << "  \"commMode\": \"" << commMode << "\",\n";
        outFile << "  \"numIterations\": " << iterPrint << ",\n";
        outFile << "  \"maxRuntimeSec\": "
                << (runtimeLimitActive ? maxRuntimeSec : 0.0) << ",\n";
        outFile << "  \"runtimeLimitUsed\": "
                << (runtimeLimitActive ? "true" : "false") << ",\n";
        outFile << "  \"completedIterations\": " << actualIters << ",\n";
        outFile << "  \"effectiveBandwidthGBps\": " << std::fixed
                << std::setprecision(3) << effectiveBandwidth << "\n";
        outFile << "}\n";
        outFile.close();
      }
    }
  }

  MPI_Finalize();
  return 0;
}