#include <getopt.h>
#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class RateLimitTest {
 private:
  int world_size;
  int world_rank;
  std::vector<int> message_sizes;
  int messages_per_rank;
  int iterations;
  std::string output_file;

  void parseArguments(int argc, char** argv)
  {
    int opt;
    while ((opt = getopt(argc, argv, "o:")) != -1)
    {
      switch (opt)
      {
        case 'o':
          output_file = optarg;
          break;
        default:
          break;
      }
    }

    if (output_file.empty())
    {
      std::cerr << "Error: Output file must be specified with -o flag"
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // Generate unique data for each message to prevent compression/optimization
  void fillBuffer(char* buffer, int size, int rank, int iter, int dest)
  {
    // Fill with pattern that depends on rank, iteration, and destination
    // This prevents any compression or caching optimizations
    for (int i = 0; i < size; i++)
    {
      buffer[i] = (char)((rank + iter + dest + i) % 256);
    }
  }

  double runTest(int msg_size)
  {
    // Allocate send and receive buffers
    std::vector<char> send_buffer(msg_size);
    std::vector<char> recv_buffer(msg_size);

    // Create requests for all sends and receives
    std::vector<MPI_Request> send_requests(world_size - 1);
    std::vector<MPI_Request> recv_requests(world_size - 1);

    // Warm-up phase - run 10 iterations before timing
    if (world_rank == 0)
    {
      std::cout << "    Warming up..." << std::flush;
    }
    for (int iter = 0; iter < 10; ++iter)
    {
      int recv_count = 0;
      for (int dest = 0; dest < world_size; ++dest)
      {
        if (dest != world_rank)
        {
          MPI_Irecv(recv_buffer.data(), msg_size, MPI_CHAR, dest, 0,
                    MPI_COMM_WORLD, &recv_requests[recv_count]);
          recv_count++;
        }
      }

      int send_count = 0;
      for (int src = 0; src < world_size; ++src)
      {
        if (src != world_rank)
        {
          fillBuffer(send_buffer.data(), msg_size, world_rank, iter, src);
          MPI_Isend(send_buffer.data(), msg_size, MPI_CHAR, src, 0,
                    MPI_COMM_WORLD, &send_requests[send_count]);
          send_count++;
        }
      }

      if (send_count > 0)
      {
        MPI_Waitall(send_count, send_requests.data(), MPI_STATUSES_IGNORE);
      }
      if (recv_count > 0)
      {
        MPI_Waitall(recv_count, recv_requests.data(), MPI_STATUSES_IGNORE);
      }
    }
    if (world_rank == 0)
    {
      std::cout << " done." << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform the test for the specified number of iterations
    for (int iter = 0; iter < iterations; ++iter)
    {
      // Post all receives first
      int recv_count = 0;
      for (int dest = 0; dest < world_size; ++dest)
      {
        if (dest != world_rank)
        {
          MPI_Irecv(recv_buffer.data(), msg_size, MPI_CHAR, dest, 0,
                    MPI_COMM_WORLD, &recv_requests[recv_count]);
          recv_count++;
        }
      }

      // Post all sends
      int send_count = 0;
      for (int src = 0; src < world_size; ++src)
      {
        if (src != world_rank)
        {
          fillBuffer(send_buffer.data(), msg_size, world_rank, iter, src);
          MPI_Isend(send_buffer.data(), msg_size, MPI_CHAR, src, 0,
                    MPI_COMM_WORLD, &send_requests[send_count]);
          send_count++;
        }
      }

      // Wait for all sends and receives to complete
      if (send_count > 0)
      {
        MPI_Waitall(send_count, send_requests.data(), MPI_STATUSES_IGNORE);
      }
      if (recv_count > 0)
      {
        MPI_Waitall(recv_count, recv_requests.data(), MPI_STATUSES_IGNORE);
      }
    }

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    return elapsed.count();
  }

  void printProgress(int msg_size, int current_index, int total_sizes)
  {
    if (world_rank == 0)
    {
      std::time_t now = std::time(nullptr);
      std::tm* local_time = std::localtime(&now);
      char time_buffer[80];
      std::strftime(time_buffer, sizeof(time_buffer), "%H:%M:%S", local_time);

      std::cout << "\n[" << time_buffer << "] ";
      std::cout << "Testing message size: " << msg_size << " bytes ";
      std::cout << "(" << (current_index + 1) << "/" << total_sizes << ")"
                << std::endl;
      std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

 public:
  RateLimitTest(int argc, char** argv)
      : message_sizes({1, 4, 16, 64, 256, 1024}),
        messages_per_rank(100000),
        iterations(1000)
  {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    parseArguments(argc, argv);

    // Ensure we have at least 2 processes
    if (world_size < 2)
    {
      if (world_rank == 0)
      {
        std::cerr << "Error: Need at least 2 MPI processes for this test"
                  << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  ~RateLimitTest() { MPI_Finalize(); }

  void run()
  {
    if (world_rank == 0)
    {
      std::cout << "\n========================================" << std::endl;
      std::cout << "Global Message Rate Limit Test" << std::endl;
      std::cout << "========================================" << std::endl;
      std::cout << "Number of processes: " << world_size << std::endl;
      std::cout << "Messages per rank per iteration: " << messages_per_rank
                << std::endl;
      std::cout << "Iterations per message size: " << iterations << std::endl;
      std::cout << "Total messages per iteration: "
                << (world_size * (world_size - 1) * messages_per_rank)
                << std::endl;
      std::cout << "Total messages per test: "
                << (world_size * (world_size - 1) * messages_per_rank *
                    iterations)
                << std::endl;
      std::cout << "Message sizes to test: ";
      for (size_t i = 0; i < message_sizes.size(); ++i)
      {
        std::cout << message_sizes[i];
        if (i < message_sizes.size() - 1) std::cout << ", ";
      }
      std::cout << " bytes" << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

    // Find the best rate limit across all message sizes
    double best_rate = 0.0;
    int best_msg_size = 0;

    // Store results for all message sizes
    std::vector<std::pair<int, double>> results;

    for (size_t i = 0; i < message_sizes.size(); ++i)
    {
      int msg_size = message_sizes[i];

      // Print progress before starting the test
      printProgress(msg_size, i, message_sizes.size());

      // Synchronize all processes before each test
      MPI_Barrier(MPI_COMM_WORLD);

      // Run the test and measure time
      double elapsed_time = runTest(msg_size);

      // Calculate total messages sent
      long long total_messages = static_cast<long long>(world_size) *
                                 (world_size - 1) * messages_per_rank *
                                 iterations;

      // Calculate rate
      double rate = total_messages / elapsed_time;

      // Store result
      results.push_back({msg_size, rate});

      // Find the best rate (highest) across all message sizes
      if (rate > best_rate)
      {
        best_rate = rate;
        best_msg_size = msg_size;
      }

      if (world_rank == 0)
      {
        std::cout << "  ✓ Completed: " << msg_size << " bytes" << std::endl;
        std::cout << "    └─ Total messages: " << total_messages << std::endl;
        std::cout << "    └─ Elapsed time: " << std::fixed
                  << std::setprecision(3) << elapsed_time << " seconds"
                  << std::endl;
        std::cout << "    └─ Rate: " << std::fixed << std::setprecision(2)
                  << rate << " messages/sec" << std::endl;
        if (rate == best_rate)
        {
          std::cout << "    └─ ★ New best rate!" << std::endl;
        }
        std::cout << std::endl;
      }
    }

    // Output JSON file with the best rate
    if (world_rank == 0)
    {
      std::ofstream json_file(output_file);
      if (!json_file.is_open())
      {
        std::cerr << "Error: Could not open output file: " << output_file
                  << std::endl;
        return;
      }

      json_file << "{\n";
      json_file << "  \"Rate Limit (messages/sec)\": " << std::fixed
                << std::setprecision(2) << best_rate << "\n";
      json_file << "}\n";

      json_file.close();

      std::cout << "========================================" << std::endl;
      std::cout << "Results Summary:" << std::endl;
      for (const auto& result : results)
      {
        std::cout << "  " << result.first << " bytes: " << std::fixed
                  << std::setprecision(2) << result.second << " messages/sec"
                  << std::endl;
      }
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "Best rate: " << std::fixed << std::setprecision(2)
                << best_rate << " messages/sec" << std::endl;
      std::cout << "Best message size: " << best_msg_size << " bytes"
                << std::endl;
      std::cout << "Results written to: " << output_file << std::endl;
      std::cout << "========================================" << std::endl;
    }
  }
};

int main(int argc, char** argv)
{
  try
  {
    RateLimitTest test(argc, argv);
    test.run();
  } catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}