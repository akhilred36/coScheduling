#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

// -----------------------------------------------------------------------------
// get_rate_limit: measure max messages/sec between sender and receiver
//    All ranks must call this collectively.
//    Only the sender rank (world_rank == sender) receives the rate value.
// -----------------------------------------------------------------------------

void get_rate_limit(int sender, int receiver, double *rate_out)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Build a group containing exactly the two ranks.
    MPI_Group world_group, pair_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int ranks[2] = {sender, receiver};
    MPI_Group_incl(world_group, 2, ranks, &pair_group);

    // Create a sub-communicator for the pair (collective on MPI_COMM_WORLD).
    MPI_Comm subcomm;
    MPI_Comm_create(MPI_COMM_WORLD, pair_group, &subcomm);
    MPI_Group_free(&pair_group);
    MPI_Group_free(&world_group);

    // Only the two involved processes have a valid communicator.
    if (subcomm != MPI_COMM_NULL) {
        int sub_rank;
        MPI_Comm_rank(subcomm, &sub_rank);   // 0 = sender, 1 = receiver

        // Test a range of small message sizes.
        int sizes[] = {0, 1, 4, 16, 64, 256, 1024, 4096, 16384};
        const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        const int M = 10000;                // messages per timed burst
        double max_rate = 0.0;

        for (int s = 0; s < num_sizes; ++s) {
            int size = sizes[s];
            std::vector<char> buf(std::max(1, size));
            double local_max = 0.0;

            for (int iter = 0; iter < 100; ++iter) {
                // Optional debug output – only once per pair/size
                if (iter == 0 && (world_rank == sender || world_rank == receiver)) {
                    std::cout << "Testing: ranks " << sender << " <-> " << receiver
                              << ", size=" << size << std::endl;
                }

                MPI_Barrier(subcomm);
                double t_start = MPI_Wtime();

                if (sub_rank == 0) {                      // sender
                    for (int i = 0; i < M; ++i)
                        MPI_Send(buf.data(), size, MPI_BYTE, 1, 0, subcomm);
                } else {                                  // receiver
                    for (int i = 0; i < M; ++i)
                        MPI_Recv(buf.data(), size, MPI_BYTE, 0, 0, subcomm, MPI_STATUS_IGNORE);
                }

                MPI_Barrier(subcomm);
                double t_end = MPI_Wtime();
                double rate = M / (t_end - t_start);
                if (rate > local_max) local_max = rate;
            }

            if (local_max > max_rate) max_rate = local_max;
        }

        // Output the rate on the sender process only.
        if (world_rank == sender)
            *rate_out = max_rate;

        // Free the sub‑communicator ONLY when it is valid
    MPI_Comm_free(&subcomm);
    }
    // For processes with subcomm == MPI_COMM_NULL: nothing to free
}

// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size < 2) {
        if (world_rank == 0)
            std::cerr << "Error: at least 2 processes are required.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parse "-o <filename>"
    std::string out_filename;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_filename = argv[i + 1];
            break;
        }
    }
    if (out_filename.empty()) {
        if (world_rank == 0)
            std::cerr << "Usage: " << argv[0] << " -o <output.csv>\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Each rank stores its own outgoing pair results:
    //   recvs[i] = destination rank,   rates[i] = messages/sec
    std::vector<int> recvs;
    std::vector<double> rates;

    // --- Main pair loop ---
    // Ordered pairs (sender, receiver) with sender != receiver.
    for (int sender = 0; sender < world_size; ++sender) {
        for (int receiver = 0; receiver < world_size; ++receiver) {
            if (sender == receiver) continue;

            // Barrier on MPI_COMM_WORLD to isolate each test.
            MPI_Barrier(MPI_COMM_WORLD);

            double rate = 0.0;
            get_rate_limit(sender, receiver, &rate);

            // The sender stores the result.
            if (world_rank == sender) {
                recvs.push_back(receiver);
                rates.push_back(rate);
            }

            MPI_Barrier(MPI_COMM_WORLD);   // finish isolation
        }
    }

    // --- Gather all results at rank 0 ---
    MPI_Barrier(MPI_COMM_WORLD);   // ensure everyone reached this point

    if (world_rank == 0) {
        // Open CSV file and write header.
        std::ofstream csv(out_filename);
        csv << "Rank A,Rank B,Rate Limit (messages/sec)\n";

        // 1. Write rank 0's own results.
        for (size_t i = 0; i < recvs.size(); ++i) {
            csv << 0 << "," << recvs[i] << "," << rates[i] << "\n";
        }

        // 2. Receive results from every other rank.
        for (int r = 1; r < world_size; ++r) {
            int count = world_size - 1;   // each rank sends N-1 pairs
            std::vector<int> r_recvs(count);
            std::vector<double> r_rates(count);

            MPI_Recv(r_recvs.data(), count, MPI_INT, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(r_rates.data(), count, MPI_DOUBLE, r, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < count; ++i) {
                csv << r << "," << r_recvs[i] << "," << r_rates[i] << "\n";
            }
        }
        csv.close();
        std::cout << "Results written to " << out_filename << std::endl;
    } else {
        // Send my own (receiver, rate) pairs to rank 0.
        int count = world_size - 1;
        MPI_Send(recvs.data(), count, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(rates.data(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}