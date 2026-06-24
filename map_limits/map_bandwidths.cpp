#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>

void print_progress(int rank, int sender, int receiver, size_t msg_size, int run) {
    double mb = msg_size / (1024.0 * 1024.0);
    std::cout << "Rank " << rank << ": Pair (" << sender << "," << receiver << ") "
              << "Msg=" << mb << "MB" << std::flush;
}

double compute_bandwidth(int sender, int receiver, size_t message_size, int num_runs = 1) {
    MPI_Comm subcomm = MPI_COMM_NULL;
    int ranks[2] = {sender, receiver};
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group new_group;
    MPI_Group_incl(world_group, 2, ranks, &new_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, new_group, 0, &subcomm);
    MPI_Group_free(&world_group);
    MPI_Group_free(&new_group);

    int comm_rank = -1;
    if (subcomm != MPI_COMM_NULL) {
        MPI_Comm_rank(subcomm, &comm_rank);
    }

    double total_time = 0.0;
    const int num_iterations = 100;

    if (comm_rank == 0) {
        std::vector<char> send_buffer(message_size, 1);
        std::vector<char> recv_buffer(message_size, 0);

        for (int i = 0; i < num_iterations; ++i) {
            MPI_Barrier(subcomm);
            auto start = std::chrono::high_resolution_clock::now();
            MPI_Send(send_buffer.data(), message_size, MPI_BYTE, 1, 0, subcomm);
            MPI_Recv(recv_buffer.data(), message_size, MPI_BYTE, 1, 0, subcomm, MPI_STATUS_IGNORE);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double>(end - start).count();
        }

        double bandwidth_mbps = (2.0 * message_size * num_iterations) / (total_time * 1024.0 * 1024.0);

        return bandwidth_mbps;
    } else if (comm_rank == 1) {
        std::vector<char> recv_buffer(message_size);
        std::vector<char> send_buffer(message_size, 0);

        for (int i = 0; i < num_iterations; ++i) {
            MPI_Barrier(subcomm);
            MPI_Recv(recv_buffer.data(), message_size, MPI_BYTE, 0, 0, subcomm, MPI_STATUS_IGNORE);
            MPI_Send(send_buffer.data(), message_size, MPI_BYTE, 0, 0, subcomm);
        }

        return 0.0;
    }

    if (subcomm != MPI_COMM_NULL) {
        MPI_Comm_free(&subcomm);
    }
    return 0.0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (num_ranks < 2) {
        if (rank == 0) {
            std::cerr << "Error: At least 2 ranks are required (N >= 2)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string output_file = "bandwidth_results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    std::vector<size_t> message_sizes;
    message_sizes.reserve(31);
    for (int i = 0; i <= 30; ++i) {
        message_sizes.push_back(1ULL << i);
    }

    std::vector<double> bandwidth_results;
    bandwidth_results.reserve(num_ranks * (num_ranks - 1));

   int total_pairs = num_ranks * (num_ranks - 1);
    int pair_index = 0;

   for (int sender = 0; sender < num_ranks; ++sender) {
        for (int receiver = 0; receiver < num_ranks; ++receiver) {
            if (sender != receiver) {
                if (rank == sender) {
                    std::cout << std::endl << "Testing Pair (" << sender << "," << receiver << "):" << std::flush;
                }
                double max_bw = 0.0;
                for (size_t msg_size : message_sizes) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    double bw = compute_bandwidth(sender, receiver, msg_size);
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (bw > max_bw) {
                        max_bw = bw;
                    }
                    if (rank == sender) {
                        print_progress(rank, sender, receiver, msg_size, 0);
                        std::cout << std::endl << std::flush;
                    }
                }
                if (rank == sender) {
                    bandwidth_results.push_back(max_bw);
                }
                pair_index++;
            }
        }
    }

    std::vector<double> gathered_results(num_ranks * (num_ranks - 1));
    int recvcounts[num_ranks];
    for (int i = 0; i < num_ranks; ++i) {
        recvcounts[i] = num_ranks - 1;
    }

    MPI_Gather(bandwidth_results.data(), bandwidth_results.size(), MPI_DOUBLE,
               gathered_results.data(), recvcounts[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream out(output_file);
        out << "Rank A,Rank B,Bandwidth (MB/s)" << std::endl;

        int idx = 0;
        for (int sender = 0; sender < num_ranks; ++sender) {
            for (int receiver = 0; receiver < num_ranks; ++receiver) {
                if (sender != receiver) {
                    out << sender << "," << receiver << "," << std::fixed << std::setprecision(2)
                        << gathered_results[idx++] << std::endl;
                }
            }
        }
        out.close();
    }

    MPI_Finalize();
    return 0;
}
