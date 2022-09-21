#include "odd-even-sort.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <limits.h>

namespace sort {
    using namespace std::chrono;


    Context::Context(int &argc, char **&argv) : argc(argc), argv(argv) {
        MPI_Init(&argc, &argv);
    }

    Context::~Context() {
        MPI_Finalize();
    }

    std::unique_ptr<Information> Context::mpi_sort(Element *begin, Element *end) const {
        int res; // result
        int rank; // rank of process
        int proc; // number of processes
        int size; // actual total size
        int global_size; // total size after padding
        int local_size; // local size (per process)
        Element send_buffer; // send buffer
        Element recv_buffer; // receive buffer
        Element* global_buffer; // global buffer containing all numbers (only available in process 0)
        Element* local_buffer; // local buffer containing part of numbers

        std::unique_ptr<Information> information{};

        res = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (MPI_SUCCESS != res) {
            throw std::runtime_error("failed to get MPI world rank");
        }

        if (0 == rank) {
            information = std::make_unique<Information>();
            information->length = end - begin;
            res = MPI_Comm_size(MPI_COMM_WORLD, &information->num_of_proc);
            if (MPI_SUCCESS != res) {
                throw std::runtime_error("failed to get MPI world size");
            };
            information->argc = argc;
            for (auto i = 0; i < argc; ++i) {
                information->argv.push_back(argv[i]);
            }
            information->start = high_resolution_clock::now();
            size = end - begin;
        }
        // Broadcast the total data size
        MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Comm_size(MPI_COMM_WORLD, &proc);
        local_size = (size + proc - 1) / proc;
        local_buffer = (Element *) malloc(sizeof(Element) * local_size);
        global_size = local_size * proc;

        // Build the global buffer
        if (0 == rank) {
            global_buffer = (Element *) malloc(sizeof(Element) * global_size);
            Element* element = begin;
            int i = 0;
            while (element != end) {
                global_buffer[i] = *element;
                i++;
                element++;
            }
            // do padding if there is empty space
            for (; i < global_size; i++) {
                global_buffer[i] = LONG_MAX;
            }
        }

        // Scatter the data
        MPI_Scatter(global_buffer, local_size, MPI_LONG, local_buffer, local_size, MPI_LONG, 0, MPI_COMM_WORLD);

        for (int iteration = 0; iteration < global_size; iteration++)
        {
            // odd round
            // if the first element in the local buffer is an odd element in the global buffer
            if ((rank * local_size) % 2 == 1) {
                for (int j = 2; j < local_size; j = j + 2) {
                    if (local_buffer[j - 1] > local_buffer[j]) {
                        swap(local_buffer + j - 1, local_buffer + j);
                    }
                }
                // send the first element back to the preceding process (rank - 1)
                if (rank > 0) {
                    send_buffer = local_buffer[0];
				    MPI_Send(&send_buffer, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(&recv_buffer, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    local_buffer[0] = recv_buffer;
                }
            }
            // otherwise, the first element in the local buffer is an even element in the global buffer
            else {
                for (int j = 1; j < local_size; j = j + 2) {
                    if (local_buffer[j - 1] > local_buffer[j]) {
                        swap(local_buffer + j - 1, local_buffer + j);
                    }
                }
            }
            // if the last element in the local buffer is an even element in the global buffer
            // receive the element from the succeeding process (rank + 1)
            if ((rank + 1) * local_size % 2 == 0 && rank < proc - 1) {
                MPI_Recv(&recv_buffer, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (recv_buffer < local_buffer[local_size - 1]) {
                    send_buffer = local_buffer[local_size - 1];
                    local_buffer[local_size - 1] = recv_buffer;
                }
                else {
                    send_buffer = recv_buffer;
                }
                MPI_Send(&send_buffer, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
            }

            // even round
            // if the first element in the local buffer is an even element in the global buffer
            if ((rank * local_size) % 2 == 0) {
                for (int j = 2; j < local_size; j = j + 2) {
                    if (local_buffer[j - 1] > local_buffer[j]) {
                        swap(local_buffer + j - 1, local_buffer + j);
                    }
                }
                // send the first element back to the preceding process (rank - 1)
                if (rank > 0) {
                    send_buffer = local_buffer[0];
				    MPI_Send(&send_buffer, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(&recv_buffer, 1, MPI_LONG, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    local_buffer[0] = recv_buffer;
                }
            }
            // otherwise, the first element in the local buffer is an odd element in the global buffer
            else {
                for (int j = 1; j < local_size; j = j + 2) {
                    if (local_buffer[j - 1] > local_buffer[j]) {
                        swap(local_buffer + j - 1, local_buffer + j);
                    }
                }
            }
            // if the last element in the local buffer is an odd element in the global buffer
            // receive the element from the succeeding process (rank + 1)
            if ((rank + 1) * local_size % 2 == 1 && rank < proc - 1) {
                MPI_Recv(&recv_buffer, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (recv_buffer < local_buffer[local_size - 1]) {
                    send_buffer = local_buffer[local_size - 1];
                    local_buffer[local_size - 1] = recv_buffer;
                }
                else {
                    send_buffer = recv_buffer;
                }
                MPI_Send(&send_buffer, 1, MPI_LONG, rank + 1, 0, MPI_COMM_WORLD);
            }
        }

        // gather the sorted array
        MPI_Gather(local_buffer, local_size, MPI_LONG, global_buffer, local_size, MPI_LONG, 0, MPI_COMM_WORLD);

        // copy the sorted array in the global buffer
        if (0 == rank) {
            Element* element = begin;
            for (int i = 0; i < size; i++) {
                *element = global_buffer[i];
                element ++;
            }
            information->end = high_resolution_clock::now();
        }
        return information;
    }

    std::ostream &Context::print_information(const Information &info, std::ostream &output) {
        auto duration = info.end - info.start;
        auto duration_count = duration_cast<nanoseconds>(duration).count();
        auto mem_size = static_cast<double>(info.length) * sizeof(Element) / 1024.0 / 1024.0 / 1024.0;
        output << "input size: " << info.length << std::endl;
        output << "proc number: " << info.num_of_proc << std::endl;
        // output << "duration (ns): " << duration_count << std::endl;
        output << "duration (ms): " << duration_count / (1000 * 1000) << std::endl;
        output << "throughput (gb/s): " << mem_size / static_cast<double>(duration_count) * 1'000'000'000.0
               << std::endl;
        return output;
    }

    void Context::sequential_sort(Element *begin, Element *end) const {
        size_t size = end - begin;
        if (size < 2)
            return;
        bool is_sorted = false;
        while (!is_sorted) {
            is_sorted = true;
            
            // odd round
            for (int i = 1; i < size; i = i + 2) {
                if (begin[i - 1] > begin[i]){
                    swap(begin + i - 1, begin + i);
                    is_sorted = false;
                }
            }
            // even round
            for (int i = 2; i < size; i = i + 2) {
                if (begin[i - 1] > begin[i]){
                    swap(begin + i - 1, begin + i);
                    is_sorted = false;
                }
            }
        }
    }

    void Context::swap(Element* a, Element *b) const {
        Element temp = *b;
        *b = *a;
        *a = temp;
    }

}
