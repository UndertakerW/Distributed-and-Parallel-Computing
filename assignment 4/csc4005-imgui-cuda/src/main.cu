#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>

template<typename ...Args>
void UNUSED(Args &&... args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

__global__ void calculate(hdist::Grid* grid,
                            hdist::State* current_state,
                            int* chunk,
                            bool* finished,
                            int* lock) {
    int tid = threadIdx.x;
    bool local_finished = false;
    if (current_state->algo == hdist::Algorithm::Jacobi) {
        local_finished = hdist::calculate_jacobi(current_state, grid, tid, *chunk);
    }
    else if (current_state->algo == hdist::Algorithm::Sor) {
        local_finished = hdist::calculate_sor(current_state, grid, tid, *chunk, 0);
        // sync threads and switch buffer
        __syncthreads();
        if (tid == 0) {
            grid->switch_buffer();
        }
        // acknowledge other threads that the buffer has been switched
        __syncthreads();
        local_finished = hdist::calculate_sor(current_state, grid, tid, *chunk, 1);
    }
    while (*lock != 0) {};
    *lock = 1;
    // update the global status (finished or not)
    *finished &= local_finished;
    *lock = 0;
    // sync threads and switch buffer
    __syncthreads();
    if (tid == 0) {
        grid->switch_buffer();
    }
}

int main(int argc, char **argv) {

    if (argc < 2) {
        exit(1);
    }
    int num_threads = atoi(argv[1]);

    bool first = true;
    bool finished = false;
    hdist::State current_state; //, last_state;
    int max_iteration = 100;
    int chunk = 0;
    int num_iteration = 0;

    if (argc >= 3) {
        current_state.room_size = atoi(argv[2]);
        current_state.source_x = current_state.room_size / 2;
        current_state.source_y = current_state.room_size / 2;
    }
    if (argc >= 4) {
        max_iteration = atoi(argv[3]);
    }
    if (argc >= 5) {
        int algo = atoi(argv[4]);
        if (algo == 0) {
            current_state.algo = hdist::Algorithm::Jacobi;
        }
        else {
            current_state.algo = hdist::Algorithm::Sor;
        }
    }

    chunk = (current_state.room_size + num_threads - 1) / num_threads;
    int capacity = chunk * num_threads;

    auto grid = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        static_cast<size_t>(capacity),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};

    cudaError_t cudaStatus;
    hdist::Grid* cuda_grid;
    hdist::State* cuda_current_state;
    int* cuda_chunk;
    bool* cuda_finished;
    int* cuda_lock;
    double* cuda_data0;
    double* cuda_data1;

    size_t cuda_current_buffer;
    int array_size = current_state.room_size * capacity * sizeof(double);

    cudaMallocManaged((void**) &cuda_grid, sizeof(grid));
    cudaMemcpy(cuda_grid, &grid, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &cuda_current_state, sizeof(current_state));
    cudaMemcpy(cuda_current_state, &current_state, sizeof(current_state), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &cuda_chunk, sizeof(int));
    cudaMemcpy(cuda_chunk, &chunk, sizeof(int), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &cuda_finished, sizeof(bool));
    cudaMemcpy(cuda_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &cuda_lock, sizeof(int));
    cudaMemset(cuda_lock, 0, sizeof(int));
    cudaMallocManaged((void**) &cuda_data0, array_size);
    cudaMallocManaged((void**) &cuda_data1, array_size);
    cuda_grid->data0 = cuda_data0;
    cuda_grid->data1 = cuda_data1;
    cudaMemcpy(cuda_grid->data0, grid.data0, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_grid->data1, grid.data1, array_size, cudaMemcpyHostToDevice);

    static std::chrono::high_resolution_clock::time_point begin, end;
    static size_t duration = 0;
    static const char* algo_list[2] = { "jacobi", "sor" };
    graphic::GraphicContext context{"Assignment 4"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 4", nullptr,
                    ImGuiWindowFlags_NoMove
                    | ImGuiWindowFlags_NoCollapse
                    | ImGuiWindowFlags_NoTitleBar
                    | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
        ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
        ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
        ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
        ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
        ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

        if (current_state.algo == hdist::Algorithm::Sor) {
            ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
        }

        // GUI paramater adjustment is disbled

        // if (current_state.room_size != last_state.room_size) {
        //     grid = hdist::Grid{
        //             static_cast<size_t>(current_state.room_size),
        //             current_state.border_temp,
        //             current_state.source_temp,
        //             static_cast<size_t>(current_state.source_x),
        //             static_cast<size_t>(current_state.source_y)};
        //     first = true;
        // }

        // if (current_state != last_state) {
        //     last_state = current_state;
        //     finished = false;
        // }

        if (first) {
            first = false;
            finished = 0;
        }

        if (!finished) {
            if (num_iteration < max_iteration) {
                begin = std::chrono::high_resolution_clock::now();

                // Launch GPU kernel
                calculate<<<1, num_threads>>>(cuda_grid, 
                                                cuda_current_state,
                                                cuda_chunk,
                                                cuda_finished,
                                                cuda_lock);
                // Wait GPU to complete calculation
                cudaDeviceSynchronize();
                cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "mykernel launch failed: %s\n",
                            cudaGetErrorString(cudaStatus));
                    return 0;
                }
                // Get the finished status
                cudaMemcpy(&finished, cuda_finished, sizeof(bool), cudaMemcpyDeviceToHost);
                end = std::chrono::high_resolution_clock::now();
                duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                num_iteration ++;
            }
        }

        const ImVec2 p = ImGui::GetCursorScreenPos();
        float x = p.x + current_state.block_size, y = p.y + current_state.block_size;


        // Receive data from device
        cudaMemcpy(&cuda_current_buffer, &cuda_grid->current_buffer,
                    sizeof(size_t), cudaMemcpyDeviceToHost);
        if (cuda_current_buffer == 0)
        {
            cudaMemcpy(grid.get_current_buffer(), cuda_grid->data0,
                        array_size, cudaMemcpyDeviceToHost);
        }
        else {
            cudaMemcpy(grid.get_current_buffer(), cuda_grid->data1,
                        array_size, cudaMemcpyDeviceToHost);
        }

        for (size_t i = 0; i < current_state.room_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {
                auto temp = grid[{i, j}];
                auto color = temp_to_color(temp);
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                y += current_state.block_size;
            }
            x += current_state.block_size;
            y = p.y + current_state.block_size;
        }

        if (finished) {
            ImGui::Text("stabilized in %lf ms", (double) duration / 1'000'000);
        }
        else if (num_iteration >= max_iteration) {
            ImGui::Text("%d iterations reached in %lf ms", max_iteration, (double) duration / 1'000'000);
        }

        ImGui::End();
    });

    return 0;
}
