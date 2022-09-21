#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <inttypes.h>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

int bodies = 20;
ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
float max_mass = 50;
int max_iteration = 100;

__global__ void calculate(BodyPool* cuda_pool,
                            float* gravity,
                            float* space,
                            float* radius,
                            float* elapse,
                            int* chunk,
                            int* lock) {

    int tid = blockIdx.x;
    int i_max = 0;
    if (*chunk * (tid + 1) > (int) cuda_pool->size()) {
        i_max = (int) cuda_pool->size();
    }
    else {
        i_max = *chunk * (tid + 1);
    }
    for (size_t i = tid * *chunk; i < i_max; ++i) {
        cuda_pool->ax[i] = 0;
        cuda_pool->ay[i] = 0;
        for (size_t j = 0; j < cuda_pool->size(); ++j) {
            if (j == i) continue;
            BodyPool::Body body_i = cuda_pool->get_body(i);
            BodyPool::Body body_j = cuda_pool->get_body(j);
            if (body_i.collide(body_j, *radius)) {
                int tid2 = (int) j / *chunk;
                if (tid2 < tid) {
                    while (lock[tid2] != 0) {};
                    lock[tid2] = 1;
                    while (lock[tid] != 0) {};
                    lock[tid] = 1;
                }
                else if (tid2 > tid) {
                    while (lock[tid] != 0) {};
                    lock[tid] = 1;
                    while (lock[tid2] != 0) {};
                    lock[tid2] = 1;
                }
                else {
                    while (lock[tid] != 0) {};
                    lock[tid] = 1;
                }
                cuda_pool->check_and_update_pthread_collision(body_i, body_j, *radius, *gravity);
                lock[tid] = 0;
                if (tid2 != tid) {
                    lock[tid2] = 0;
                }
            }
            else {
                cuda_pool->check_and_update_pthread_no_collision(body_i, body_j, *radius, *gravity);
            }
        }
    }
    __syncthreads();
    for (size_t i = tid * *chunk; i < i_max; ++i) {
        cuda_pool->get_body(i).update_for_tick(*elapse, *space, *radius);
    }
    __syncthreads();
}

int main(int argc, char **argv) {

    if (argc < 2) {
        exit(1);
    }
    int num_threads = atoi(argv[1]);

    float host_gravity = 100;
    float host_space = 800;
    float host_radius = 5;
    float host_elapse = 0.001;

    if (argc >= 3) {
        host_gravity = atof(argv[2]);
    }
    if (argc >= 4) {
        bodies = atof(argv[3]);
    }
    if (argc >= 5) {
        host_elapse = atof(argv[4]);
    }
    if (argc >= 6) {
        max_iteration = atoi(argv[5]);
    }

    int host_chunk = (bodies + num_threads - 1) / num_threads;

    BodyPool* cuda_pool;
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    double* m;
    float* gravity;
    float* space;
    float* radius;
    float* elapse;
    int* chunk;
    int* lock;

    BodyPool bp(static_cast<size_t>(bodies), static_cast<size_t>(host_chunk * num_threads), host_space, max_mass);

    cudaMallocManaged((void**) &cuda_pool, sizeof(bp));
    cudaMemcpy(cuda_pool, &bp, sizeof(bp), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &gravity, sizeof(float));
    cudaMemcpy(gravity, &host_gravity, sizeof(float), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &space, sizeof(float));
    cudaMemcpy(space, &host_space, sizeof(float), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &radius, sizeof(float));
    cudaMemcpy(radius, &host_radius, sizeof(float), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &elapse, sizeof(float));
    cudaMemcpy(elapse, &host_elapse, sizeof(float), cudaMemcpyHostToDevice);
    cudaMallocManaged((void**) &chunk, sizeof(int));
    cudaMemcpy(chunk, &host_chunk, sizeof(int), cudaMemcpyHostToDevice);

    size_t array_size = sizeof(double) * host_chunk * num_threads;

    cudaMallocManaged((void**) &x, array_size);
    cudaMallocManaged((void**) &y, array_size);
    cudaMallocManaged((void**) &vx, array_size);
    cudaMallocManaged((void**) &vy, array_size);
    cudaMallocManaged((void**) &ax, array_size);
    cudaMallocManaged((void**) &ay, array_size);
    cudaMallocManaged((void**) &m, array_size);
    cuda_pool->x = x;
    cuda_pool->y = y;
    cuda_pool->vx = vx;
    cuda_pool->vy = vy;
    cuda_pool->ax = ax;
    cuda_pool->ay = ay;
    cuda_pool->m = m;
    cudaMemcpy(cuda_pool->m, bp.m, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_pool->x, bp.x, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_pool->y, bp.y, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_pool->vx, bp.vx, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_pool->vy, bp.vy, array_size, cudaMemcpyHostToDevice);

    cudaMallocManaged((void**) &lock, sizeof(int) * num_threads);
    cudaMemset(lock, 0, sizeof(int) * num_threads);

    cudaError_t cudaStatus;
    size_t duration = 0;
    size_t bodies_count = 0;

    static float current_space = host_space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;
    int num_iteration = 0;
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "mykernel launch failed: %s\n",
                cudaGetErrorString(cudaStatus));
        return 0;
    }

    graphic::GraphicContext context{"Assignment 2"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 2", nullptr,
                    ImGuiWindowFlags_NoMove
                    | ImGuiWindowFlags_NoCollapse
                    | ImGuiWindowFlags_NoTitleBar
                    | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragFloat("Space", &current_space, 10, 200, 1600, "%f");
        ImGui::DragFloat("Gravity", &host_gravity, 0.5, 0, 1000, "%f");
        ImGui::DragFloat("Radius", &host_radius, 0.5, 2, 20, "%f");
        ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
        ImGui::DragFloat("Elapse", &host_elapse, 0.001, 0.001, 10, "%f");
        ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
        ImGui::ColorEdit4("Color", &color.x);
        
        // Parameter adjustment in GUI is disabled
        auto begin = std::chrono::high_resolution_clock::now();
        if (num_iteration < max_iteration) {
            // Launch to GPU kernel
            calculate<<<1, num_threads>>>(cuda_pool, gravity, space, radius, elapse, chunk, lock);
            cudaDeviceSynchronize();

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "mykernel launch failed: %s\n",
                        cudaGetErrorString(cudaStatus));
                return 0;
            }
            num_iteration ++;
            bodies_count += bodies;
        }
        auto end = std::chrono::high_resolution_clock::now();
        duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // only print the result once
        static bool isResultPrinted = false;
        if (!isResultPrinted && num_iteration >= max_iteration) {
            std::cout << bodies_count << " bodies in last " << duration << " nanoseconds\n";
            auto speed = static_cast<double>(bodies_count) / static_cast<double>(duration) * 1e6;
            std::cout << "speed: " << std::setprecision(12) << speed << " bodies per millisecond" << std::endl;
            isResultPrinted = true;
        }
        // draw the balls
        {
            // Receive data from device
            cudaMemcpy(bp.x, cuda_pool->x, array_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(bp.y, cuda_pool->y, array_size, cudaMemcpyDeviceToHost);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            for (size_t i = 0; i < bodies; ++i) {
                auto x = p.x + static_cast<float>(bp.x[i]);
                auto y = p.y + static_cast<float>(bp.y[i]);
                draw_list->AddCircleFilled(ImVec2(x, y), host_radius, ImColor{color});
            }
        }

        ImGui::End();
    });
    
    cudaDeviceReset();

    return 0;
}
