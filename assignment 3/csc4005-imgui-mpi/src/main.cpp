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

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

int main(int argc, char **argv) {

    int size;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_iteration = 0;
    int max_iteration = 100;
    size_t duration = 0;
    size_t bodies_count = 0;

    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 0.001;
    static ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;

    if (argc >= 2) {
        gravity = atof(argv[1]);
    }
    if (argc >= 3) {
        bodies = atof(argv[2]);
    }
    if (argc >= 4) {
        elapse = atof(argv[3]);
    }
    if (argc >= 5) {
        max_iteration = atoi(argv[4]);
    }

    static float current_space = space;
    static float current_max_mass = max_mass;
    static int current_bodies = bodies;

    double* x;
    double* y;
    double* vx;
    double* vy;

    int chunk = (bodies + size - 1) / size;

    // The process-local bodypool
    BodyPool* local_pool = new BodyPool(static_cast<size_t>(bodies), static_cast<size_t>(chunk * size), space, max_mass);
   
    // Max index for each process
    int i_max = 0;
    if (chunk * (rank + 1) > (int) local_pool->size()) {
        i_max = (int) local_pool->size();
    }
    else {
        i_max = chunk * (rank + 1);
    }
    if (0 == rank) {
        // Broadcast the mass
        MPI_Bcast(local_pool->m.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Buffer for MPI gather
        x = new double[chunk * size];
        y = new double[chunk * size];
        vx = new double[chunk * size];
        vy = new double[chunk * size];
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
            ImGui::DragFloat("Gravity", &gravity, 0.5, 0, 1000, "%f");
            ImGui::DragFloat("Radius", &radius, 0.5, 2, 20, "%f");
            ImGui::DragInt("Bodies", &current_bodies, 1, 2, 100, "%d");
            ImGui::DragFloat("Elapse", &elapse, 0.001, 0.001, 10, "%f");
            ImGui::DragFloat("Max Mass", &current_max_mass, 0.5, 5, 100, "%f");
            ImGui::ColorEdit4("Color", &color.x);
            
            // Parameter adjustment in GUI is disabled
            auto begin = std::chrono::high_resolution_clock::now();

            if (num_iteration < max_iteration) {
                // Broadcast the location and velocity
                MPI_Bcast(local_pool->x.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(local_pool->y.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(local_pool->vx.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(local_pool->vy.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                for (size_t i = rank * chunk; i < i_max; ++i) {
                    local_pool->ax[i] = 0;
                    local_pool->ay[i] = 0;
                    for (size_t j = 0; j < local_pool->size(); ++j) {
                        if (j == i) continue;
                        local_pool->check_and_update_mpi(local_pool->get_body(i), local_pool->get_body(j), radius, gravity);
                    }
                }
                for (size_t i = rank * chunk; i < i_max; ++i) {
                    local_pool->get_body(i).update_for_tick(elapse, space, radius);
                }
                // Gather the location and velocity
                MPI_Gather(&local_pool->x[chunk * rank], chunk, MPI_DOUBLE, x, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&local_pool->y[chunk * rank], chunk, MPI_DOUBLE, y, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&local_pool->vx[chunk * rank], chunk, MPI_DOUBLE, vx, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Gather(&local_pool->vy[chunk * rank], chunk, MPI_DOUBLE, vy, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                for (int i = 0; i < local_pool->size(); i++) {
                    local_pool->x[i] = x[i];
                    local_pool->y[i] = y[i];
                    local_pool->vx[i] = vx[i];
                    local_pool->vy[i] = vy[i];
                }
                num_iteration ++;
                bodies_count += bodies;
            }
            auto end = std::chrono::high_resolution_clock::now();
            duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
            // Only print the result once
            static bool isResultPrinted = false;
            if (!isResultPrinted && num_iteration >= max_iteration) {
                std::cout << bodies_count << " bodies in last " << duration << " nanoseconds\n";
                auto speed = static_cast<double>(bodies_count) / static_cast<double>(duration) * 1e6;
                std::cout << "speed: " << std::setprecision(12) << speed << " bodies per millisecond" << std::endl;
                isResultPrinted = true;
            }
            // Draw the balls
            {
                const ImVec2 p = ImGui::GetCursorScreenPos();
                for (size_t i = 0; i < local_pool->size(); ++i) {
                    auto body = local_pool->get_body(i);
                    auto x = p.x + static_cast<float>(body.get_x());
                    auto y = p.y + static_cast<float>(body.get_y());
                    draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                }
            }

            ImGui::End();
        });
    }

    else {
        // Receive the mass
        MPI_Bcast(local_pool->m.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int it = 0; it < max_iteration; it++) {
            // Synchronize the location and velocity
            MPI_Bcast(local_pool->x.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(local_pool->y.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(local_pool->vx.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(local_pool->vy.data(), (int) local_pool->size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (size_t i = rank * chunk; i < i_max; ++i) {
                local_pool->ax[i] = 0;
                local_pool->ay[i] = 0;
                for (size_t j = 0; j < local_pool->size(); ++j) {
                    if (j == i) continue;
                    local_pool->check_and_update_mpi(local_pool->get_body(i), local_pool->get_body(j), radius, gravity);
                }
            }
            for (size_t i = rank * chunk; i < i_max; ++i) {
                local_pool->get_body(i).update_for_tick(elapse, space, radius);
            }
            // Send back the location and velocity
            MPI_Gather(&local_pool->x[chunk * rank], chunk, MPI_DOUBLE, x, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_pool->y[chunk * rank], chunk, MPI_DOUBLE, y, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_pool->vx[chunk * rank], chunk, MPI_DOUBLE, vx, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&local_pool->vy[chunk * rank], chunk, MPI_DOUBLE, vy, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

}
