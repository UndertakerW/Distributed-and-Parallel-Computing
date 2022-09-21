#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <nbody/body.hpp>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <cstddef>

template <typename ...Args>
void UNUSED(Args&&... args [[maybe_unused]]) {}

omp_lock_t* lock;

int main(int argc, char **argv) {

    if (argc < 2) {
        exit(1);
    }
    int n_threads = atoi(argv[1]);

    static float gravity = 100;
    static float space = 800;
    static float radius = 5;
    static int bodies = 20;
    static float elapse = 0.001;
    ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
    static float max_mass = 50;
    int max_iteration = 100;
    int chunk = 0;

    if (argc >= 3) {
        gravity = atof(argv[2]);
    }
    if (argc >= 4) {
        bodies = atof(argv[3]);
    }
    if (argc >= 5) {
        elapse = atof(argv[4]);
    }
    if (argc >= 6) {
        max_iteration = atoi(argv[5]);
    }

    lock = new omp_lock_t[n_threads];
    for (int i = 0; i < n_threads; i++) {
        omp_init_lock(&lock[i]); 
    }

    chunk = (bodies + n_threads - 1) / n_threads;
    BodyPool bp(static_cast<size_t>(bodies), static_cast<size_t>(chunk * n_threads), space, max_mass);
    BodyPool* global_pool = &bp;

    #pragma omp parallel num_threads(n_threads)
    {
        long tid = omp_get_thread_num();
        int num_iteration = 0;
        
        // max index for each process
        int i_max = 0;
        if (chunk * (tid + 1) > (int) global_pool->size()) {
            i_max = (int) global_pool->size();
        }
        else {
            i_max = chunk * (tid + 1);
        }

        if (0 == tid) {
            size_t duration = 0;
            size_t bodies_count = 0;

            static float current_space = space;
            static float current_max_mass = max_mass;
            static int current_bodies = bodies;

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
                    for (size_t i = tid * chunk; i < i_max; ++i) {
                        global_pool->ax[i] = 0;
                        global_pool->ay[i] = 0;
                        for (size_t j = 0; j < global_pool->size(); ++j) {
                            if (j == i) continue;
                            BodyPool::Body body_i = global_pool->get_body(i);
                            BodyPool::Body body_j = global_pool->get_body(j);
                            if (body_i.collide(body_j, radius)) {
                                int tid2 = (int) j / chunk;
                                if (tid2 < tid) {
                                    omp_set_lock(&lock[tid2]);
                                    omp_set_lock(&lock[tid]);
                                }
                                else if (tid2 > tid) {
                                    omp_set_lock(&lock[tid]);
                                    omp_set_lock(&lock[tid2]);
                                }
                                else {
                                    omp_set_lock(&lock[tid]);
                                }
                                global_pool->check_and_update_pthread_collision(body_i, body_j, radius, gravity);
                                omp_unset_lock(&lock[tid]);
                                if (tid2 != tid) {
                                    omp_unset_lock(&lock[tid2]);
                                }
                            }
                            else {
                                global_pool->check_and_update_pthread_no_collision(body_i, body_j, radius, gravity);
                            }
                        }
                    }
                    for (size_t i = tid * chunk; i < i_max; ++i) {
                        global_pool->get_body(i).update_for_tick(elapse, space, radius);
                    }
                    num_iteration ++;
                    bodies_count += bodies;
                    #pragma omp barrier
                }
                auto end = std::chrono::high_resolution_clock::now();
                duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
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
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    for (size_t i = 0; i < global_pool->size(); ++i) {
                        auto body = global_pool->get_body(i);
                        auto x = p.x + static_cast<float>(body.get_x());
                        auto y = p.y + static_cast<float>(body.get_y());
                        draw_list->AddCircleFilled(ImVec2(x, y), radius, ImColor{color});
                    }
                }

                ImGui::End();
            });
        }
        else {
            while (true) {
                for (size_t i = tid * chunk; i < i_max; ++i) {
                    global_pool->ax[i] = 0;
                    global_pool->ay[i] = 0;
                    for (size_t j = 0; j < global_pool->size(); ++j) {
                        if (j == i) continue;
                        BodyPool::Body body_i = global_pool->get_body(i);
                        BodyPool::Body body_j = global_pool->get_body(j);
                        if (body_i.collide(body_j, radius)) {
                            int tid2 = (int) j / chunk;
                            if (tid2 < tid) {
                                omp_set_lock(&lock[tid2]);
                                omp_set_lock(&lock[tid]);
                            }
                            else if (tid2 > tid) {
                                omp_set_lock(&lock[tid]);
                                omp_set_lock(&lock[tid2]);
                            }
                            else {
                                omp_set_lock(&lock[tid]);
                            }
                            global_pool->check_and_update_pthread_collision(body_i, body_j, radius, gravity);
                            omp_unset_lock(&lock[tid]);
                            if (tid2 != tid) {
                                omp_unset_lock(&lock[tid2]);
                            }
                        }
                        else {
                            global_pool->check_and_update_pthread_no_collision(body_i, body_j, radius, gravity);
                        }
                        
                    }
                }
                for (size_t i = tid * chunk; i < i_max; ++i) {
                    global_pool->get_body(i).update_for_tick(elapse, space, radius);
                }
                #pragma omp barrier
            }
        }
    }

    return 0;
}
