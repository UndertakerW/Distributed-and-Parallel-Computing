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

pthread_mutex_t* mutex;
pthread_barrier_t barrier;

float gravity = 100;
float space = 800;
float radius = 5;
int bodies = 20;
float elapse = 0.001;
ImVec4 color = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
float max_mass = 50;
int max_iteration = 100;
int chunk = 0;

BodyPool* global_pool;

void* gui_loop(void* t) {

    int* int_ptr = (int *) t;
    int tid = *int_ptr;
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
                                pthread_mutex_lock(&mutex[tid2]);
                                pthread_mutex_lock(&mutex[tid]);
                            }
                            else if (tid2 > tid) {
                                pthread_mutex_lock(&mutex[tid]);
                                pthread_mutex_lock(&mutex[tid2]);
                            }
                            global_pool->check_and_update_pthread_collision(body_i, body_j, radius, gravity);
                            pthread_mutex_unlock(&mutex[tid]);
                            if (tid2 != tid) {
                                pthread_mutex_unlock(&mutex[tid2]);
                            }
                        }
                        else {
                            global_pool->check_and_update_pthread_no_collision(body_i, body_j, radius, gravity);
                        }
                    }
                }
                pthread_barrier_wait(&barrier);
                for (size_t i = tid * chunk; i < i_max; ++i) {
                    global_pool->get_body(i).update_for_tick(elapse, space, radius);
                }
                num_iteration ++;
                bodies_count += bodies;
                pthread_barrier_wait(&barrier);
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
                            pthread_mutex_lock(&mutex[tid2]);
                            pthread_mutex_lock(&mutex[tid]);
                        }
                        else if (tid2 > tid) {
                            pthread_mutex_lock(&mutex[tid]);
                            pthread_mutex_lock(&mutex[tid2]);
                        }
                        else {
                            pthread_mutex_lock(&mutex[tid]);
                        }
                        global_pool->check_and_update_pthread_collision(body_i, body_j, radius, gravity);
                        pthread_mutex_unlock(&mutex[tid]);
                        if (tid2 != tid) {
                            pthread_mutex_unlock(&mutex[tid2]);
                        }
                    }
                    else {
                        global_pool->check_and_update_pthread_no_collision(body_i, body_j, radius, gravity);
                    }
                }
            }
            pthread_barrier_wait(&barrier);
            for (size_t i = tid * chunk; i < i_max; ++i) {
                global_pool->get_body(i).update_for_tick(elapse, space, radius);
            }
            pthread_barrier_wait(&barrier);
        }
    }
    return nullptr;
}


int main(int argc, char **argv) {

    if (argc < 2) {
        exit(1);
    }
    int num_threads = atoi(argv[1]);

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

    chunk = (bodies + num_threads - 1) / num_threads;
    global_pool = new BodyPool(static_cast<size_t>(bodies), static_cast<size_t>(chunk * num_threads), space, max_mass);

    mutex = new pthread_mutex_t[num_threads];
    for (int i = 0; i < num_threads; i++) {
        mutex[i] = PTHREAD_MUTEX_INITIALIZER;
    }

    pthread_barrier_init(&barrier, nullptr, num_threads);
    pthread_t threads[num_threads];
    int tids[num_threads];
    for (int i = 0; i < num_threads; i++) {
        tids[i] = i;
        void* tid = (void*) &tids[i];
        pthread_create(&threads[i], NULL, gui_loop, tid);
    }
    for (auto &i : threads) {
        pthread_join(i, nullptr);
    }
    return 0;
}
