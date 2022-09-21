#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <omp.h>

template<typename ...Args>
void UNUSED(Args &&... args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

int main(int argc, char **argv) {

    if (argc < 2) {
        exit(1);
    }
    int n_threads = atoi(argv[1]);

    hdist::State current_state, last_state;
    bool first = true;
    int finished = 0;
    int local_finished = finished;
    int max_iteration = 100;
    int chunk = 0;

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

    chunk = (current_state.room_size + n_threads - 1) / n_threads;
    int capacity = chunk * n_threads;

    hdist::Grid grid(
        static_cast<size_t>(current_state.room_size),
        static_cast<size_t>(capacity),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y));

    omp_lock_t lock;
    omp_init_lock(&lock); 

    #pragma omp parallel num_threads(n_threads)
    {
        long tid = omp_get_thread_num();
        int num_iteration = 0;

        if (0 == tid) {
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
                        if (current_state.algo == hdist::Algorithm::Jacobi) {
                            local_finished = (int) hdist::calculate_jacobi(current_state, grid, tid, chunk);
                        }
                        else if (current_state.algo == hdist::Algorithm::Sor) {
                            local_finished = (int) hdist::calculate_sor(current_state, grid, tid, chunk, 0);
                            // sync threads and switch buffer
                            #pragma omp barrier
                            grid.switch_buffer();
                            // acknowledge other threads that the buffer has been switched
                            #pragma omp barrier
                            local_finished = (int) hdist::calculate_sor(current_state, grid, tid, chunk, 1);
                        }
                        omp_set_lock(&lock);
                        // update the global status (finished or not)
                        finished &= local_finished;
                        omp_unset_lock(&lock);
                        // sync threads and switch buffer
                        #pragma omp barrier
                        grid.switch_buffer();
                        // acknowledge other threads that the buffer has been switched
                        #pragma omp barrier

                        // hence this does not count into the total time
                        end = std::chrono::high_resolution_clock::now();
                        duration += duration_cast<std::chrono::nanoseconds>(end - begin).count();
                        num_iteration ++;
                    }
                }

                const ImVec2 p = ImGui::GetCursorScreenPos();
                float x = p.x + current_state.block_size, y = p.y + current_state.block_size;

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
        }
        else {
            while (num_iteration < max_iteration) {
                if (!finished) {
                    if (current_state.algo == hdist::Algorithm::Jacobi) {
                        local_finished = (int) hdist::calculate_jacobi(current_state, grid, tid, chunk);
                    }
                    else if (current_state.algo == hdist::Algorithm::Sor) {
                        local_finished = (int) hdist::calculate_sor(current_state, grid, tid, chunk, 0);
                        // sync threads for switching buffer
                        #pragma omp barrier
                        // wait thread 0 to switch buffer
                        #pragma omp barrier
                        local_finished = (int) hdist::calculate_sor(current_state, grid, tid, chunk, 1);
                    }
                    omp_set_lock(&lock);
                    // update the global status (finished or not)
                    finished &= local_finished;
                    omp_unset_lock(&lock);
                    // sync threads for switching buffer
                    #pragma omp barrier
                    // wait thread 0 to switch buffer
                    #pragma omp barrier
                }
                num_iteration ++;
            }
        }
    }

    return 0;
}
