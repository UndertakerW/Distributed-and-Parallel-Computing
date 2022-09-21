#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cstring>

struct Square {
    std::vector<int> buffer;
    size_t length;

    explicit Square(size_t length) : buffer(length), length(length * length) {}

    void resize(size_t new_length) {
        buffer.assign(new_length * new_length, false);
        length = new_length;
    }

    auto& operator[](std::pair<size_t, size_t> pos) {
        return buffer[pos.second * length + pos.first];
    }
};

void calculate(Square &buffer, int size, int scale, double x_center, double y_center, int k_value) {
    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(i) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            buffer[{i, j}] = k;
        }
    }
}

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier_begin;
pthread_barrier_t barrier_end;
int chunk_size = 10;
int global_row_number;

void pthread_calculate(Square &buffer, int size, int scale, double x_center, double y_center, int k_value, int row_number) {

    if (size == 0 || scale == 0) {
        return;
    }

    double cx = static_cast<double>(size) / 2 + x_center;
    double cy = static_cast<double>(size) / 2 + y_center;
    double zoom_factor = static_cast<double>(size) / 4 * scale;

    int i_max = size;
    if (row_number + chunk_size < size) {
        i_max = row_number + chunk_size;
    }

    for (int i = row_number; i < i_max; ++i) {
        for (int j = 0; j < size; ++j) {
            double x = (static_cast<double>(j) - cx) / zoom_factor;
            double y = (static_cast<double>(i) - cy) / zoom_factor;
            std::complex<double> z{0, 0};
            std::complex<double> c{x, y};
            int k = 0;
            do {
                z = z * z + c;
                k++;
            } while (norm(z) < 2.0 && k < k_value);
            buffer[{i, j}] = k;
        }
    }
}

static constexpr float MARGIN = 4.0f;
static constexpr float BASE_SPACING = 2000.0f;
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;
int center_x = 0;
int center_y = 0;
int size = 800;
int scale = 1;
int k_value = 100;
Square canvas(100);

void* gui_loop(void* t) {
    int* int_ptr = (int *) t;
    int tid = *int_ptr;
    if (0 == tid) {
        graphic::GraphicContext context{"Assignment 2"};
        size_t duration = 0;
        size_t pixels = 0;
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
            {
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
                static ImVec4 col = ImVec4(1.0f, 1.0f, 0.4f, 1.0f);
                ImGui::DragInt("Center X", &center_x, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Center Y", &center_y, 1, -4 * size, 4 * size, "%d");
                ImGui::DragInt("Fineness", &size, 10, 100, 1000, "%d");
                ImGui::DragInt("Scale", &scale, 1, 1, 100, "%.01f");
                ImGui::DragInt("K", &k_value, 1, 100, 1000, "%d");
                ImGui::ColorEdit4("Color", &col.x);
                {
                    using namespace std::chrono;
                    auto spacing = BASE_SPACING / static_cast<float>(size);
                    auto radius = spacing / 2;
                    const ImVec2 p = ImGui::GetCursorScreenPos();
                    const ImU32 col32 = ImColor(col);
                    float x = p.x + MARGIN, y = p.y + MARGIN;
                    canvas.resize(size);
                    auto begin = high_resolution_clock::now();
                    pthread_mutex_lock(&mutex);
                    global_row_number = 0;
                    pthread_mutex_unlock(&mutex);
                    pthread_barrier_wait(&barrier_begin);
                    int row_number;
                    do {
                        pthread_mutex_lock(&mutex);
                        row_number = global_row_number;
                        if (row_number < size) {
                            global_row_number += chunk_size;
                        }
                        pthread_mutex_unlock(&mutex);
                        if (row_number < size) {
                            pthread_calculate(canvas, size, scale, center_x, center_y, k_value, row_number);
                        }
                    } while (row_number + chunk_size < size);
                    pthread_barrier_wait(&barrier_end);
                    auto end = high_resolution_clock::now();
                    pixels += size;
                    duration += duration_cast<nanoseconds>(end - begin).count();
                    if (duration > SHOW_THRESHOLD) {
                        std::cout << pixels << " pixels in last " << duration << " nanoseconds\n";
                        auto speed = static_cast<double>(pixels) / static_cast<double>(duration) * 1e9;
                        std::cout << "speed: " << speed << " pixels per second" << std::endl;
                        pixels = 0;
                        duration = 0;
                    }
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            if (canvas[{i, j}] == k_value) {
                                draw_list->AddCircleFilled(ImVec2(x, y), radius, col32);
                            }
                            x += spacing;
                        }
                        y += spacing;
                        x = p.x + MARGIN;
                    }
                }
                ImGui::End();
            }
        });
    }
    else {
        while (true) {
            pthread_barrier_wait(&barrier_begin);
            int row_number;
            do {
                pthread_mutex_lock(&mutex);
                row_number = global_row_number;
                if (row_number < size) {
                    global_row_number += chunk_size;
                }
                pthread_mutex_unlock(&mutex);
                if (row_number < size) {
                    pthread_calculate(canvas, size, scale, center_x, center_y, k_value, row_number);
                }
            } while (row_number + chunk_size < size);
            pthread_barrier_wait(&barrier_end);
        }
    }
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        exit(1);
    }
    if (argc > 3) {
        size = atoi(argv[2]);
        k_value = atoi(argv[3]);
    }
    if (argc > 4) {
        chunk_size = atoi(argv[4]);
    }
    int num_threads = atoi(argv[1]);
    pthread_barrier_init(&barrier_begin, nullptr, num_threads);
    pthread_barrier_init(&barrier_end, nullptr, num_threads);
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
