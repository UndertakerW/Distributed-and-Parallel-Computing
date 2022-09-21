#include <chrono>
#include <iostream>
#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <vector>
#include <complex>
#include <mpi.h>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <memory>

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

void calculate(int* global_buffer, int size, int scale, double x_center, double y_center, int k_value) {
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
            global_buffer[i * size + j] = k;
        }
    }
}

void MPI_calculate(int* local_buffer, int size, int scale, double x_center,
        double y_center, int k_value, int row_number, int chunk_size) {

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

    int count = 0;
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
            local_buffer[count++] = k;
        }
    }
}

static constexpr float MARGIN = 4.0f;
static constexpr float BASE_SPACING = 2000.0f;
static constexpr size_t SHOW_THRESHOLD = 500000000ULL;


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int num_proc;
    int points_per_proc;
    int* global_buffer;
    int* local_buffer;
    int* int_params_buffer;
    double* double_params_buffer;
    int chunk_size = 10;
    int global_row_number;
    int_params_buffer = new int[4];
    double_params_buffer = new double[2];
    int center_x = 0;
    int center_y = 0;
    int size = 800;
    int scale = 1;
    int k_value = 100;

    if (argc > 2) {
        size = atoi(argv[1]);
        k_value = atoi(argv[2]);
    }
    if (argc > 3) {
        chunk_size = atoi(argv[3]);
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    if (0 == rank) {
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
                    auto begin = high_resolution_clock::now();
                    global_buffer = new int[size * size + size];
                    if (num_proc == 1) {
                        calculate(global_buffer, size, scale, center_x, center_y, k_value);
                    }
                    else {
                        int_params_buffer[0] = k_value;
                        int_params_buffer[1] = scale;
                        int_params_buffer[2] = size;
                        int_params_buffer[3] = chunk_size;
                        double_params_buffer[0] = center_x;
                        double_params_buffer[1] = center_y;
                        MPI_Bcast(int_params_buffer, 4, MPI_INT, 0, MPI_COMM_WORLD);
                        MPI_Bcast(double_params_buffer, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        int finished_count = 0;
                        int row_offset;
                        int current_row = 0;

                        while (finished_count < size) {
                            MPI_Status status;
                            MPI_Recv(&row_offset, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                            if (row_offset != -1) {
                                MPI_Recv(global_buffer + row_offset * size, 
                                    chunk_size * size, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                finished_count += chunk_size;
                            }
                            if (current_row < size) {
                                MPI_Send(&current_row, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                                current_row += chunk_size;
                            }
                            else {
                                int code = -1;
                                MPI_Send(&code, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                            }
                        }
                    }
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
                            if (global_buffer[i * size + j] == k_value) {
                                draw_list->AddCircleFilled(ImVec2(x, y), radius, col32);
                            }
                            x += spacing;
                        }
                        y += spacing;
                        x = p.x + MARGIN;
                    }
                    delete[] global_buffer;
                }
                ImGui::End();
            }
        });
    }
    else {
        while (true) {
            MPI_Bcast(int_params_buffer, 4, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(double_params_buffer, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int buffer_size = int_params_buffer[2] * int_params_buffer[3];
            int row_offset = -1;
            MPI_Send(&row_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            do {
                MPI_Recv(&row_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (row_offset != -1) {
                    local_buffer = new int[buffer_size];
                    MPI_calculate(local_buffer, int_params_buffer[2], int_params_buffer[1], double_params_buffer[0], 
                        double_params_buffer[1], int_params_buffer[0], row_offset, int_params_buffer[3]);
                    MPI_Send(&row_offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(local_buffer, buffer_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    delete[] local_buffer;
                }
            } while (row_offset != -1);
        }
    }
    delete[] int_params_buffer;
    delete[] double_params_buffer;
    MPI_Finalize();
    return 0;
}
