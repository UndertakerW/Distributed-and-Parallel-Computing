#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>

namespace hdist {

    enum class Algorithm : int {
        Jacobi = 0,
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const;
    };

    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        double* data0;
        double* data1;
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                    : length(size) {
            data0 = new double[size * size];
            data1 = new double[size * size];
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        explicit Grid(size_t size,
                      size_t capacity,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                    : length(size) {
            data0 = new double[size * capacity];
            data1 = new double[size * capacity];
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        double* get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        __device__ double* get_current_buffer_device() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        __device__ double* &get_alt_buffer() {
            if (current_buffer == 0) return data1;
            return data0;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[
                    std::get<1>(index) * length + std::get<2>(index)];
        }

        __device__ double &get_element(size_t i, size_t j) {
            return get_current_buffer_device()[i * length + j];
        }

        __device__ double &get_element_alt(size_t i, size_t j) {
            return current_buffer == 1 ? data0[i * length + j] : data1[i * length + j];
        }

        __device__ void switch_buffer() {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult {
        bool stable;
        double temp;
    };

    __device__ UpdateResult update_single(size_t i, size_t j, Grid *grid, const State *state) {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state->room_size - 1 || j == state->room_size - 1) {
            result.temp = state->border_temp;
        } else if (i == state->source_x && j == state->source_y) {
            result.temp = state->source_temp;
        } else {
            auto sum = (grid->get_element(i + 1, j) + grid->get_element(i - 1, j) + grid->get_element(i, j + 1) + grid->get_element(i, j - 1));
            switch (state->algo) {
                case Algorithm::Jacobi:
                    result.temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    result.temp = grid->get_element(i, j) + (1.0 / state->sor_constant) * (sum - 4.0 * grid->get_element(i, j));
                    break;
            }
        }
        result.stable = std::fabs(grid->get_element(i, j) - result.temp) < state->tolerance;
        return result;
    }

    __device__ bool calculate(const State &state, Grid &grid) {
        bool stabilized = true;

        switch (state.algo) {
            case Algorithm::Jacobi:
                for (size_t i = 0; i < state.room_size; ++i) {
                    for (size_t j = 0; j < state.room_size; ++j) {
                        auto result = update_single(i, j, &grid, &state);
                        stabilized &= result.stable;
                        grid.get_element_alt(i, j) = result.temp;
                    }
                }
                grid.switch_buffer();
                break;
            case Algorithm::Sor:
                for (auto k : {0, 1}) {
                    for (size_t i = 0; i < state.room_size; i++) {
                        for (size_t j = 0; j < state.room_size; j++) {
                            if (k == ((i + j) & 1)) {
                                auto result = update_single(i, j, &grid, &state);
                                stabilized &= result.stable;
                                grid.get_element_alt(i, j) = result.temp;
                            } else {
                                grid.get_element_alt(i, j) = grid.get_element(i, j);
                            }
                        }
                    }
                    grid.switch_buffer();
                }
        }
        return stabilized;
    };

    __device__ bool calculate_jacobi(const State *state, Grid *grid, int rank, int chunk) {

        bool stabilized = true;

        int i_max = 0;
        if ((rank + 1) * chunk < state->room_size) {
            i_max = (rank + 1) * chunk;
        }
        else {
            i_max = state->room_size;
        }

        for (size_t i = rank * chunk; i < i_max; ++i) {
            for (size_t j = 0; j < state->room_size; ++j) {
                auto result = update_single(i, j, grid, state);
                stabilized &= result.stable;
                grid->get_element_alt(i, j) = result.temp;
            }
        }

        return stabilized;
    };

    __device__ bool calculate_sor(const State *state, Grid *grid, int rank, int chunk, int k) {

        bool stabilized = true;

        int i_max = 0;
        if ((rank + 1) * chunk < state->room_size) {
            i_max = (rank + 1) * chunk;
        }
        else {
            i_max = state->room_size;
        }

        for (size_t i = rank * chunk; i < i_max; ++i) {
            for (size_t j = 0; j < state->room_size; j++) {
                if (k == ((i + j) & 1)) {
                    auto result = update_single(i, j, grid, state);
                    stabilized &= result.stable;
                    grid->get_element_alt(i, j) = result.temp;
                } else {
                    grid->get_element_alt(i, j) = grid->get_element(i, j);
                }
            }
        }

        return stabilized;
    };

} // namespace hdist