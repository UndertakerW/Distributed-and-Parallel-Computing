//
// Created by schrodinger on 11/2/21.
//
#pragma once

#include <random>
#include <utility>

__global__ class BodyPool {

public:
    // provides in this way so that
    // it is easier for you to send a the vector with MPI
    double* x;
    double* y;
    double* vx;
    double* vy;
    double* ax;
    double* ay;
    double* m;
    size_t actual_size;
    // so the movements of bodies are calculated discretely.
    // if after the collision, we do not separate the bodies a little bit, it may
    // results in strange outcomes like infinite acceleration.
    // hence, we will need to set up a ratio for separation.
    static constexpr double COLLISION_RATIO = 0.01;

    __global__ class Body {
        size_t index;
        BodyPool &pool;

        friend class BodyPool;

        __device__ Body(size_t index, BodyPool &pool) : index(index), pool(pool) {}

    public:
        __device__ double &get_x() {
            return pool.x[index];
        }

        __device__ double &get_y() {
            return pool.y[index];
        }

        __device__ double &get_vx() {
            return pool.vx[index];
        }

        __device__ double &get_vy() {
            return pool.vy[index];
        }

        __device__ double &get_ax() {
            return pool.ax[index];
        }

        __device__ double &get_ay() {
            return pool.ay[index];
        }

        __device__ double &get_m() {
            return pool.m[index];
        }

        __device__ double distance_square(Body &that) {
            auto delta_x = get_x() - that.get_x();
            auto delta_y = get_y() - that.get_y();
            return delta_x * delta_x + delta_y * delta_y;
        }

        __device__ double distance(Body &that) {
            return std::sqrt(distance_square(that));
        }

        __device__ double delta_x(Body &that) {
            return get_x() - that.get_x();
        }

        __device__ double delta_y(Body &that) {
            return get_y() - that.get_y();
        }

        __device__ bool collide(Body &that, double radius) {
            return distance_square(that) <= radius * radius;
        }

        // collision with wall
        __device__ void handle_wall_collision(double position_range, double radius) {
            bool flag = false;
            if (get_x() <= radius) {
                flag = true;
                get_x() = radius + radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            } else if (get_x() >= position_range - radius) {
                flag = true;
                get_x() = position_range - radius - radius * COLLISION_RATIO;
                get_vx() = -get_vx();
            }

            if (get_y() <= radius) {
                flag = true;
                get_y() = radius + radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            } else if (get_y() >= position_range - radius) {
                flag = true;
                get_y() = position_range - radius - radius * COLLISION_RATIO;
                get_vy() = -get_vy();
            }
            if (flag) {
                get_ax() = 0;
                get_ay() = 0;
            }
        }

        __device__ void update_for_tick(
                double elapse,
                double position_range,
                double radius) {
            get_vx() += get_ax() * elapse;
            get_vy() += get_ay() * elapse;
            handle_wall_collision(position_range, radius);
            get_x() += get_vx() * elapse;
            get_y() += get_vy() * elapse;
            handle_wall_collision(position_range, radius);
        }

    };

    BodyPool(size_t size, double position_range, double mass_range) {
        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        x = new double[size];
        y = new double[size];
        vx = new double[size]; 
        vy = new double[size];
        ax = new double[size];
        ay = new double[size];
        m = new double[size];
        for (int i = 0; i < size; i++) {
            x[i] = position_dist(engine);
        }
        for (int i = 0; i < size; i++) {
            y[i] = position_dist(engine);
        }
        for (int i = 0; i < size; i++) {
            m[i] = mass_dist(engine);
        }
    }

    BodyPool(size_t size, size_t vec_size, double position_range, double mass_range) {
        std::random_device device;
        std::default_random_engine engine{device()};
        std::uniform_real_distribution<double> position_dist{0, position_range};
        std::uniform_real_distribution<double> mass_dist{0, mass_range};
        x = new double[vec_size];
        y = new double[vec_size];
        vx = new double[vec_size]; 
        vy = new double[vec_size];
        ax = new double[vec_size];
        ay = new double[vec_size];
        m = new double[vec_size];
        for (int i = 0; i < size; i++) {
            x[i] = position_dist(engine);
        }
        for (int i = 0; i < size; i++) {
            y[i] = position_dist(engine);
        }
        for (int i = 0; i < size; i++) {
            m[i] = mass_dist(engine);
        }
        actual_size = size;
    }

    __device__ Body get_body(size_t index) {
        return {index, *this};
    }

    __device__ void clear_acceleration() {
        for (int i = 0; i < size(); i++) {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }
    }

    __device__ size_t size() {
        return actual_size;
    }

    __device__ void check_and_update_pthread_collision(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        auto dot_prod = delta_x * (i.get_vx() - j.get_vx())
                        + delta_y * (i.get_vy() - j.get_vy());
        auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
        i.get_vx() -= scalar * delta_x * j.get_m();
        i.get_vy() -= scalar * delta_y * j.get_m();
        j.get_vx() += scalar * delta_x * i.get_m();
        j.get_vy() += scalar * delta_y * i.get_m();
        // now relax the distance a bit: after the collision, there must be
        // at least (ratio * radius) between them
        i.get_x() += delta_x / distance * ratio * radius / 2.0;
        i.get_y() += delta_y / distance * ratio * radius / 2.0;
        j.get_x() -= delta_x / distance * ratio * radius / 2.0;
        j.get_y() -= delta_y / distance * ratio * radius / 2.0;
    }

    __device__ void check_and_update_pthread_no_collision(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        auto scalar = gravity / distance_square / distance;
        i.get_ax() -= scalar * delta_x * j.get_m();
        i.get_ay() -= scalar * delta_y * j.get_m();
    }

    __device__ void check_and_update(Body i, Body j, double radius, double gravity) {
        auto delta_x = i.delta_x(j);
        auto delta_y = i.delta_y(j);
        auto distance_square = i.distance_square(j);
        auto ratio = 1 + COLLISION_RATIO;
        if (distance_square < radius * radius) {
            distance_square = radius * radius;
        }
        auto distance = i.distance(j);
        if (distance < radius) {
            distance = radius;
        }
        if (i.collide(j, radius)) {
            auto dot_prod = delta_x * (i.get_vx() - j.get_vx())
                            + delta_y * (i.get_vy() - j.get_vy());
            auto scalar = 2 / (i.get_m() + j.get_m()) * dot_prod / distance_square;
            i.get_vx() -= scalar * delta_x * j.get_m();
            i.get_vy() -= scalar * delta_y * j.get_m();
            j.get_vx() += scalar * delta_x * i.get_m();
            j.get_vy() += scalar * delta_y * i.get_m();
            // now relax the distance a bit: after the collision, there must be
            // at least (ratio * radius) between them
            i.get_x() += delta_x / distance * ratio * radius / 2.0;
            i.get_y() += delta_y / distance * ratio * radius / 2.0;
            j.get_x() -= delta_x / distance * ratio * radius / 2.0;
            j.get_y() -= delta_y / distance * ratio * radius / 2.0;
        } else {
            // update acceleration only when no collision
            auto scalar = gravity / distance_square / distance;
            i.get_ax() -= scalar * delta_x * j.get_m();
            i.get_ay() -= scalar * delta_y * j.get_m();
            j.get_ax() += scalar * delta_x * i.get_m();
            j.get_ay() += scalar * delta_y * i.get_m();
        }
    }

    __device__ void update_for_tick(double elapse,
                         double gravity,
                         double position_range,
                         double radius) {
        for (int i = 0; i < size(); i++) {
            ax[i] = 0.0;
            ay[i] = 0.0;
        }
        for (size_t i = 0; i < size(); ++i) {
            for (size_t j = i + 1; j < size(); ++j) {
                check_and_update(get_body(i), get_body(j), radius, gravity);
            }
        }
        for (size_t i = 0; i < size(); ++i) {
            get_body(i).update_for_tick(elapse, position_range, radius);
        }
    }
};

