#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mcts/value_iteration.hpp>

#define SIZE 5
#define GOAL SIZE
#define PROB 0.2

namespace {

    struct GridWorld {
        template <typename State>
        double operator()(State state, size_t action)
        {
            State tmp = state.move_with(action);
            if (tmp._x == (GOAL - 1) && tmp._y == (GOAL - 1))
                return 1.0;

            return 0.0;
        }

        template <typename State>
        double operator()(State init_state, State final_state, size_t action)
        {
            if (init_state.can_reach(final_state, action) && final_state._x == (GOAL - 1) && final_state._y == (GOAL - 1))
                return 1.0;

            return 0.0;
        }

        template <typename State>
        bool final(const State& state)
        {
            if (state._x == (GOAL - 1) && state._y == (GOAL - 1)) {
                return true;
            }
            return false;
        }
    };

    struct GridState {
        size_t _x, _y, _N;

        GridState()
        {
            _x = _y = 0;
            _N = 5;
        }

        GridState(size_t x, size_t y, size_t N)
        {
            _x = x;
            _y = y;
            _N = N;
        }

        bool can_reach(const GridState& other, size_t action, bool prop = true)
        {
            int x_new = _x, y_new = _y;

            if (action == 0) // up
            {
                y_new++;
                if (y_new >= (int)_N)
                    y_new--;
            }
            else if (action == 1) // down
            {
                y_new--;
                if (y_new < 0)
                    y_new++;
            }
            else if (action == 2) // right
            {
                x_new++;
                if (x_new >= (int)_N)
                    x_new--;
            }
            else if (action == 3) // left
            {
                x_new--;
                if (x_new < 0)
                    x_new++;
            }

            size_t new_action = (action + 1) % 4;
            GridState tmp = GridState(x_new, y_new, _N);
            return (tmp == other || (prop && can_reach(other, new_action, false)));
        }

        bool valid(size_t action)
        {
            int x_new = _x, y_new = _y;
            if (action == 0) // up
            {
                y_new++;
                if (y_new >= (int)_N)
                    return false;
            }
            else if (action == 1) // down
            {
                y_new--;
                if (y_new < 0)
                    return false;
            }
            else if (action == 2) // right
            {
                x_new++;
                if (x_new >= (int)_N)
                    return false;
            }
            else if (action == 3) // left
            {
                x_new--;
                if (x_new < 0)
                    return false;
            }
            return true;
        }

        GridState move_with(size_t action, bool prob = true)
        {
            int x_new = _x, y_new = _y;
            if (prob) {
                double r = std::rand() / (double)RAND_MAX;
                if (r < PROB)
                    action = (action + 1) % 4;
            }

            if (action == 0) // up
            {
                y_new++;
                if (y_new >= (int)_N)
                    y_new--;
            }
            else if (action == 1) // down
            {
                y_new--;
                if (y_new < 0)
                    y_new++;
            }
            else if (action == 2) // right
            {
                x_new++;
                if (x_new >= (int)_N)
                    x_new--;
            }
            else if (action == 3) // left
            {
                x_new--;
                if (x_new < 0)
                    x_new++;
            }

            return GridState(x_new, y_new, _N);
        }

        bool operator==(const GridState& other)
        {
            assert(_N == other._N);
            return (_x == other._x && _y == other._y);
        }
    };
}

int main()
{
    std::srand(std::time(0));

    GridWorld world;

    size_t N_ACTIONS = 4;

    std::vector<GridState> states;
    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            states.push_back(GridState(i, j, SIZE));
        }
    }

    std::vector<std::vector<std::vector<double>>> transitions(states.size(), std::vector<std::vector<double>>(states.size(), std::vector<double>(N_ACTIONS, 0.0)));
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t j = 0; j < states.size(); j++) {
            for (size_t a = 0; a < N_ACTIONS; a++) {
                // std::cout << "Setting for " << states[i]._x << " " << states[i]._y << " -> " << states[j]._x << " " << states[j]._y << " with action " << a << std::endl;
                GridState new_state = states[i].move_with(a, false);
                GridState new_state2 = states[i].move_with((a + 1) % N_ACTIONS, false);
                // std::cout << "new_state " << new_state._x << " " << new_state._y << std::endl;
                // std::cout << "new_state2 " << new_state2._x << " " << new_state2._y << std::endl;
                if (new_state == states[j])
                    transitions[i][j][a] = 1.0 - PROB;
                else if (new_state2 == states[j])
                    transitions[i][j][a] = PROB;
                else
                    transitions[i][j][a] = 0.0;
                // std::cout << "value: " << transitions[i][j][a] << std::endl;
                // std::cin.get();
            }
        }
    }

    // NORMALIZE PROBABILITIES
    for (size_t i = 0; i < states.size(); i++) {
        for (size_t a = 0; a < N_ACTIONS; a++) {
            double sum = 0.0;
            for (size_t j = 0; j < states.size(); j++) {
                sum += transitions[i][j][a];
            }
            if (sum > 1e-6 && sum < 1.0) {
                for (size_t j = 0; j < states.size(); j++) {
                    transitions[i][j][a] /= sum;
                }
            }
        }
    }

    // for (size_t i = 0; i < states.size(); i++) {
    //     std::cout << states[i]._x << " " << states[i]._y << ": " << std::endl;
    //     for (size_t j = 0; j < states.size(); j++) {
    //         std::cout << states[j]._x << " " << states[j]._y << ": ";
    //         for (size_t a = 0; a < N_ACTIONS; a++) {
    //             std::cout << a << " -> " << transitions[i][j][a] << ", ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    ValueIteration<GridState> vi(states, transitions);
    vi.solve(world);

    std::cout << "POLICY: " << std::endl;
    auto p = vi.policy();
    for (size_t i = 0; i < p.size(); i++) {
        std::cout << states[i]._x << " " << states[i]._y << ": " << p[i] << std::endl;
    }

    // std::cout << "VALUE: " << std::endl;
    // auto v = vi.value();
    // for (size_t i = 0; i < v.size(); i++) {
    //     std::cout << states[i]._x << " " << states[i]._y << ": " << v[i] << std::endl;
    // }

    // std::ofstream file("results.txt");
    // file << n << std::endl;
    // file.close();

    return 0;
}
