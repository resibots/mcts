#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mcts/mcts_node.hpp>

#define SIZE 3
#define GOAL SIZE
#define PROB 0.0

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

        GridState move_with(size_t action)
        {
            int x_new = _x, y_new = _y;

            double r = std::rand() / (double)RAND_MAX;
            if (r < PROB)
                action = (action + 1) % 4;

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

    const size_t N_ACTIONS = 4;
    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            GridState init(i, j, SIZE);
            auto tree = std::make_shared<mcts::MCTSNode<GridState, mcts::UCTValue, mcts::UniformRandomPolicy>>(N_ACTIONS, init, 10000);
            const int N_ITERATIONS = 1000;
            for (int k = 0; k < N_ITERATIONS; ++k) {
                tree->iterate(world);
            }
            size_t bestAction = tree->best_action();
            std::cout << init._x << " " << init._y << ": " << bestAction << std::endl;
        }
    }
    // GridState init(0, 0, SIZE);
    // int bestAction;
    // size_t n = 0;
    //
    // while ((init._x != (GOAL - 1)) || (init._y != (GOAL - 1))) {
    //     auto tree = std::make_shared<mcts::MCTSNode<GridState, mcts::UCTValue, mcts::UniformRandomPolicy>>(N_ACTIONS, init, 20);
    //
    //     const int N_ITERATIONS = 1000;
    //     for (int k = 0; k < N_ITERATIONS; ++k) {
    //         tree->iterate(world);
    //     }
    //
    //     bestAction = tree->best_action();
    //
    //     init = init.move_with(bestAction);
    //     n++;
    //     // std::cout << init._x << " " << init._y << std::endl;
    // }
    // std::cout << n << std::endl;
    //
    // std::ofstream file("results.txt");
    // file << n << std::endl;
    // file.close();

    return 0;
}
