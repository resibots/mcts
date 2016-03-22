#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mcts/mcts_node.hpp>

#define SIZE 10

namespace {

    struct GridWorld {
        template <typename State>
        double operator()(State state, size_t action)
        {
            State tmp = state.move_with(action);
            // std::cout << tmp._x << " " << tmp._y << std::endl;
            if (tmp._x == (SIZE - 1) && tmp._y == (SIZE - 1)) {
                // std::cout << "Goal!" << std::endl;
                return 1.0;
            }
            return -1.0;
        }

        template <typename State>
        bool final(const State& state)
        {
            if (state._x == (SIZE - 1) && state._y == (SIZE - 1)) {
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

        GridState move_with(size_t action)
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

            // std::cout << _x << " " << _y << " -> " << x_new << " " << y_new << " with action " << action << std::endl;

            return GridState(x_new, y_new, _N);
        }
    };
}

int main()
{
    std::srand(std::time(0));

    GridWorld world;

    const size_t N_ACTIONS = 4;
    GridState init(0, 0, SIZE);
    int bestAction;
    size_t n = 0;

    while ((init._x != (SIZE - 1)) || (init._y != (SIZE - 1))) {
        mcts::MCTSNode<GridState, mcts::UCTValue, mcts::UniformRandomPolicy> tree(N_ACTIONS, init, 20);

        const int N_ITERATIONS = 1000;
        for (int k = 0; k < N_ITERATIONS; ++k) {
            //  std::cout << "tree: " << tree << std::endl;
            // std::cout << "iteration: " << k << std::endl;
            tree.iterate(world);
        }
        // std::cout << "tree: " << tree << std::endl;

        bestAction = tree.best_action();

        init = init.move_with(bestAction);
        n++;
        // std::cout << init._x << " " << init._y << std::endl;
    }

    std::ofstream file("results.txt");
    file << n << std::endl;
    file.close();
    // std::cout << "Best Action: " << bestAction << std::endl;
    // int nNodes = tree.nodes();
    // std::cout << "Number of Nodes: " << nNodes << std::endl;
    // int maxDepth = tree.max_depth();
    // std::cout << "Max Depth: " << maxDepth << std::endl;
}