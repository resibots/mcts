#include <iostream>
#include <ctime>
#include <fstream>
#include <mcts/uct.hpp>
#include <chrono>

size_t GOAL;

struct Params {
    struct uct {
        MCTS_PARAM(double, c, 10.0);
    };

    struct mcts_node {
        MCTS_PARAM(size_t, parallel_roots, 1);
    };
};

struct GridState {
    size_t _x, _y, _N;
    double _prob;
    std::vector<size_t> _used_actions;

    GridState()
    {
        _x = _y = 0;
        _N = 10;
        _prob = 0.0;
    }

    GridState(size_t x, size_t y, size_t N, double prob)
    {
        _x = x;
        _y = y;
        _N = N;
        _prob = prob;
    }

    bool valid(size_t action) const
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

    size_t next_action()
    {
        return random_action();
    }

    GridState move(size_t action, bool prob = true) const
    {
        int x_new = _x, y_new = _y;

        double r = std::rand() / (double)RAND_MAX;
        if ((r - _prob) < 0 && prob)
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
        // if (r < PROB)
        //     return GridState(_x, _y, _N);

        return GridState(x_new, y_new, _N, _prob);
    }

    size_t random_action() const
    {
        size_t act;
        do {
            act = static_cast<size_t>(std::rand() * 4.0 / (double)RAND_MAX);
        } while (!valid(act));

        return act;
    }

    size_t best_action() const
    {
        size_t act = 0;
        double v = std::numeric_limits<double>::max();
        for (size_t i = 0; i < 4; i++) {
            if (!valid(i))
                continue;
            GridState tmp = move(i, false);
            double dx = tmp._x - GOAL + 1;
            double dy = tmp._y - GOAL + 1;
            double d = dx * dx + dy * dy;
            if (d < v) {
                act = i;
                v = d;
            }
        }

        return act;
    }

    bool terminal() const
    {
        if (_x == (GOAL - 1) && _y == (GOAL - 1))
            return true;
        return false;
    }

    bool operator==(const GridState& other) const
    {
        assert(_N == other._N);
        return (_x == other._x && _y == other._y);
    }
};

struct GridWorld {
    template <typename State>
    double operator()(std::shared_ptr<State> from_state, size_t action, std::shared_ptr<State> to_state)
    {
        if (to_state->_x == (GOAL - 1) && to_state->_y == (GOAL - 1))
            return max_reward();

        return min_reward();
    }

    double max_reward()
    {
        return 1.0;
    }

    double min_reward()
    {
        return 0.0;
    }
};

template <typename State, typename Action>
struct BestHeuristicPolicy {
    Action operator()(const std::shared_ptr<State>& state)
    {
        return state->best_action();
    }
};

int main()
{
    std::srand(std::time(0));
    mcts::par::init();

    GridWorld world;

    for (size_t s = 5; s <= 40; s += 5) {

        GOAL = s;

        std::ofstream file("results_" + std::to_string(s) + ".txt");

        for (double p = 0.0; p <= 0.4; p += 0.1) {

            size_t c = 0;
            size_t avg = 0;
            double avg_time = 0.0;

            for (size_t i = 0; i < s; i++) {
                for (size_t j = 0; j < s; j++) {
                    auto t1 = std::chrono::steady_clock::now();
                    GridState init(i, j, s, p);
                    auto tree = std::make_shared<mcts::MCTSNode<Params, GridState, mcts::SimpleStateInit<GridState>, mcts::SimpleValueInit, mcts::UCTValue<Params>, BestHeuristicPolicy<GridState, size_t>, size_t, mcts::SimpleSelectPolicy, mcts::SimpleOutcomeSelect>>(init, 10000);
                    const int N_ITERATIONS = 10000;
                    const int MIN_ITERATIONS = 1000;
                    int k;
                    for (k = 0; k < N_ITERATIONS; ++k) {
                        tree->iterate(world);
                        if (k >= MIN_ITERATIONS) {
                            auto best = tree->best_action();
                            if (best != nullptr && (best->action() == 0 || best->action() == 2)) {
                                if (!(init._x == (s - 1) && best->action() != 0) && !(init._y == (s - 1) && best->action() != 2))
                                    break;
                            }
                        }
                    }
                    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
                    avg_time += time_running / 1000.0;
                    avg += k;
                    // tree->print();
                    // std::cout << "------------------------" << std::endl;
                    auto best = tree->best_action();
                    if (best == nullptr && !init.terminal())
                        c++;
                    if (best != nullptr && best->action() != 0 && best->action() != 2)
                        c++;
                    // if (best == nullptr)
                    //     std::cout << init._x << " " << init._y << ": Terminal!" << std::endl;
                    // else {
                    //     if (best->action() != 0 && best->action() != 2)
                    //         c++;
                    //     std::cout << init._x << " " << init._y << ": " << best->action() << std::endl;
                    // }
                    // std::cin.get();
                }
            }

            file << c << " " << double(avg) / double(s * s) << " " << avg_time / double(s * s) << std::endl;
            // std::cout << "Errors: " << c << std::endl;
        }
        file.close();
    }
    return 0;
}
