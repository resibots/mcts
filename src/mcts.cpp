#include <iostream>
#include <ctime>
#include <mcts/mcts.hpp>

#define SIZE 5
#define GOAL SIZE
#define PROB 0.1

struct GridState {
    size_t _x, _y, _N;
    std::vector<size_t> _used_actions;

    GridState()
    {
        _x = _y = 0;
        _N = SIZE;
    }

    GridState(size_t x, size_t y, size_t N)
    {
        _x = x;
        _y = y;
        _N = N;
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
        size_t mult = 1; //(PROB > 0.0) ? 2 : 1;

        size_t i;
        while (true) {
            i = static_cast<size_t>(std::rand() * 4.0 / (double)RAND_MAX);
            if (valid(i) && std::count(_used_actions.begin(), _used_actions.end(), i) < (int)mult) {
                _used_actions.push_back(i);
                return i;
            }
        }
        assert(false);
        return 0;
    }

    size_t valid_actions() const
    {
        size_t _valid = 0;
        for (size_t i = 0; i < 4; i++) {
            if (valid(i))
                _valid++;
        }

        return _valid;
    }

    bool has_actions()
    {
        size_t mult = 1; //(PROB > 0.0) ? 2 : 1;
        return _used_actions.size() < valid_actions() * mult;
    }

    GridState move(size_t action) const
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

    size_t random_action() const
    {
        size_t act;
        do {
            act = static_cast<size_t>(std::rand() * 4.0 / (double)RAND_MAX);
        } while (!valid(act));

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
    double operator()(std::shared_ptr<State> state, size_t action)
    {
        State tmp = state->move(action);
        if (tmp._x == (GOAL - 1) && tmp._y == (GOAL - 1))
            return 1.0;

        return 0.0;
    }

    double max_reward()
    {
        return 1.0;
    }
};

int main()
{
    std::srand(std::time(0));

    GridWorld world;

    for (size_t i = 0; i < SIZE; i++) {
        for (size_t j = 0; j < SIZE; j++) {
            GridState init(i, j, SIZE);
            auto tree = std::make_shared<mcts::MCTSNode<GridState, mcts::SimpleStateInit, mcts::SimpleValueInit, mcts::UCTValue, mcts::UniformRandomPolicy<GridState, size_t>, size_t>>(init);
            const int N_ITERATIONS = 10000;
            for (int k = 0; k < N_ITERATIONS; ++k) {
                tree->iterate(world);
                // std::cin.get();
            }
            auto best = tree->best_child();
            if (best == nullptr)
                std::cout << init._x << " " << init._y << ": Terminal!" << std::endl;
            else
                std::cout << init._x << " " << init._y << ": " << best->parent()->action() << std::endl;
        }
    }
    return 0;
}
