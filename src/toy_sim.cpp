#include <iostream>
#include <ctime>
#include <mcts/uct.hpp>

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<T> gaussian(m, v);

    return gaussian(gen);
}

struct Params {
    struct uct {
        MCTS_PARAM(double, c, 50.0);
    };

    struct spw {
        MCTS_PARAM(double, a, 0.5);
    };

    struct cont_outcome {
        MCTS_PARAM(double, b, 0.6);
    };

    struct mcts_node {
#ifdef SINGLE
        MCTS_PARAM(size_t, parallel_roots, 1);
#else
        MCTS_PARAM(size_t, parallel_roots, 4);
#endif
    };
};

namespace global {
    double goal_x, goal_y;
}

struct SimpleState {
    double _x, _y;
    const double _epsilon = 1e-6;

    SimpleState()
    {
        _x = _y = 0;
    }

    SimpleState(double x, double y)
    {
        _x = x;
        _y = y;
    }

    double next_action() const
    {
        // using domain knowledge - have to check literature
        double th = gaussian_rand(best_action(), 0.3);
        if (th > M_PI)
            th -= 2 * M_PI;
        if (th < -M_PI)
            th += 2 * M_PI;
        return th;
    }

    double random_action() const
    {
        return (std::rand() * 2.0 * M_PI / double(RAND_MAX) - M_PI);
    }

    double best_action() const
    {
        double th = std::atan2(global::goal_y - _y, global::goal_x - _x);
        if (th > M_PI)
            th -= 2 * M_PI;
        if (th < -M_PI)
            th += 2 * M_PI;
        return th;
    }

    SimpleState move(double theta, bool prob = true) const
    {
        double r = 0.1;
        double th = theta;
        if (prob) {
            double p = std::rand() / double(RAND_MAX);
            if (p < 0.2) {
                th += 0.1;
                if (th > M_PI)
                    th -= 2 * M_PI;
                if (th < -M_PI)
                    th += 2 * M_PI;
            }
        }
        double s = std::sin(th), c = std::cos(th);
        double x_new = r * c + _x, y_new = r * s + _y;

        return SimpleState(x_new, y_new);
    }

    bool terminal() const
    {
        double dx = _x - global::goal_x;
        double dy = _y - global::goal_y;

        if ((dx * dx + dy * dy) < 0.01)
            return true;
        return false;
    }

    bool operator==(const SimpleState& other) const
    {
        double dx = _x - other._x;
        double dy = _y - other._y;
        return ((dx * dx + dy * dy) < _epsilon);
    }
};

struct RewardFunction {
    template <typename State>
    double operator()(std::shared_ptr<State> from_state, double action, std::shared_ptr<State> to_state)
    {
        if (to_state->terminal())
            return 10.0;
        return -1.0;
    }
};

namespace mcts {
    template <typename State, typename Action>
    struct BestHeuristicPolicy {
        Action operator()(const std::shared_ptr<State>& state)
        {
            return state->best_action();
        }
    };
}

int main()
{
    std::srand(std::time(0));
    mcts::par::init();

    global::goal_x = 2.0;
    global::goal_y = 2.0;

    RewardFunction world;
    SimpleState init(0.0, 0.0);

    auto tree = std::make_shared<mcts::MCTSNode<Params, SimpleState, mcts::SimpleStateInit<SimpleState>, mcts::SimpleValueInit, mcts::UCTValue<Params>, mcts::BestHeuristicPolicy<SimpleState, double>, double, mcts::SPWSelectPolicy<Params>, mcts::ContinuousOutcomeSelect<Params>>>(init, 2000);
#ifdef SINGLE
    const int n_iter = 400000;
#else
    const int n_iter = 200000;
#endif

    auto t1 = std::chrono::steady_clock::now();

    tree->compute(world, n_iter);

    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout << "Time in sec: " << time_running / 1000.0 << std::endl;

    auto best = tree->best_action();
    if (best == nullptr)
        std::cout << init._x << " " << init._y << ": Terminal!" << std::endl;
    else
        std::cout << init._x << " " << init._y << ": " << best->action() << " -> " << init.move(best->action(), false)._x << " " << init.move(best->action(), false)._y << std::endl;

    return 0;
}
