#include <iostream>
#include <ctime>
#include <mcts/uct.hpp>

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
    double a = 70;
    double h = 100;
    double l = 1;
    double w = 0.7;
}

struct SimpleState {
    double _x, _R;
    int _time;
    const double _epsilon = 1e-6;

    SimpleState()
    {
        _x = 0;
        _R = 0.01;
        _time = 0;
    }

    SimpleState(double x, int t = 0, double R = 0.01)
    {
        _x = x;
        _R = R;
        _time = t;
    }

    double next_action() const
    {
        return random_action();
    }

    double random_action() const
    {
        return (std::rand() / double(RAND_MAX));
    }

    SimpleState move(double d) const
    {
        double x_new = _x + d + _R * (std::rand() / double(RAND_MAX));
        return SimpleState(x_new, _time + 1, _R);
    }

    bool terminal() const
    {
        return (_time >= 2);
    }

    bool operator==(const SimpleState& other) const
    {
        double dx = _x - other._x;
        return ((dx * dx) < _epsilon);
    }
};

struct RewardFunction {
    template <typename State>
    double operator()(std::shared_ptr<State> from_state, double action, std::shared_ptr<State> to_state)
    {
        if (to_state->_x < global::l)
            return global::a;
        else if (to_state->_x < (global::l + global::w))
            return 0.0;
        else if (to_state->_x > (global::l + global::w))
            return global::h;
        assert(false);
        return 0.0;
    }
};

int main()
{
    std::srand(std::time(0));
    mcts::par::init();

    RewardFunction world;
    SimpleState init;

#ifdef SIMPLE
    auto tree = std::make_shared<mcts::MCTSNode<Params, SimpleState, mcts::SimpleStateInit<SimpleState>, mcts::SimpleValueInit, mcts::UCTValue<Params>, mcts::UniformRandomPolicy<SimpleState, double>, double, mcts::SPWSelectPolicy<Params>, mcts::SimpleOutcomeSelect>>(init, 2, 1.0);
#else
    auto tree = std::make_shared<mcts::MCTSNode<Params, SimpleState, mcts::SimpleStateInit<SimpleState>, mcts::SimpleValueInit, mcts::UCTValue<Params>, mcts::UniformRandomPolicy<SimpleState, double>, double, mcts::SPWSelectPolicy<Params>, mcts::ContinuousOutcomeSelect<Params>>>(init, 2, 1.0);
#endif

#ifdef SINGLE
    const int n_iter = 50000;
#else
    const int n_iter = 18000;
#endif

    auto t1 = std::chrono::steady_clock::now();

    tree->compute(world, n_iter);

    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout << "Time in sec: " << time_running / 1000.0 << std::endl;

    auto best = tree->best_action();
    if (best != nullptr) {
        std::cout << best->action() << std::endl;
        std::cout << best->value() / best->visits() << std::endl;
        auto new_state = init.move(best->action());
        std::cout << "Moving to: " << new_state._x << std::endl;

        // tree = std::make_shared<mcts::MCTSNode<Params, SimpleState, mcts::SimpleStateInit<SimpleState>, mcts::SimpleValueInit, mcts::UCTValue<Params>, mcts::UniformRandomPolicy<SimpleState, double>, double, mcts::SPWSelectPolicy<Params>, mcts::ContinuousOutcomeSelect<Params>>>(new_state, 2);
        //
        // best = tree->best_action();
        // if (best != nullptr)
        //     std::cout << best->action() << std::endl;
    }

    return 0;
}
