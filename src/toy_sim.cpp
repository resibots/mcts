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
        double th = gaussian_rand(best_action(), 0.1);
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
        // I should find the bug here - if any
        double th = std::atan2(global::goal_y, global::goal_x) - std::atan2(_y, _x);
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

        if ((dx * dx + dy * dy) < _epsilon)
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

struct ValueFunction {
    template <typename State>
    double operator()(std::shared_ptr<State> state, double action)
    {
        State tmp = state->move(action);
        if (tmp.terminal())
            return 10.0;
        return -1.0;
    }
};

namespace mcts {
    struct SPWSelectPolicy {
        const double _a = 0.5;

        template <typename Node>
        bool operator()(const std::shared_ptr<Node>& node)
        {
            if (node->visits() == 0 || std::pow((double)node->visits(), _a) > node->children().size())
                return true;
            return false;
        }
    };

    struct ContOutcomeSelect {
        const double _b = 0.5;

        template <typename Action>
        auto operator()(const std::shared_ptr<Action>& action) -> std::shared_ptr<typename std::remove_reference<decltype(*(action->parent()))>::type>
        {
            using NodeType = typename std::remove_reference<decltype(*(action->parent()))>::type;

            if (action->visits() == 0 || std::pow((double)action->visits(), _b) > action->children().size()) {
                auto st = action->parent()->state()->move(action->action());
                auto to_add = std::make_shared<NodeType>(st, action->parent()->rollout_depth(), action->parent()->gamma());
                auto it = std::find_if(action->children().begin(), action->children().end(), [&](std::shared_ptr<NodeType> const& p) { return *(p->state()) == *(to_add->state()); });
                if (action->children().size() == 0 || it == action->children().end()) {
                    to_add->parent() = action;
                    action->children().push_back(to_add);
                    return to_add;
                }

                return (*it);
            }

            // Choose child with probability: n(c)/Sum(n(c'))
            size_t sum = 0;
            for (size_t i = 0; i < action->children().size(); i++) {
                sum += action->children()[i]->visits();
            }
            size_t r = static_cast<size_t>(std::rand() * double(sum) / double(RAND_MAX));
            size_t p = 0;
            for (auto child : action->children()) {
                p += child->visits();
                if (r <= p)
                    return child;
            }

            // we should never reach here
            assert(false);
            return nullptr;
        }
    };

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

    global::goal_x = 0.2;
    global::goal_y = 0.0;

    ValueFunction world;
    SimpleState init(0.0, 0.0);

    auto tree = std::make_shared<mcts::MCTSNode<SimpleState, mcts::SimpleStateInit, mcts::SimpleValueInit, mcts::UCTValue, mcts::BestHeuristicPolicy<SimpleState, double>, double, mcts::SPWSelectPolicy, mcts::ContOutcomeSelect>>(init, 2000);
    const int n_iter = 10000;
    int k;
    for (k = 0; k < n_iter; ++k) {
        tree->iterate(world);
    }

    auto best = tree->best_action();
    if (best == nullptr)
        std::cout << init._x << " " << init._y << ": Terminal!" << std::endl;
    else
        std::cout << init._x << " " << init._y << ": " << best->action() << " -> " << init.move(best->action(), false)._x << " " << init.move(best->action(), false)._y << std::endl;

    return 0;
}
