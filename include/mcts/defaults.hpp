#ifndef MCTS_DEFAULTS_HPP
#define MCTS_DEFAULTS_HPP

namespace mcts {

    template <typename State>
    struct SimpleStateInit {
        std::shared_ptr<State> operator()()
        {
            // assumes the default constructor of State is the init state
            return std::make_shared<State>();
        }
    };

    struct SimpleValueInit {
        template <typename State>
        double operator()(const std::shared_ptr<State>& state)
        {
            return 0.0;
        }
    };

    struct SimpleSelectPolicy {
        template <typename Node>
        bool operator()(const std::shared_ptr<Node>& node)
        {
            return true;
        }
    };

    struct SimpleOutcomeSelect {
        template <typename MCTSAction>
        auto operator()(const std::shared_ptr<MCTSAction>& action) -> std::shared_ptr<typename std::remove_reference<decltype(*(action->parent()))>::type>
        {
            using NodeType = typename std::remove_reference<decltype(*(action->parent()))>::type;
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
    };

    template <typename Params>
    struct UCTValue {
        // c parameter in Params struct
        const double _epsilon = 1e-6;

        template <typename MCTSAction>
        double operator()(const std::shared_ptr<MCTSAction>& action)
        {
            // return action->value() / (double(action->visits()) + _epsilon) + _c * std::sqrt(2.0 * std::log(action->parent()->visits() + 1.0) / (double(action->visits()) + _epsilon));
            return action->value() / (double(action->visits()) + _epsilon) + 2.0 * Params::uct::c() * std::sqrt(std::log(action->parent()->visits() + 1.0) / (double(action->visits()) + _epsilon));
        }
    };

    struct GreedyValue {
        const double _epsilon = 1e-6;

        template <typename MCTSAction>
        double operator()(const std::shared_ptr<MCTSAction>& action)
        {
            return action->value() / (double(action->visits()) + _epsilon);
        }
    };

    template <typename State, typename Action>
    struct UniformRandomPolicy {
        Action operator()(const std::shared_ptr<State>& state)
        {
            return state->random_action();
        }
    };

    template <typename Params>
    struct SPWSelectPolicy {
        // a parameter in Params struct

        template <typename Node>
        bool operator()(const std::shared_ptr<Node>& node)
        {
            if (node->visits() == 0 || std::pow((double)node->visits(), Params::spw::a()) > node->children().size())
                return true;
            return false;
        }
    };

    template <typename Params>
    struct ContinuousOutcomeSelect {
        // b parameter in Params struct

        template <typename Action>
        auto operator()(const std::shared_ptr<Action>& action) -> std::shared_ptr<typename std::remove_reference<decltype(*(action->parent()))>::type>
        {
            using NodeType = typename std::remove_reference<decltype(*(action->parent()))>::type;

            if (action->visits() == 0 || std::pow((double)action->visits(), Params::cont_outcome::b()) > action->children().size()) {
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
}

#endif
