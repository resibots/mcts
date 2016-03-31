#ifndef UCT_HPP
#define UCT_HPP

#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

// class MCTSNode {
//     double search(node_ptr node, size_t depth)
//     {
//         if (node->terminal())
//             return 0;
//         if (node->leaf())
//             return _simulation(node);
//         action_ptr next_action = node->_select_action(depth);
//         std::pair<node_ptr, double> state_reward = node->_model_simulation(next_action);
//         double q = std::get<1>(state_reward) + _gamma * search(std::get<0>(state_reward), depth + 1);
//         node->_update(next_action, q, depth);
//         return q;
//     }
// };

namespace mcts {

    struct SimpleStateInit {
        template <typename State>
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

    struct UCTValue {
        const double _c = 10.0; //1.0 / std::sqrt(2.0);
        const double _epsilon = 1e-6;

        template <typename MCTSAction>
        double operator()(const std::shared_ptr<MCTSAction>& action)
        {
            // return action->value() / (double(action->visits()) + _epsilon) + _c * std::sqrt(2.0 * std::log(action->parent()->visits() + 1.0) / (double(action->visits()) + _epsilon));
            return action->value() / (double(action->visits()) + _epsilon) + 2.0 * _c * std::sqrt(std::log(action->parent()->visits() + 1.0) / (double(action->visits()) + _epsilon));
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

    template <typename NodeType, typename ActionValue, typename ActionType = size_t>
    class MCTSAction : public std::enable_shared_from_this<MCTSAction<NodeType, ActionValue, ActionType>> {
    public:
        using node_ptr = std::shared_ptr<NodeType>;

        MCTSAction(const ActionType& action, const node_ptr& parent) : _parent(parent), _action(action), _visits(0) {}

        node_ptr parent() const
        {
            return _parent;
        }

        std::vector<node_ptr> children() const
        {
            return _children;
        }

        ActionType action() const
        {
            return _action;
        }

        size_t visits() const
        {
            return _visits;
        }

        size_t& visits()
        {
            return _visits;
        }

        double value() const
        {
            return _value;
        }

        double& value()
        {
            return _value;
        }

        bool operator==(const MCTSAction& other) const
        {
            return _action == other._action;
        }

        node_ptr node()
        {
            auto st = _parent->state()->move(_action);
            node_ptr to_add = std::make_shared<NodeType>(st, _parent->rollout_depth(), _parent->gamma());
            auto it = std::find_if(_children.begin(), _children.end(), [&](node_ptr const& p) { return *(p->state()) == *(to_add->state()); });
            if (it == _children.end()) {
                to_add->parent() = this->shared_from_this();
                _children.push_back(to_add);
                return to_add;
            }

            return (*it);
        }

        void update_stats(double value)
        {
            _value += value;
            _visits++;
        }

    protected:
        node_ptr _parent;
        std::vector<node_ptr> _children;
        ActionType _action;
        double _value;
        size_t _visits;
    };

    template <typename State, typename StateInit, typename ValueInit, typename ActionValue, typename DefaultPolicy, typename Action>
    class MCTSNode : public std::enable_shared_from_this<MCTSNode<State, StateInit, ValueInit, ActionValue, DefaultPolicy, Action>> {
    public:
        using node_type = MCTSNode<State, StateInit, ValueInit, ActionValue, DefaultPolicy, Action>;
        using action_type = MCTSAction<node_type, ActionValue, Action>;
        using action_ptr = std::shared_ptr<action_type>;
        using node_ptr = std::shared_ptr<node_type>;
        using state_ptr = std::shared_ptr<State>;

        MCTSNode(size_t rollout_depth = 1000, double gamma = 0.9) : _gamma(gamma), _visits(0), _rollout_depth(rollout_depth)
        {
            _state = std::make_shared<State>(StateInit()());
            _value = ValueInit()(_state);
        }

        MCTSNode(State state, size_t rollout_depth = 1000, double gamma = 0.9) : _gamma(gamma), _visits(0), _rollout_depth(rollout_depth)
        {
            _state = std::make_shared<State>(state);
            _value = ValueInit()(_state);
        }

        action_ptr parent() const
        {
            return _parent;
        }

        action_ptr& parent()
        {
            return _parent;
        }

        std::vector<action_ptr> children() const
        {
            return _children;
        }

        state_ptr state() const
        {
            return _state;
        }

        size_t visits() const
        {
            return _visits;
        }

        size_t& visits()
        {
            return _visits;
        }

        size_t rollout_depth() const
        {
            return _rollout_depth;
        }

        double gamma() const
        {
            return _gamma;
        }

        double value() const
        {
            return _value;
        }

        template <typename ValueFunc>
        void iterate(ValueFunc vfun)
        {
            std::vector<node_ptr> visited;
            std::vector<double> rewards;

            node_ptr cur_node = this->shared_from_this();
            visited.push_back(cur_node);
            rewards.push_back(0.0);
            // std::cout << "Iterate!" << std::endl;

            while (!cur_node->_state->terminal() && cur_node->fully_expanded()) {
                // std::cout << "(" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
                action_ptr next_action = cur_node->_select_action();
                // std::cout << "Selected action: " << next_action->action() << std::endl;
                rewards.push_back(vfun(cur_node->_state, next_action->action()));
                cur_node = next_action->node();
                visited.push_back(cur_node);
            }

            // std::cout << "Expanding (" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
            double value;
            if (cur_node->_state->terminal()) {
                value = 0.0;
            }
            else {
                cur_node->_expand();
                action_ptr next_action = cur_node->_select_action();
                // std::cout << "Best action: " << next_action->action() << std::endl;
                rewards.push_back(vfun(cur_node->_state, next_action->action()));
                cur_node = next_action->node();
                visited.push_back(cur_node);

                // std::cout << "Simulating: (" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
                value = cur_node->_simulate(vfun);
            }

            for (int i = visited.size() - 1; i >= 0; i--) {
                value = rewards[i] + _gamma * value;
                visited[i]->_update_stats(value);
                if (visited[i]->_parent != nullptr)
                    visited[i]->_parent->update_stats(value);
            }
        }

        template <typename Value = GreedyValue>
        action_ptr best_action()
        {
            if (_state->terminal())
                return nullptr;
            double v = -std::numeric_limits<double>::max();
            action_ptr best_action = nullptr;

            for (auto child : _children) {
                double d = Value()(child);

                if (d > v) {
                    v = d;
                    best_action = child;
                }
            }

            return best_action;
        }

        bool fully_expanded() const
        {
            return !_state->has_actions();
        }

        // void print(size_t d = 0) const
        // {
        //     std::cout << d << ": " << _state->_x << " " << _state->_y << " -> " << _value << ", " << _visits; // << std::endl;
        //     if (_parent != nullptr)
        //         std::cout << " act: " << _parent->action();
        //     std::cout << std::endl;
        //     for (size_t i = 0; i < _children.size(); i++) {
        //         for (size_t k = 0; k < _children[i]->children().size(); k++) {
        //             _children[i]->children()[k]->print(d + 1);
        //         }
        //     }
        // }

    protected:
        action_ptr _parent;
        std::vector<action_ptr> _children;
        state_ptr _state;
        double _value, _gamma;
        size_t _visits, _rollout_depth;

        action_ptr _expand()
        {
            Action act = _state->next_action();
            action_ptr next_action = std::make_shared<action_type>(act, this->shared_from_this()); // i am not sure about this!?
            auto it = std::find_if(_children.begin(), _children.end(), [&](action_ptr const& p) { return *p == *next_action; });
            if (_children.size() == 0 || it == _children.end()) {
                _children.push_back(next_action);
                it = _children.end();
                it--;
            }

            return (*it);
        }

        action_ptr _select_action()
        {
            if (_state->terminal())
                return nullptr;
            double v = -std::numeric_limits<double>::max();
            action_ptr best_action = nullptr;

            for (auto child : _children) {
                double d = ActionValue()(child);

                if (d > v) {
                    v = d;
                    best_action = child;
                }
            }

            return best_action;
        }

        template <typename ValueFunc>
        double _simulate(ValueFunc vfun)
        {
            double discount = 1.0;
            double reward = 0.0;

            state_ptr cur_state = _state;

            for (size_t k = 0; k < _rollout_depth; ++k) {
                // Choose action according to default policy
                Action action = DefaultPolicy()(cur_state);

                // Get value from (PO)MDP
                reward += discount * vfun(cur_state, action);

                // Update state
                cur_state = std::make_shared<State>(cur_state->move(action));

                // Check if terminal state
                if (cur_state->terminal())
                    break;
                discount *= _gamma;
            }

            return reward;
        }

        void _update_stats(double value)
        {
            _value += value;
            _visits++;
        }
    };
}

#endif
