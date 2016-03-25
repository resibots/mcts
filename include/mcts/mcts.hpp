#ifndef MCTS_HPP
#define MCTS_HPP

#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

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
        const double _c = 1.0 / std::sqrt(2.0);
        const double _epsilon = 1e-6;

        template <typename Node>
        double operator()(const std::shared_ptr<Node>& node)
        {
            // std::cout << "UCT: " << node->state()->_x << " " << node->state()->_y << std::endl;
            // std::cout << "value: " << node->value() << " visits: " << node->visits() << std::endl;
            // std::cout << "parent: " << node->parent()->action() << std::endl;
            // std::cout << "parent: " << node->parent()->parent()->visits() << std::endl;
            // state->action->state
            return node->value() / (double(node->visits()) + _epsilon) + _c * std::sqrt(2.0 * std::log(node->parent()->parent()->visits()) / (double(node->visits()) + _epsilon));
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

        MCTSAction(const ActionType& action, const node_ptr& parent) : _parent(parent), _action(action) {}

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

        bool operator==(const MCTSAction& other) const
        {
            return _action == other._action;
        }

        node_ptr expand()
        {
            // std::cout << "Expanding action" << std::endl;
            // std::cout << "Moved to " << _parent->state()->move(_action)._x << " " << _parent->state()->move(_action)._y << std::endl;
            node_ptr to_add = std::make_shared<NodeType>(_parent->state()->move(_action), _parent->rollout_depth(), _parent->gamma());
            to_add->parent() = this->shared_from_this();
            _children.push_back(to_add);
            return to_add;
        }

        std::pair<double, node_ptr> value()
        {
            double v = -std::numeric_limits<double>::max();
            node_ptr best = nullptr;
            for (auto node : _children) {
                double d = ActionValue()(node);
                if (d > v) {
                    v = d;
                    best = node;
                }
            }

            return std::make_pair(v, best);
        }

    protected:
        node_ptr _parent;
        std::vector<node_ptr> _children;
        ActionType _action;
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
            // std::cout << "Iterate!" << std::endl;
            node_ptr to_simulate = _tree_policy();
            // std::cout << "Node to simulate: " << to_simulate->_state->_x << " " << to_simulate->_state->_y << std::endl;
            double value = _simulate(vfun, to_simulate);
            _back_prop(to_simulate, value);
        }

        node_ptr best_child() const
        {
            if (_state->terminal())
                return nullptr;
            assert(_children.size() > 0);

            double v = -std::numeric_limits<double>::max();
            node_ptr best = nullptr;

            for (auto child : _children) {
                std::pair<double, node_ptr> v_cur = child->value();
                double d = std::get<0>(v_cur);
                if (d > v) {
                    v = d;
                    best = std::get<1>(v_cur);
                }
            }

            return best;
        }

        bool fully_expanded() const
        {
            return !_state->has_actions();
        }

    protected:
        action_ptr _parent;
        std::vector<action_ptr> _children;
        state_ptr _state;
        double _value, _gamma;
        size_t _visits, _rollout_depth;

        node_ptr _tree_policy()
        {
            // std::cout << "Tree policy" << std::endl;
            node_ptr cur_node = this->shared_from_this();
            while (!cur_node->_state->terminal()) {
                // std::cout << cur_node->_state->_x << " " << cur_node->_state->_y << std::endl;
                if (!cur_node->fully_expanded()) {
                    // std::cout << "Not expanded!" << std::endl;
                    return cur_node->_expand();
                }
                else {
                    // std::cout << "Select child!" << std::endl;
                    cur_node = cur_node->best_child();
                }
            }

            return cur_node;
        }

        node_ptr _expand()
        {
            // std::cout << "Expanding" << std::endl;
            Action act = _state->next_action();
            // std::cout << "Next action: " << act << std::endl;
            action_ptr next_action = std::make_shared<action_type>(act, this->shared_from_this()); // i am not sure about this!?
            auto it = std::find_if(_children.begin(), _children.end(), [&](action_ptr const& p) { return *p == *next_action; });
            if (_children.size() == 0 || it == _children.end()) {
                // std::cout << "Not in children" << std::endl;
                _children.push_back(next_action);
                it = _children.end();
                it--;
            }

            return (*it)->expand();
        }

        template <typename ValueFunc>
        double _simulate(ValueFunc vfun, node_ptr to_simulate)
        {
            // std::cout << "Simulating.." << std::endl;
            double discount = _gamma;
            double reward = 0.0;

            state_ptr cur_state = to_simulate->_state;

            // Check if current state is terminal
            if (cur_state->terminal())
                return vfun.max_reward();

            for (size_t k = 0; k < _rollout_depth; ++k) {
                // Choose action according to default policy
                Action action = DefaultPolicy()(cur_state);
                // // This assumes that all states have at least one valid action
                // while (!cur_state->valid(action)) {
                //     action = DefaultPolicy()(cur_state);
                // }

                // Get value from (PO)MDP
                reward += discount * vfun(cur_state, action);

                // Update state
                cur_state = std::make_shared<State>(cur_state->move(action));

                // Check if terminal state
                if (cur_state->terminal())
                    break;
                discount *= _gamma;
            }

            // std::cout << "Returning " << reward << std::endl;

            return reward;
        }

        void _back_prop(node_ptr simulated, double value)
        {
            node_ptr cur_node = simulated;
            while (cur_node != nullptr) {
                cur_node->_value += value;
                cur_node->_visits++;
                if (cur_node->_parent == nullptr)
                    break;
                cur_node = cur_node->_parent->parent(); // state->action->state
            }
        }
    };
}

#endif
