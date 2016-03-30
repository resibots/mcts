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
        const double _c = 1e-2; //1.0 / std::sqrt(2.0);
        const double _epsilon = 1e-6;

        template <typename Node>
        double operator()(const std::shared_ptr<Node>& node)
        {
            // std::cout << "UCT: " << node->state()->_x << " " << node->state()->_y << std::endl;
            // std::cout << "value: " << node->value() << " visits: " << node->visits() << std::endl;
            // std::cout << "parent: " << node->parent()->action() << std::endl;
            // std::cout << "parent: " << node->parent()->parent()->visits() << std::endl;
            // state->action->state
            // std::cout << node->value() / (double(node->visits()) + _epsilon) << " vs " << _c * std::sqrt(2.0 * std::log(node->parent()->parent()->visits()) / (double(node->visits()) + _epsilon)) << std::endl;
            return node->value() / (double(node->visits()) + _epsilon); // + _c * std::sqrt(2.0 * std::log(node->parent()->parent()->visits()) / (double(node->visits()) + _epsilon));
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
            auto st = _parent->state()->move(_action);
            // std::cout << "Moved to " << st._x << " " << st._y << std::endl;
            node_ptr to_add = std::make_shared<NodeType>(st, _parent->rollout_depth(), _parent->gamma());
            // if (*(to_add->state()) == *(_parent->state())) {
            //     _parent->visits()++;
            //     return nullptr;
            // }
            to_add->parent() = this->shared_from_this();
            _children.push_back(to_add);
            return to_add;
        }

        node_ptr add_child(node_ptr to_add)
        {
            // if (*(to_add->state()) == *(_parent->state())) {
            //     if (_children.size() == 0) {
            //         _parent->visits()++;
            //         return nullptr;
            //     }
            //     return _children[std::rand() * _children.size() / double(RAND_MAX)];
            // }

            auto it = std::find_if(_children.begin(), _children.end(), [&](node_ptr const& p) { return *(p->state()) == *(to_add->state()); });
            if (it == _children.end()) {
                // std::cout << "not in " << _parent->state()->_x << " " << _parent->state()->_y << " children!" << std::endl;
                // std::cout << to_add->state()->_x << " " << to_add->state()->_y << std::endl;
                // for (size_t i = 0; i < _children.size(); i++)
                //     std::cout << _children[i]->state()->_x << " " << _children[i]->state()->_y << std::endl;
                // std::cout << "---------------------------" << std::endl;
                to_add->parent() = this->shared_from_this();
                _children.push_back(to_add);
                return to_add;
            }

            return (*it);
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

        double avg_value()
        {
            double v = 0.0;
            for (auto node : _children) {
                v += ActionValue()(node);
            }
            return v / double(_children.size());
        }

        node_ptr best_child()
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
            return best;
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
            // std::cout << "Iterate!" << std::endl;
            node_ptr to_simulate = _tree_policy(vfun);
            if (to_simulate == nullptr)
                return;
            // if (to_simulate->_state->terminal()) {
            //     _back_prop(vfun, to_simulate, vfun.max_reward());
            // }
            // if (to_simulate->_parent != nullptr && to_simulate->_visits == 0) {
            //     double v = _simulate(vfun, to_simulate->_parent->parent());
            //     _back_prop(vfun, to_simulate->_parent->parent(), v);
            // }
            // std::cout << "Node to simulate: " << to_simulate->_state->_x << " " << to_simulate->_state->_y << std::endl;
            // if (to_simulate->_parent != nullptr && to_simulate->_parent->parent()->state()->terminal()) {
            //     return;
            // }
            double value = _simulate(vfun, to_simulate);

            assert(_visited.size() == _rewards.size());

            for (int i = _visited.size() - 1; i >= 0; i--) {
                value = _rewards[i] + _gamma * value;
                _visited[i]->_visits++;
                _visited[i]->_value += value;
            }
            // _back_prop(vfun, to_simulate, value);
        }

        std::pair<node_ptr, action_ptr> best_child(bool simulate = true)
        {
            if (_state->terminal())
                return std::make_pair<node_ptr, action_ptr>(nullptr, nullptr);
            assert(_children.size() > 0);

            double v = -std::numeric_limits<double>::max();
            action_ptr best_action = nullptr;

            for (auto child : _children) {
                // std::pair<double, node_ptr> v_cur = child->value();
                // double d = std::get<0>(v_cur);
                // if (d > v) {
                //     v = d;
                //     best = std::get<1>(v_cur);
                // }
                double d = child->avg_value();
                if (!simulate)
                    std::cout << child->action() << " -> " << d << std::endl;
                if (d > v) {
                    v = d;
                    best_action = child;
                }
            }

            node_ptr best;
            if (simulate) {
                best = std::make_shared<node_type>(_state->move(best_action->action()), _rollout_depth, _gamma);
                best = best_action->add_child(best);
            }
            else {
                best = best_action->best_child();
            }

            return std::make_pair(best, best_action);
        }

        bool fully_expanded() const
        {
            return !_state->has_actions();
        }

        void print(size_t d = 0) const
        {
            std::cout << d << ": " << _state->_x << " " << _state->_y << " -> " << _value << ", " << _visits; // << std::endl;
            if (_parent != nullptr)
                std::cout << " act: " << _parent->action();
            std::cout << std::endl;
            for (size_t i = 0; i < _children.size(); i++) {
                for (size_t k = 0; k < _children[i]->children().size(); k++) {
                    _children[i]->children()[k]->print(d + 1);
                }
            }
        }

    protected:
        action_ptr _parent;
        std::vector<action_ptr> _children;
        state_ptr _state;
        double _value, _gamma;
        size_t _visits, _rollout_depth;
        // helper
        std::vector<node_ptr> _visited;
        std::vector<double> _rewards;

        template <typename ValueFunc>
        node_ptr _tree_policy(ValueFunc vfun)
        {
            _visited.clear();
            _rewards.clear();
            // std::cout << "Tree policy" << std::endl;
            node_ptr cur_node = this->shared_from_this();
            _visited.push_back(cur_node);
            _rewards.push_back(0.0);
            while (!cur_node->state()->terminal()) { // || cur_node->_children.size() != 0) {
                // std::cout << cur_node->_state->_x << " " << cur_node->_state->_y << std::endl;
                if (!cur_node->fully_expanded()) {
                    // std::cout << "Not expanded!" << std::endl;
                    auto tmp = cur_node->_expand();
                    _visited.push_back(std::get<0>(tmp));
                    _rewards.push_back(vfun(std::get<0>(tmp)->_state, std::get<1>(tmp)->action()));
                    return std::get<0>(tmp);
                }
                else {
                    // std::cout << "Select child!" << std::endl;
                    auto tmp = cur_node->best_child();
                    cur_node = std::get<0>(tmp);
                    _visited.push_back(cur_node);
                    _rewards.push_back(vfun(cur_node->_state, std::get<1>(tmp)->action()));
                }

                // if (cur_node == nullptr)
                //     return nullptr;
            }

            return cur_node;
        }

        std::pair<node_ptr, action_ptr> _expand()
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

            auto tmp = (*it)->expand();
            return std::make_pair(tmp, *it);
        }

        template <typename ValueFunc>
        double _simulate(ValueFunc vfun, node_ptr to_simulate)
        {
            // std::cout << "Simulating.." << std::endl;
            double discount = 1.0;
            double reward = 0.0;

            state_ptr cur_state = to_simulate->_state;

            // Check if current state is terminal
            // if (cur_state->terminal()) {
            //     return vfun.max_reward();
            // }

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

        template <typename ValueFunc>
        void _back_prop(ValueFunc vfun, node_ptr simulated, double value)
        {
            node_ptr cur_node = simulated;
            double v = value;
            while (cur_node != nullptr) {
                cur_node->_value += v;
                cur_node->_visits++;
                if (cur_node->_parent == nullptr)
                    break;
                cur_node = cur_node->_parent->parent(); // state->action->state
                // v = v * _gamma;
                // if (cur_node->_parent != nullptr)
                //     v += vfun(cur_node->_state, cur_node->_parent->action());
            }
        }
    };
}

#endif
