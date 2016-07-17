#ifndef MCTS_UCT_HPP
#define MCTS_UCT_HPP

#include <cassert>
#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <mutex>
#include <mcts/defaults.hpp>
#include <mcts/macros.hpp>
#include <mcts/parallel.hpp>

namespace mcts {

    template <typename Params, typename NodeType, typename OutcomeSelection, typename ActionType = size_t>
    class MCTSAction : public std::enable_shared_from_this<MCTSAction<Params, NodeType, OutcomeSelection, ActionType>> {
    public:
        using action_type = MCTSAction<Params, NodeType, OutcomeSelection, ActionType>;
        using node_ptr = std::shared_ptr<NodeType>;

        MCTSAction(const ActionType& action, const node_ptr& parent, double value) : _parent(parent), _action(action), _value(value), _visits(0) {}

        node_ptr parent()
        {
            return _parent;
        }

        std::vector<node_ptr> children() const
        {
            return _children;
        }

        std::vector<node_ptr>& children()
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
            return OutcomeSelection()(this->shared_from_this());
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

    template <typename Params, typename State, typename StateInit, typename ValueInit, typename ActionValue, typename DefaultPolicy, typename Action, typename SelectionPolicy, typename OutcomeSelection>
    class MCTSNode : public std::enable_shared_from_this<MCTSNode<Params, State, StateInit, ValueInit, ActionValue, DefaultPolicy, Action, SelectionPolicy, OutcomeSelection>> {
    public:
        using node_type = MCTSNode<Params, State, StateInit, ValueInit, ActionValue, DefaultPolicy, Action, SelectionPolicy, OutcomeSelection>;
        using action_type = MCTSAction<Params, node_type, OutcomeSelection, Action>;
        using action_ptr = std::shared_ptr<action_type>;
        using node_ptr = std::shared_ptr<node_type>;
        using state_ptr = std::shared_ptr<State>;

        MCTSNode(size_t rollout_depth = 1000, double gamma = 0.9) : _gamma(gamma), _visits(0), _rollout_depth(rollout_depth)
        {
            _state = StateInit()();
        }

        MCTSNode(State state, size_t rollout_depth = 1000, double gamma = 0.9) : _gamma(gamma), _visits(0), _rollout_depth(rollout_depth)
        {
            _state = std::make_shared<State>(state);
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

        template <typename RewardFunc>
        void compute(RewardFunc rfun, size_t iterations)
        {
            if (Params::mcts_node::parallel_roots() > 1) {
                par::vector<node_ptr> roots;
                par::replicate(Params::mcts_node::parallel_roots(), [&]() {
                  node_ptr to_ret = std::make_shared<node_type>(*this->_state, this->_rollout_depth, this->_gamma);
                  for (size_t k = 0; k < iterations; ++k) {
                      to_ret->iterate(rfun);
                  }

                  roots.push_back(to_ret);
                });

                node_ptr cur_node = this->shared_from_this();
                for (size_t i = 0; i < roots.size(); i++) {
                    cur_node->merge_inplace(roots[i]);
                }
            }
            else {
                for (size_t k = 0; k < iterations; ++k) {
                    this->iterate(rfun);
                }
            }
        }

        template <typename RewardFunc>
        void iterate(RewardFunc rfun)
        {
            std::vector<node_ptr> visited;
            std::vector<double> rewards;

            node_ptr cur_node = this->shared_from_this();
            visited.push_back(cur_node);
            rewards.push_back(0.0);
            // std::cout << "Iterate!" << std::endl;

            do {
                node_ptr prev_node = cur_node;
                // std::cout << "(" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
                action_ptr next_action = cur_node->_expand();
                // std::cout << "Selected action: " << next_action->action() << std::endl;
                cur_node = next_action->node();
                rewards.push_back(rfun(prev_node->_state, next_action->action(), cur_node->_state));
                // std::cout << "TO: (" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
                visited.push_back(cur_node);
            } while (!cur_node->_state->terminal() && cur_node->visits() > 0);

            double value;
            if (cur_node->_state->terminal()) {
                value = 0.0;
            }
            else {
                // std::cout << "Simulating: (" << cur_node->_state->_x << ", " << cur_node->_state->_y << ")" << std::endl;
                value = cur_node->_simulate(rfun);
            }

            for (int i = visited.size() - 1; i >= 0; i--) {
                value = rewards[i] + _gamma * value;
                visited[i]->_visits++;
                if (visited[i]->_parent != nullptr)
                    visited[i]->_parent->update_stats(value);
            }
        }

        size_t max_depth(size_t parent_depth = 0)
        {
            if (this->_children.size() == 0) {
                return parent_depth + 1;
            }

            size_t maxDepth = 0;
            for (size_t k = 0; k < this->_children.size(); ++k) {
                for (size_t j = 0; j < this->_children[k]->children().size(); j++) {
                    size_t curDepth = this->_children[k]->children()[j]->max_depth(parent_depth + 1);
                    if (maxDepth < curDepth) {
                        maxDepth = curDepth;
                    }
                }
            }

            return maxDepth;
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

        node_ptr merge_with(const node_ptr& other)
        {
            node_ptr to_ret = std::make_shared<node_type>(*this->_state, this->_rollout_depth, this->_gamma);
            to_ret->merge_inplace(other);

            return to_ret;
        }

        void merge_inplace(const node_ptr& other)
        {
            node_ptr to_ret = this->shared_from_this();

            for (auto child : other->_children) {
                auto it = std::find_if(to_ret->_children.begin(), to_ret->_children.end(), [&](action_ptr const& p) { return *p == *child; });
                if (it == to_ret->_children.end())
                    to_ret->_children.push_back(child);
                else {
                    (*it)->value() += child->value();
                    (*it)->visits() += child->visits();
                }
            }
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
        double _gamma;
        size_t _visits, _rollout_depth;

        action_ptr _expand()
        {
            if (SelectionPolicy()(this->shared_from_this())) {
                Action act = _state->next_action();
                action_ptr next_action = std::make_shared<action_type>(act, this->shared_from_this(), ValueInit()(_state));
                auto it = std::find_if(_children.begin(), _children.end(), [&](action_ptr const& p) { return *p == *next_action; });
                if (_children.size() == 0 || it == _children.end()) {
                    _children.push_back(next_action);
                    return next_action;
                }

                return (*it);
            }

            return _select_action();
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

        template <typename RewardFunc>
        double _simulate(RewardFunc rfun)
        {
            double discount = 1.0;
            double reward = 0.0;

            state_ptr cur_state = _state;

            for (size_t k = 0; k < _rollout_depth; ++k) {
                // Choose action according to default policy
                Action action = DefaultPolicy()(cur_state);
                state_ptr prev_state = cur_state;

                // Update state
                cur_state = std::make_shared<State>(cur_state->move(action));

                // Get value from (PO)MDP
                reward += discount * rfun(prev_state, action, cur_state);

                // Check if terminal state
                if (cur_state->terminal())
                    break;
                discount *= _gamma;
            }

            return reward;
        }
    };
}

#endif
