#ifndef MCTS_NODE_HPP
#define MCTS_NODE_HPP

#include <cmath>
#include <cstdlib>
#include <cassert>
#include <limits>
#include <vector>
#include <stack>
#include <memory>
#include <random>
#include <mutex>

namespace mcts {

    // usage :
    // rgen_double_t(0.0, 1.0);
    // double r = rgen.rand();
    // template <typename D>
    // class RandomGenerator {
    // public:
    //     using result_type = typename D::result_type;
    //     RandomGenerator(result_type min, result_type max) : _dist(min, max), _rgen(std::random_device()()) {}
    //     result_type rand()
    //     {
    //         std::lock_guard<std::mutex> lock(_mutex);
    //         return _dist(_rgen);
    //     }
    //
    // private:
    //     D _dist;
    //     std::mt19937 _rgen;
    //     std::mutex _mutex;
    // };
    // using rdist_double_t = std::uniform_real_distribution<double>;
    // using rdist_int_t = std::uniform_int_distribution<int>;
    //
    // using rgen_double_t = RandomGenerator<rdist_double_t>;
    // using rgen_int_t = RandomGenerator<rdist_int_t>;

    struct UCTValue {
        const double _c = 1.0 / std::sqrt(2.0);

        template <typename Node>
        double operator()(const std::shared_ptr<Node>& node)
        {
            return node->value() / (double(node->visits()) + node->epsilon()) + _c * std::sqrt(2.0 * std::log(node->parent()->visits() + 1.0) / (double(node->visits()) + node->epsilon()));
            // return node->value() + _c * std::sqrt(2.0 * std::log(node->parent()->visits() + 1.0) / (double(node->visits()) + node->parent()->epsilon()));
        }
    };

    struct UniformRandomPolicy {
        template <typename State>
        size_t operator()(const State& state, size_t actions_size)
        {
            // rgen_double_t random(0, 1);
            // return static_cast<size_t>(random.rand() * actions_size);
            return static_cast<size_t>(std::rand() * actions_size / RAND_MAX);
        }
    };

    struct EmptyState {
        EmptyState() {}

        EmptyState move_with(size_t action)
        {
            return EmptyState();
        }

        bool operator==(const EmptyState& other)
        {
            return false;
        }
    };

    struct PassThrough {
        size_t operator()(size_t action, size_t actions_size)
        {
            return action;
        }
    };

    template <typename State, typename ValueFunc, typename DefaultPolicy, typename ChooseActions = PassThrough>
    class MCTSNode : public std::enable_shared_from_this<MCTSNode<State, ValueFunc, DefaultPolicy, ChooseActions>> {
    public:
        using node_type = MCTSNode<State, ValueFunc, DefaultPolicy>;
        using node_ptr = std::shared_ptr<node_type>;

        MCTSNode(size_t n_actions, State state = State(), size_t rollout_depth = 5, double gamma = 0.9)
            : _parent(nullptr), _state(state), _leaf(true), _visits(0), _n_actions(n_actions), _rollout_depth(rollout_depth), _value(0.0), _gamma(gamma) {}

        State state() { return _state; }
        node_ptr parent() { return _parent; }
        std::vector<node_ptr> children() { return _children; }
        bool leaf() { return _leaf; }
        size_t visits() { return _visits; }
        size_t n_actions() { return _n_actions; }
        size_t rollout_depth() { return _rollout_depth; }
        double value() { return _value; }
        double gamma() { return _gamma; }
        double epsilon() { return _epsilon; }

        template <typename ValueSimulator>
        void iterate(ValueSimulator mdp)
        {
            node_ptr cur_node = this->shared_from_this();
            std::stack<node_ptr> visited;
            visited.push(cur_node);

            std::stack<double> rewards;
            rewards.push(0.0);

            size_t action = 0;
            double cur_reward = 0.0;
            while (!cur_node->leaf() && cur_node->_children.size() > 0) {
                action = cur_node->select_child();
                cur_reward = mdp(cur_node->_state, action);
                cur_node = cur_node->_children[action];
                visited.push(cur_node);
                rewards.push(cur_reward);
            }

            cur_node->expand();
            if (cur_node->_children.size() > 0) {
                action = cur_node->select_child();
                cur_reward = mdp(cur_node->_state, action);
                cur_node = cur_node->_children[action];
                visited.push(cur_node);
                rewards.push(cur_reward);
            }

            double value = rollout(cur_node, mdp);

            while (!visited.empty()) {
                assert(visited.size() == rewards.size());
                value = rewards.top() + _gamma * value;
                cur_node = visited.top();
                cur_node->update_stats(value);
                visited.pop();
                rewards.pop();
            }
        }

        size_t best_action()
        {
            if (_leaf) {
                // rgen_double_t random(0, 1);
                // return static_cast<size_t>(random.rand() * _children.size());
                return static_cast<size_t>(std::rand() * _children.size() / RAND_MAX);
            }

            size_t selected = 0;
            double best = -std::numeric_limits<double>::max();

            for (size_t k = 0; k < _children.size(); ++k) {
                node_ptr cur_node = _children[k]; // ptr to current child node
                assert(cur_node != nullptr);

                double exp_value = cur_node->_value / (double(cur_node->_visits) + _epsilon);

                // Add small random value to break ties
                exp_value += std::rand() * _epsilon / RAND_MAX;

                if (exp_value > best) {
                    selected = cur_node->_action;
                    best = exp_value;
                }
            }

            return selected;
        }

        double exp_value() const
        {
            return _value / double(_visits);
        }

        double q_value(size_t action) const
        {
            assert(0 <= action);
            assert(_children.size() > action);
            return _children[action]->exp_value();
        }

        size_t nodes() const
        {
            if (_leaf) {
                return 1;
            }

            size_t result = 1;
            for (size_t k = 0; k < _children.size(); ++k) {
                result += _children[k]->nodes();
            }
            return result;
        }

        size_t max_depth(int parentDepth = 0) const
        {
            if (_leaf) {
                return parentDepth + 1;
            }

            size_t maxDepth = 0;
            for (size_t k = 0; k < _children.size(); ++k) {
                size_t curDepth = _children[k]->max_depth(parentDepth + 1);
                if (maxDepth < curDepth) {
                    maxDepth = curDepth;
                }
            }

            return maxDepth;
        }

        void print(int depth = 0)
        {
            std::cout << depth << ": " << _state._x << " " << _state._y << " -> " << _value << " N: " << _visits << std::endl;
            for (size_t i = 0; i < _children.size(); i++) {
                _children[i]->print(depth + 1);
            }
        }

    protected:
        // members
        node_ptr _parent;
        std::vector<node_ptr> _children;
        State _state;
        bool _leaf;
        size_t _visits, _n_actions, _rollout_depth, _action;
        double _value, _gamma, _epsilon;

        size_t select_child()
        {
            assert(!_leaf);
            size_t selected = 0;
            double best = -std::numeric_limits<double>::max();

            for (size_t k = 0; k < _children.size(); ++k) {

                assert(_children[k]);

                double value = ValueFunc()(_children[k]);

                // Add small random value to break ties
                value += std::rand() * _epsilon / RAND_MAX;

                if (value > best) {
                    selected = k;
                    best = value;
                }
            }

            return selected;
        }

        void expand()
        {
            if (!_leaf) {
                return;
            }

            _leaf = false;
            for (size_t k = 0; k < _n_actions; ++k) {
                size_t action = ChooseActions()(k, _n_actions);
                if (!_state.valid(action))
                    continue;
                State to_add = _state.move_with(action);
                node_ptr tmp = node_state(to_add);
                if (tmp != nullptr) {
                    tmp->_visits++;
                    continue;
                }
                //     _children.push_back(tmp);
                // else {
                _children.push_back(std::make_shared<node_type>(_n_actions, to_add, _rollout_depth, _gamma));
                _children.back()->_parent = this->shared_from_this();
                _children.back()->_action = action;
                // }
            }
        }

        node_ptr node_state(const State& state)
        {
            node_ptr root = this->shared_from_this();
            while (root->_parent != nullptr) {
                root = root->_parent;
            }
            if (root == nullptr)
                return nullptr;
            return root->find(state);
        }

        node_ptr find(const State& state)
        {
            if (_state == state)
                return this->shared_from_this();
            if (_leaf)
                return nullptr;
            for (size_t k = 0; k < _children.size(); k++) {
                auto tmp = _children[k]->find(state);
                if (tmp != nullptr)
                    return tmp;
            }
            return nullptr;
        }

        template <typename ValueSimulator>
        double rollout(const node_ptr& cur_node, ValueSimulator mdp)
        {
            double discount = 1.0;
            double reward = 0.0;

            State cur_state = cur_node->_state;

            for (size_t k = 0; k < _rollout_depth; ++k) {
                // Choose action according to policy
                size_t action = DefaultPolicy()(cur_state, _n_actions);
                // This assumes that all states have at least one valid action
                while (!cur_state.valid(action)) {
                    action = DefaultPolicy()(cur_state, _n_actions);
                }

                // Get value from (PO)MDP
                reward += discount * mdp(cur_state, action);

                // Update state
                cur_state = cur_state.move_with(action);

                // Check if terminal state
                if (mdp.final(cur_state))
                    break;
                discount *= _gamma;
            }

            return reward;
        }

        void update_stats(double value)
        {
            _visits++;
            _value += value;
        }
    };
}

#endif
