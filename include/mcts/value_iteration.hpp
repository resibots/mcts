#ifndef VALUE_ITERATION_HPP
#define VALUE_ITERATION_HPP

#include <numeric>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>

template <typename Iter_T>
long double vector_norm(Iter_T first, Iter_T last)
{
    return std::sqrt(std::inner_product(first, last, first, 0.0L));
}

template <typename State>
class ValueIteration {
public:
    ValueIteration(const std::vector<State>& states, const std::vector<std::vector<std::vector<double>>>& transitions, double gamma = 0.9, double epsilon = 1e-6)
        : _states(states), _prob(transitions), _gamma(gamma), _epsilon(epsilon)
    {
        assert(states.size() == transitions.size());
        assert(transitions[0].size() > 0);
        assert(transitions.size() == transitions[0].size());
        assert(transitions[0][0].size() > 0);
        _n_actions = transitions[0][0].size();
    }

    template <typename MDP>
    void solve(MDP mdp, size_t k = 10000)
    {
        std::vector<double> next_value(_states.size(), 0.0);
        _value = std::vector<double>(_states.size(), 0.0);

        // iterations
        for (size_t i = 0; i < k; i++) {
            for (size_t s = 0; s < _states.size(); s++) {
                double max = -std::numeric_limits<double>::max();
                for (size_t a = 0; a < _n_actions; a++) {
                    double v = _action_value(mdp, s, a);
                    if (v > max) {
                        max = v;
                    }
                }
                next_value[s] = max;
            }
            std::vector<double> results(_states.size());
            std::transform(next_value.begin(), next_value.end(), _value.begin(), results.begin(), [&](double l, double r) { return std::abs(l - r); });
            _value = next_value;
            size_t n = 0;
            for (auto d : results) {
                if (d < _epsilon)
                    n++;
            }
            if (n == _states.size())
                break;
        }

        // get policy
        _policy.resize(_states.size());
        for (size_t s = 0; s < _states.size(); s++) {
            size_t best = 0;
            double max = -std::numeric_limits<double>::max();
            for (size_t a = 0; a < _n_actions; a++) {
                double v = _action_value(mdp, s, a);
                // if (std::abs(v - max) < _epsilon) {
                //     std::cout << "Tie: (" << _states[s]._x << ", " << _states[s]._y << ") " << a << " vs " << best << " with: " << v << std::endl;
                // }
                if (v > max) {
                    max = v;
                    best = a;
                }
            }
            _policy[s] = best;
        }
    }

    std::vector<size_t> policy() { return _policy; }
    std::vector<double> value() { return _value; }
    double gamma() { return _gamma; }
    double epsilon() { return _epsilon; }
    size_t n_actions() { return _n_actions; }

protected:
    std::vector<State> _states;
    std::vector<std::vector<std::vector<double>>> _prob; // SxSxA
    std::vector<double> _value;
    std::vector<size_t> _policy;
    size_t _n_actions;
    double _gamma, _epsilon;

    template <typename MDP>
    double _action_value(MDP mdp, size_t state, size_t action)
    {
        double sum = 0.0;
        for (size_t i = 0; i < _states.size(); i++) {
            sum += _prob[state][i][action] * (mdp(_states[state], _states[i], action) + _gamma * _value[i]);
        }
        return sum;
    }
};

#endif
