#include <iostream>
#include <ctime>
#include <mcts/uct.hpp>

#define SIZE 4
#define GOAL 4
#define PROB_WIND 0.2

struct SailingState {
    int _x, _y, _wind_dir;
    std::vector<int> _used_actions;

    SailingState()
    {
        _x = _y = 0;
        _wind_dir = static_cast<int>(std::rand() * 8.0 / double(RAND_MAX));
    }

    SailingState(int x, int y)
    {
        _x = x;
        _y = y;
        _wind_dir = static_cast<int>(std::rand() * 8.0 / double(RAND_MAX));
    }

    SailingState(int x, int y, int wind_dir)
    {
        _x = x;
        _y = y;
        _wind_dir = wind_dir;
    }

    bool valid(int action) const
    {
        SailingState st = move(action, false);
        if (st._x < 0 || st._y < 0 || st._x >= SIZE || st._y >= SIZE)
            return false;
        if (std::abs(action - _wind_dir) == 1)
            return false;
        return true;
    }

    int next_action()
    {
        int i;
        do {
            i = static_cast<int>(std::rand() * 8.0 / double(RAND_MAX));
        } while (!valid(i) || std::count(_used_actions.begin(), _used_actions.end(), i) > 0);
        _used_actions.push_back(i);
        return i;
    }

    size_t valid_actions() const
    {
        if (terminal())
            return 0;
        size_t _valid = 0;
        for (int i = 0; i < 8; i++) {
            if (valid(i))
                _valid++;
        }

        return _valid;
    }

    bool has_actions()
    {
        return _used_actions.size() < valid_actions();
    }

    SailingState move(int action, bool refine = true) const
    {
        int x_new = _x, y_new = _y;

        // double r = std::rand() / (double)RAND_MAX;
        // if (r < PROB_WIND) {
        //     _wind_dir = static_cast<int>(std::rand() * 8.0 / double(RAND_MAX));
        // }

        if (action == 0) // up
        {
            y_new++;
        }
        else if (action == 1) // down
        {
            y_new--;
        }
        else if (action == 2) // right
        {
            x_new++;
        }
        else if (action == 3) // left
        {
            x_new--;
        }
        else if (action == 4) //diag up left
        {
            y_new++;
            x_new--;
        }
        else if (action == 5) //diag down right
        {
            y_new--;
            x_new++;
        }
        else if (action == 6) //diag up right
        {
            y_new++;
            x_new++;
        }
        else if (action == 7) //diag down left
        {
            y_new--;
            x_new--;
        }

        if (refine) {
            if (x_new < 0)
                x_new++;
            if (y_new < 0)
                y_new++;
            if (x_new >= SIZE)
                x_new--;
            if (y_new >= SIZE)
                y_new--;
        }

        return SailingState(x_new, y_new, _wind_dir);
    }

    int random_action() const
    {
        int act;
        do {
            act = static_cast<size_t>(std::rand() * 8.0 / (double)RAND_MAX);
        } while (!valid(act));

        return act;
    }

    int best_action() const
    {
        int act = 0;
        double v = std::numeric_limits<double>::max();
        for (int i = 0; i < 8; i++) {
            if (!valid(i))
                continue;
            SailingState tmp = move(i);
            double dx = tmp._x - GOAL + 1;
            double dy = tmp._y - GOAL + 1;
            double d = dx * dx + dy * dy;
            if (d < v) {
                act = i;
                v = d;
            }
        }

        return act;
    }

    bool terminal() const
    {
        if (_x == (GOAL - 1) && _y == (GOAL - 1))
            return true;
        return false;
    }

    bool operator==(const SailingState& other) const
    {
        return (_x == other._x && _y == other._y);
    }
};

int main()
{
    std::srand(std::time(0));

    return 0;
}
