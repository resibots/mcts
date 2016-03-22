/**
 * @file mdpHarness.cpp
 * Test harness for MCTS applied to simple MDP
 */
#include <ctime>
#include <exception>
#include <iostream>
#include <mcts/mcts_node.hpp>

namespace {

    struct SimpleBandit_m {
        template <typename State>
        double operator()(const State& state, size_t action)
        {
            return static_cast<double>(rand() % RAND_MAX) / RAND_MAX;
        }
    };
}

int main()
{
    try {
        //************************************************************************
        // Set random seed using current time
        //************************************************************************
        std::srand(std::time(0));

        //************************************************************************
        // Create a simple bandit process for test purposes
        //************************************************************************
        SimpleBandit_m bandit;

        //************************************************************************
        // Try instantiate a tree node
        //************************************************************************
        const size_t N_ACTIONS = 10000; //1000000;
        mcts::MCTSNode<mcts::EmptyState, mcts::UCTValue, mcts::UniformRandomPolicy> tree(N_ACTIONS);

        //************************************************************************
        // Try expanding the tree a few times
        //************************************************************************
        const int N_ITERATIONS = 10;
        for (int k = 0; k < N_ITERATIONS; ++k) {
            //  std::cout << "tree: " << tree << std::endl;
            std::cout << "iteration: " << k << std::endl;
            tree.iterate(bandit);
        }
        // std::cout << "tree: " << tree << std::endl;

        //************************************************************************
        // Log the best action, depth and number of nodes
        //************************************************************************
        int bestAction = tree.best_action();
        std::cout << "Best Action: " << bestAction << std::endl;
        int nNodes = tree.nodes();
        std::cout << "Number of Nodes: " << nNodes << std::endl;
        int maxDepth = tree.max_depth();
        std::cout << "Max Depth: " << maxDepth << std::endl;

        //************************************************************************
        // Figure out the true best action
        //************************************************************************
        int correctAction = 0;
        double bestQ = -std::numeric_limits<double>::max();
        for (size_t k = 0; k < N_ACTIONS; ++k) {
            double curVal = tree.q_value(k);
            if (bestQ <= curVal) {
                bestQ = curVal;
                correctAction = k;
            }
        }

        //************************************************************************
        // Check that the reported best action is correct
        //************************************************************************
        if (correctAction != bestAction) {
            std::cout << "Wrong best action - should be: " << correctAction
                      << std::endl;
            return EXIT_FAILURE;
        }
        else {
            std::cout << "Correct best action" << std::endl;
        }

        //************************************************************************
        // Check that the number of nodes is correct (this at least is
        // predicable, because we expand by N_ACTIONS on each iteration).
        //************************************************************************
        const int EXP_N_NODES = 1 + N_ACTIONS * N_ITERATIONS;
        if (EXP_N_NODES != nNodes) {
            std::cout << "Unexpected number of nodes. Should be: " << EXP_N_NODES << std::endl;
            return EXIT_FAILURE;
        }
        else {
            std::cout << "Number of nodes is correct: " << EXP_N_NODES << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cout << "Caught error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    //***************************************************************************
    // Return sucessfully
    //***************************************************************************
    return EXIT_SUCCESS;
}
