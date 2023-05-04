#ifndef MCTS_PARALLEL_HPP
#define MCTS_PARALLEL_HPP

#include <algorithm>
#include <vector>

#ifdef USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#ifndef USE_TBB_ONEAPI
#include <tbb/task_scheduler_init.h>
#endif
#endif

namespace mcts {
    namespace par {
#ifdef USE_TBB
        template <typename X>
        using vector = tbb::concurrent_vector<X>; // Template alias (for GCC 4.7 and later)

        /// @ingroup par_tools
        /// convert a std::vector to something else (e.g. a std::list)
        template <typename V>
        std::vector<typename V::value_type> convert_vector(const V& v)
        {
            std::vector<typename V::value_type> v2(v.size());
            std::copy(v.begin(), v.end(), v2.begin());
            return v2;
        }
#else
        template <typename X>
        using vector = std::vector<X>; // Template alias (for GCC 4.7 and later)

        template <typename V>
        V convert_vector(const V& v)
        {
            return v;
        }

#endif

#if (defined USE_TBB) && !(defined USE_TBB_ONEAPI)
        inline void init()
        {
            static tbb::task_scheduler_init init;
        }
#else
        /// @ingroup par_tools
        /// init TBB (if activated) for multi-core computing
        void init()
        {
        }
#endif

        ///@ingroup par_tools
        /// parallel for
        template <typename F>
        inline void loop(size_t begin, size_t end, const F& f)
        {
#ifdef USE_TBB
            tbb::parallel_for(size_t(begin), end, size_t(1), [&](size_t i) {
                // clang-format off
                f(i);
                // clang-format on
            });
#else
            for (size_t i = begin; i < end; ++i)
                f(i);
#endif
        }

        /// @ingroup par_tools
        /// parallel for_each
        template <typename Iterator, typename F>
        inline void for_each(Iterator begin, Iterator end, const F& f)
        {
#ifdef USE_TBB
            tbb::parallel_for_each(begin, end, f);
#else
            for (Iterator i = begin; i != end; ++i)
                f(*i);
#endif
        }

        /// @ingroup par_tools
        /// parallel max
        template <typename T, typename F, typename C>
        T max(const T& init, int num_steps, const F& f, const C& comp)
        {
#ifdef USE_TBB
            auto body = [&](const tbb::blocked_range<size_t>& r, T current_max) -> T {
                // clang-format off
            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                T v = f(i);
                if (comp(v, current_max))
                  current_max = v;
            }
            return current_max;
                // clang-format on
            };
            auto joint = [&](const T& p1, const T& p2) -> T {
                // clang-format off
            if (comp(p1, p2))
                return p1;
            return p2;
                // clang-format on
            };
            return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_steps), init,
                body, joint);
#else
            T current_max = init;
            for (size_t i = 0; i < num_steps; ++i) {
                T v = f(i);
                if (comp(v, current_max))
                    current_max = v;
            }
            return current_max;
#endif
        }
        /// @ingroup par_tools
        /// parallel sort
        template <typename T1, typename T2, typename T3>
        inline void sort(T1 i1, T2 i2, T3 comp)
        {
#ifdef USE_TBB
            tbb::parallel_sort(i1, i2, comp);
#else
            std::sort(i1, i2, comp);
#endif
        }

        /// @ingroup par_tools
        /// replicate a function nb times
        template <typename F>
        inline void replicate(size_t nb, const F& f)
        {
#ifdef USE_TBB
            tbb::parallel_for(size_t(0), nb, size_t(1), [&](size_t i) {
                // clang-format off
                f();
                // clang-format on
            });
#else
            for (size_t i = 0; i < nb; ++i)
                f();
#endif
        }
    } // namespace par
} // namespace mcts

#endif
