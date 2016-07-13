#!/usr/bin/env python
# encoding: utf-8

VERSION = '0.0.1'
APPNAME = 'mcts'

srcdir = '.'
blddir = 'build'

from waflib.Build import BuildContext
import tbb


def options(opt):
    opt.load('compiler_cxx')
    opt.load('compiler_c')
    opt.load('tbb')


def configure(conf):
    conf.load('compiler_cxx')
    conf.load('compiler_c')
    conf.load('tbb')

    conf.check_tbb()

    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -xHost  -march=native -mtune=native -unroll -fma -g"
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -march=native -g"
    else:
        if int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 47:
            common_flags = "-Wall -std=c++0x"
        else:
            common_flags = "-Wall -std=c++11"
        opt_flags = " -O3 -march=native -g"

    all_flags = common_flags + opt_flags
    conf.env['CXXFLAGS'] = conf.env['CXXFLAGS'] + all_flags.split(' ')
    print conf.env['CXXFLAGS']


def build(bld):
    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/uct.cpp',
              includes = './include',
              target='uct')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/benchmarks/trap.cpp',
              includes = './include',
              defines = ['SINGLE'],
              target='src/benchmarks/trap')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/benchmarks/trap.cpp',
              includes = './include',
              target='src/benchmarks/trap_parallel')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/benchmarks/trap.cpp',
              includes = './include',
              defines = ['SIMPLE', 'SINGLE'],
              target='src/benchmarks/trap_simple')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/benchmarks/trap.cpp',
              includes = './include',
              defines = ['SIMPLE'],
              target='src/benchmarks/trap_simple_parallel')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/toy_sim.cpp',
              includes = './include',
              target='toy_sim')

    bld.program(features = 'cxx',
              uselib = "TBB",
              install_path = None,
              source='src/toy_sim.cpp',
              includes = './include',
              defines = 'SINGLE',
              target='toy_sim_single')

    bld.install_files('${PREFIX}/include/mcts', 'include/mcts/uct.hpp')
    bld.install_files('${PREFIX}/include/mcts', 'include/mcts/defaults.hpp')
    bld.install_files('${PREFIX}/include/mcts', 'include/mcts/macros.hpp')
    bld.install_files('${PREFIX}/include/mcts', 'include/mcts/parallel.hpp')
