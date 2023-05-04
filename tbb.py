#!/usr/bin/env python
# encoding: utf-8
#| Konstantinos Chatzilygeroudis 2018-2023

"""
Quick n dirty tbb detection
"""

from waflib.Configure import conf

def options(opt):
    opt.add_option('--tbb', type='string', help='path to Intel TBB', dest='tbb')

# check if a lib exists for both osx (darwin) and GNU/linux
def check_lib(self, name, path):
    if self.env['DEST_OS'] == 'darwin':
        libname = name + '.dylib'
    else:
        libname = name + '.so'
    res = self.find_file(libname, path)
    lib = res[:-len(libname)-1]
    return lib

@conf
def check_tbb(self, *k, **kw):
    def get_directory(filename, dirs):
        res = self.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    if self.options.tbb:
        includes_tbb = [self.options.tbb + '/include']
        libpath_tbb = [self.options.tbb + '/lib']
    else:
        includes_tbb = ['/usr/local/include/oneapi', '/usr/include/oneapi', '/usr/local/include', '/usr/include', '/opt/intel/tbb/include']
        libpath_tbb = ['/usr/local/lib/', '/usr/lib', '/opt/intel/tbb/lib', '/usr/lib/x86_64-linux-gnu/']

    self.start_msg('Checking Intel TBB includes')
    incl = ''
    lib = ''
    try:
        incl = get_directory('tbb/parallel_for.h', includes_tbb)
        self.end_msg(incl)
    except:
        if required:
            self.fatal('Not found in %s' % str(includes_tbb))
        self.end_msg('Not found in %s' % str(includes_tbb), 'YELLOW')
        return

    check_oneapi = False
    try:
        incl = get_directory('tbb/version.h', includes_tbb)
        check_oneapi = True
    except:
        pass

    self.start_msg('Checking Intel TBB libs')
    try:
        lib = check_lib(self, 'libtbb', libpath_tbb)
        self.end_msg(lib)
    except:
        if required:
            self.fatal('Not found in %s' % str(libpath_tbb))
        self.end_msg('Not found in %s' % str(libpath_tbb), 'YELLOW')
        return

    self.env.LIBPATH_TBB = [lib]
    self.env.LIB_TBB = ['tbb']
    self.env.INCLUDES_TBB = [incl]
    self.env.DEFINES_TBB = ['USE_TBB']
    if check_oneapi:
        self.env.DEFINES_TBB.append('USE_TBB_ONEAPI')
