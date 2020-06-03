import time
import numpy as np
import scipy.io

# run tdcalc functions: TDdispFS, TDdispHS, TDstrainFS, TDstrainHS
# with two example triangles and compare to MATLAB output for validation.
#
# Eric Lindsey, 2020
# modified from cutde tests by Ben Thompson:
# https://github.com/tbenthompson/cutde/blob/master/tests/test_tde.py

import tdcalc

# run tests:
def test_all(N_test):
    test_FS_simple(N_test)
    test_FS_complex(N_test)
    test_HS_simple(N_test)
    test_HS_complex(N_test)
    
def test_FS_simple(N_test):
    tdcalc_test('matlab_source/FS_simple.mat', tdcalc.TDdispFS, tdcalc.TDstrainFS, N_test)

def test_FS_complex(N_test):
    tdcalc_test('matlab_source/FS_complex.mat', tdcalc.TDdispFS, tdcalc.TDstrainFS, N_test)
    
def test_HS_simple(N_test):
    tdcalc_test('matlab_source/HS_simple.mat', tdcalc.TDdispHS, tdcalc.TDstrainHS,  N_test)

def test_HS_complex(N_test):
    tdcalc_test('matlab_source/HS_complex.mat', tdcalc.TDdispHS, tdcalc.TDstrainHS, N_test)

# main driver routine
def tdcalc_test(matfile, disp_fnc, strain_fnc, N_test = -1):
    
    print('loading %s'%matfile)
    correct = scipy.io.loadmat(matfile,squeeze_me=True)
    
    if N_test == -1:
        N_test = correct['UEf'].shape[0]
    
    # run displacement test
    startd = time.time()
    disp = disp_fnc(correct['obs'][:N_test,:], correct['tri'].astype(float),
                       correct['slip'].astype(float), correct['nu']) # now vectorized! 
    endd = time.time()
    dtime = endd - startd
    
    # validate displacement results and print timing info
    np.testing.assert_almost_equal(disp[:,0], correct['UEf'][:N_test])
    np.testing.assert_almost_equal(disp[:,1], correct['UNf'][:N_test])
    np.testing.assert_almost_equal(disp[:,2], correct['UVf'][:N_test])
    print('displacements for %d points generated in %f seconds' %(N_test, dtime))
    print('time rel. to MATLAB: ',(dtime/N_test)/(correct['dtime']/correct['UEf'].shape[0]))
 
    # run strain test
    starts = time.time()
    strain = strain_fnc(correct['obs'][:N_test,:], correct['tri'].astype(float),
                       correct['slip'].astype(float), correct['nu']).T # now vectorized! 
    ends = time.time()
    stime = ends - starts
    
    # validate strain results and print timing info
    np.testing.assert_almost_equal(strain, correct['Strain'][:N_test,:].T)
    print('strains for %d points generated in %f seconds' %(N_test, stime))
    print('time rel. to MATLAB: ',(stime/N_test)/(correct['stime']/correct['UEf'].shape[0]))

    # test success if we made it this far!
    print('Test success!\n')

if __name__ == "__main__":
    ''' run all tests. This requires the matlab outputs to be generated first. '''
    test_all(-1)

