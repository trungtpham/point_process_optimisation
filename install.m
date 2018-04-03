addpath('misc')
addpath('source')
eval(['cd ', 'source']);
eval(['mex ', 'lsa_tr_mex.cxx']);
eval(['cd ', '..']);