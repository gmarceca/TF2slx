function def = policy_compile()

library_md5 = "d6e168725e5a552664c69af867048aad975bc11d";
header_md5 = "54bbb05dd1f9ae789841cd2b31dbfc1cd352afc0";
dmliblocation = fullfile(fileparts(mfilename('fullpath')),'lib');


try
    
library = sprintf('%s/lib_CNNLSTM_LHD_states_03032021.so',dmliblocation);

assert(logical(exist(dmliblocation,'dir')),'%s does not exist',dmliblocation);

%if (system('which md5sum') == 0)
%  actual_md5 = get_md5(library);
%  header_actual_md5 = get_md5('inc/my_policy_export.h');
%  if ~strcmp(actual_md5, library_md5) 
%    fprintf("md5sum of library does not match.\nActual: %s\n Expected: %s\n", actual_md5, library_md5);  
%    return
%  end
%  if ~strcmp(header_actual_md5, header_md5)
%    fprintf("md5sum of header does not match.\nActual: %s\n Expected: %s\n", header_actual_md5, header_md5);
%    return;
%  end
%else
%  warning('Could not check md5sum, no idea if this is the correct .so version');
%end

%% Clean up
delete policy_tlc.mexa64;
delete policy_tlc.c;
delete policy_tlc.tlc;
delete policy_tlc.*;
% unload the libraries
clear('policy_tlc');


%% Setup Legacy Code
def = legacy_code('initialize');

%n_meas = 120; n_ref=24; n_ff=20; n_out=20; % input, output sizes

def.SFunctionName = 'policy_tlc';
def.StartFcnSpec  = 'CreateNetwork()';
def.OutputFcnSpec = 'run(single u1[2][40], single y1[3], boolean u2)';
def.TerminateFcnSpec = 'DeleteNetwork()';
def.HeaderFiles   = {'CNNLSTM_LHD_states.h'};
def.SourceFiles   = {};
def.IncPaths      = {'inc'};
def.SrcPaths      = {};
def.TargetLibFiles  = {'lib_CNNLSTM_LHD_states_03032021.so'};
def.HostLibFiles  = {'lib_CNNLSTM_LHD_states_03032021.so'};
def.LibPaths      = {'lib'};
def.Options.language = 'C';
def.Options.useTlcWithAccel = false;


legacy_code('sfcn_tlc_generate', def);
legacy_code('generate_for_sim', def);
legacy_code('compile', def);

% when changing sizes etc, you must regenerate the block and copy it to
% policy_sim.slx
legacy_code('slblock_generate', def);

%delete tmp.*;
delete policy_tlc.c;
% unload the libraries
clear('policy_tlc')

catch ME
    rethrow(ME)
end

%% 
function [md5_sum] = get_md5(filename)
    [~, output] = system(['md5sum ', filename]);
    md5_cell = split(output);  % output contains sum and filename
    md5_sum = strip(md5_cell{1});
end

end
