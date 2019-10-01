
% Add ukf to the path to make all components available.

addpath(pwd);

% Recursively add directories to the Matlab path.
cd examples;
addpath(genpath(pwd));
cd ..;

cd benchmarks;
addpath(genpath(pwd));
cd ..;

cd ukfm;
addpath(genpath(pwd));
cd ..;

cd geometry;
addpath(genpath(pwd));
cd ..;


% Ask user if the path should be saved or not
fprintf('UKF was added to Matlab''s path.\n');
response = input('Save path for future Matlab sessions? [Y/N] ', 's');
if strcmpi(response, 'Y')
    failed = savepath();
    if ~failed
        fprintf('Path saved: no need to call importmanopt next time.\n');
    else
        fprintf(['Something went wrong.. Perhaps missing permission ' ...
                 'to write on pathdef.m?\nPath not saved: ' ...
                 'please re-call importmanopt next time.\n']);
    end
else
    fprintf('Path not saved: please re-call importukfm next time.\n');
end
