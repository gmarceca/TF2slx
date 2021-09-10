function [window_t] = buffer_func_10khz(u2)
%#codegen
% for output
persistent data_10khz
% counter
persistent index_10khz

% for the first time step
if isempty(data_10khz)
  data_10khz = zeros(10, 2);
  index_10khz = 0;
end

window_size = size(data_10khz,1);

index_10khz = index_10khz +1;

enable = index_10khz>=window_size;

% true only if index <= window size (reset LSTM hidden states)
%reset = index<=window_size;
% Here we return enable instead of reset
% (change true --> false in cpp?)

data_10khz(1:window_size-1, :) = data_10khz(2:window_size, :);
data_10khz(end, :) = u2;

window_t = single(data_10khz*enable);
