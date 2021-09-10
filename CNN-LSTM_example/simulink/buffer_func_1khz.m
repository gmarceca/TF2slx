function [window_t, reset] = buffer_func_1khz(u2)
%#codegen
% for output
persistent data
% counter
persistent index 

input_size = size(u2,1); % 10x3

% for the first time step
if isempty(data)
  data = zeros(2,40);
  index = 0;
end

window_size = size(data,2);

enable = index*input_size>=window_size;

% true only if index <= window size (reset LSTM hidden states)
reset = index*input_size<=window_size;
% (change true --> false in cpp?)

index = index +1;

data(:, 1:window_size-1*input_size) = data(:, input_size+1:window_size);
data(:, end-(input_size-1):end) = u2';

window_t = single(data*enable);

