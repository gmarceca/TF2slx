function [u_feed] = get_input_example(shot) 

if shot == 61043 || shot == 64770 || shot == 64774 || shot == 69514
    name = sprintf('input_signals_%d_ch13.mat',shot);
elseif shot == 64662
    name = sprintf('input_signals_%d_ch14.mat',shot);
else
    name = sprintf('input_signals_%d_ch01.mat',shot);
end

load(fullfile('./inputs/',name));

X = [FIR, PD];
X = fillmissing(X,'constant',0);

length = size(X,1);
tt = 0:1e-4:(length-1)/1e4;

u_feed = zeros(size(tt,2), size(X,2));

u_feed(:,:) = X(:,:);

u_feed = timeseries(u_feed, tt);

end

