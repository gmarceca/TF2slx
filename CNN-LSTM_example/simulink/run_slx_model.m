shots = {'59073', '61714', '61274', '59065', '61010', '64369', '64060', '64376', '57093', '57095', '61021', '32911', '30268', '45105', '62744', '60097', '58460', '61057', '31807', '33459', '34309', '53601', '42197', '61043', '64770', '64774', '64662'};

len = numel(shots);

for i = 1:len
    shot = str2num(shots{i});
    u_feed = get_input_example(shot);
    eval(['input =u_feed']);
    sim CNNLSTM_LHD_states_v1  
    myout(i) = ans.output_name;
    filename = sprintf('./slx_predictions/slxpred_%d.npy',shot);
    writeNPY(myout(i).Data, filename);
end
