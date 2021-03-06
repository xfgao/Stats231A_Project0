function net = cnn_cifar_init(opts)

lr = [.1 2] ;

% Define network CIFAR10-quick
net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(5,5,3,32, 'single'), zeros(1, 32, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,32,32, 'single'), zeros(1,32,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; % Emulate caffe

% Block 3
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.05*randn(5,5,32,64, 'single'), zeros(1,64,'single')}}, ...
%                            'learningRate', lr, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'pool', ...
%                            'method', 'avg', ...
%                            'pool', [3 3], ...
%                            'stride', 2, ...
%                            'pad', [0 1 0 1]) ; % Emulate caffe

% Block 4
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.05*randn(4,4,64,64, 'single'), zeros(1,64,'single')}}, ...
%                            'learningRate', lr, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;

% Original Block 5
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.05*randn(1,1,64,10, 'single'), zeros(1,10,'single')}}, ...
%                            'learningRate', .1*lr, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
                       
% Block 5 for Block 1
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.01*randn(16,16,32,10, 'single'), zeros(1,10,'single')}}, ...
%                            'learningRate', 0.001*lr, ...
%                            'stride', 1, ...
%                            'pad', 0) ;

% Block 5 for Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(8,8,32,10, 'single'), zeros(1,10,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;        

% Block 5 for Block 3
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{0.01*randn(4,4,64,10, 'single'), zeros(1,10,'single')}}, ...
%                            'learningRate', .1*lr, ...
%                            'stride', 1, ...
%                            'pad', 0) ;                       

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;