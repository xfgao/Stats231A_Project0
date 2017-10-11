% cd('/home/binroot/tutorial/code/examples');
pathTemp = genpath('./');
addpath(pathTemp);
% setup;

% task 1
% cnn_cifar;
% cd('data/cifar-lenet/');
% load('net-epoch-30.mat');
% disp(info.train.error);
% disp(info.val.error);
% t = 1:30;
% plot(t,info.train.error);
% hold on
% plot(t,info.val.error);
% title('training & validation error');
% legend('train','val');

% task 2
% net = cnn_cifar_init();
% cnn_cifar;
% load('net-epoch-30-full.mat');
% disp(info.train.error);
% disp(info.val.error);
% t = 1:30;
% plot(t,info.train.error);
% hold on
% plot(t,info.val.error);
% title('training & validation error');
% legend('train','val');
% xlabel('Epoch');
% ylabel('Error');
% ylim([0,1]);
% copyfile model.mat model-blk3.mat;
% copyfile data/cifar-lenet/net-epoch-30.mat data/cifar-lenet/net-epoch-30-blk3.mat;

% task 3
close all
model = load('model-full.mat');
model.net.layers = model.net.layers(1:end-1); % remove softmax
for i = 1:10
    img = imread(['images/',num2str(i),'.png']);
    img = single(img) - model.net.averageImage;
    res = vl_simplenn(model.net,img);
    response = res(2).x;
    count = 1;
    figure;
    for j = 1:32
        subplot(4,8,count);
        image(response(:,:,j));
        axis tight;
        axis off;
        daspect([1 1 1]);
        count = count + 1;
    end
    print(['filter_response/',num2str(i)],'-djpeg');
end





