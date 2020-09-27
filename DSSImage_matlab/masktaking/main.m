I=imread('../../examples/images/inputs-sem/rose3_sem.png');
I_p=imread('../../examples/images/inputs/rose3.jpg');
I_prime=rgb2gray(I_p);
I_prime=cat(3,I_prime,I_prime,I_prime);

[m,n,c]=size(I);



bw_others=ones(m,n);

I1 = rgb2hsv(I); 
h = I1(:, :, 1); 

bw1 = im2bw(h, 0.4); 
bw2 = im2bw(h, 0.8);
bw3=bw1-bw1.*bw2;
figure;
subplot(2,4,1);imshow(bw3);
% bw3 = ~bw3; 
% bw1 = imfill(bw3, 'holes'); 
% bw1 = imopen(bw1, strel('disk', 5)); 
% bw1 = bwareaopen(bw1, 2000); 

bw2 = cat(3, bw3, bw3, bw3); 
I1 = I .* uint8(bw2); 

subplot(2, 4, 5); imshow(I1); title('???插?惧??', 'FontWeight', 'Bold');
bw_others=bw_others-bw3;






bw1 = im2bw(h, 0.2); 
bw2 = im2bw(h, 0.4); 
bw3=bw1-bw1.*bw2;

subplot(2,4,2);imshow(bw3);
% bw3 = ~bw3; 
% bw1 = imfill(bw3, 'holes'); 
% bw1 = imopen(bw1, strel('disk', 5)); 
% bw1 = bwareaopen(bw1, 2000); 

bw2 = cat(3, bw3, bw3, bw3); 
I2 = I .* uint8(bw2);
subplot(2, 4, 6); imshow(I2); title('???插?惧??', 'FontWeight', 'Bold');
bw_others=bw_others-bw3;


bw1 = im2bw(h, 0.1); 
bw2 = im2bw(h, 0.2); 
bw3=bw1-bw1.*bw2;
subplot(2,4,3);imshow(bw3);
% bw3 = ~bw3; 
% bw1 = imfill(bw3, 'holes'); 
% % bw1 = imopen(bw1, strel('disk', 5)); 
% bw1 = bwareaopen(bw1, 2000);
bw2 = cat(3, bw3, bw3, bw3); 
I3= I .* uint8(bw2);
subplot(2, 4, 7); imshow(I3); title('???插?惧??', 'FontWeight', 'Bold');
bw_others=bw_others-bw3;

bw3=zeros(m,n);
bw3(find(h<0.1))=1;
bw2 = cat(3, bw3, bw3, bw3); 
I4= I .* uint8(bw2); 
subplot(2, 4, 4); imshow(I4); title('???插?惧??')

bw_others=bw_others-bw3;
bw2 = cat(3, bw_others, bw_others, bw_others); % ????妯℃??
I5= I .* uint8(bw2); 

I6=0.6*I3+0.6*I2+0.6*I1+0.6*I4+0.6*I5+0.4*I_prime;
% I6=0.6*I3+0.6*I2+0.6*I1+0.6*I4+0.6*I5+0.4*I_p;%colored 
subplot(2, 4, 8); imshow(I6); title('result');



I = im2double(I_p);
 

 I_HE=I;
 tic

Ycbcr=rgb2ycbcr(I);


Y=Ycbcr(:,:,1);
cb=Ycbcr(:,:,2);
cr=Ycbcr(:,:,3);
% figure(111);subplot(1,2,1);imshow(Y);subplot(1,2,2);imshow(L0/255);%yiyangde


hh=min([m n]);


sigma1=max([ 5 round(hh*1/200)]);%double(uint8(h*0.03));%小尺寸为图像大小的3%
sigma2=round(hh*2/50);%double(uint8(h*0.13));%中尺寸为图像大小的13%
sigma3=round(hh*1/10);%double(uint8(h*0.40));%大尺寸为图像大小的40%
aa=3;
%%%%%%%%%%%%%%%%%%%%生成高斯模板%%%%%%%%%%%%%%%%

F1=fspecial('gaussian',[3*sigma1,3*sigma1],sigma1);
F2=fspecial('gaussian',[3*sigma2,3*sigma2],sigma2);
F3=fspecial('gaussian',[3*sigma3,3*sigma3],sigma3);

%%%%%%%%%%%%%%%%%%%%处理R通道%%%%%%%%%%%%%%%%%%%


[m,n]=size(I_p);

normL3=ones(m,n);
Rr=ones(m,n);refilteredRr=ones(m,n);exprefilteredRr=ones(m,n);change_refilteredRr=ones(m,n);
refilteredG=ones(m,n);change_refilteredG=ones(m,n);exprefilteredG=ones(m,n);


G=zeros(m,n);

%  sd = sqrt(.001);
%  win_size = 10;
% reL1=imresize(L1,[round(m/8),round(n/8)]);
%  filteredimageray = guided_filter(imagegrgay,imagegrgay,  .01, win_size);
% refilteredimageray=imresize(filteredimageray,[round(m/8),round(n/8)]);
%  refilteredimageray2=imresize(refilteredimageray,[m,n]);%
% reL12=imresize(L1,[round(m/8),round(n/8)]);
% reL2=imresize(L2,[round(m/8),round(n/8)]);
% reL3=imresize(L3,[round(m/8),round(n/8)]);

L1 = fastguidedfilter(Y,Y, 3*sigma1, .005, 3*sigma1);

mi1=min(L1(:));
ma1=max(L1(:));
L1=(L1-mi1)*1.0/(2.0-mi1);


I_light=im2uint8(cat(3,L1,L1,L1));

I7=0.5*I3+0.5*I2+0.5*I1+0.5*I4+0.5*I5+0.5*I_light;
figure;imshow(I6); title('result');

figure;imshow(I_light); title('result');

imwrite(I7,'../../examples/images/inputs-semlight/rose3_semlight.png');



