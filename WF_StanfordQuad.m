% Implementation of the Wirtinger Flow (WF) algorithm presented in the paper 
% "Phase Retrieval via Wirtinger Flow: Theory and Algorithms" 
% by E. J. Candes, X. Li, and M. Soltanolkotabi

% The input data are coded diffraction patterns about an RGB image. Each
% color band is acquired separately (the same codes are used for all 3 bands)

% Code written by M. Soltanolkotabi and E. J. Candes

%%  Read Image 

% Below X is n1 x n2 x 3; i.e. we have three n1 x n2 images, one for each of the 3 color channels  
namestr = 'stanford' ;
% stanstr = 'jpeg'      ;
stanstr = 'jpeg'      ;
% X       = mat2gray(imread([namestr,'.',stanstr])) ;

% mit photo: Photo by Muzammil Soorma on Unsplash
  

X2       = mat2gray(imread([namestr,'.',stanstr])) ;
ds =1;
X = zeros(ceil(size(X2,1)/ds), ceil(size(X2,2)/ds), 3);
for i=1:3
    X(:,:,i) = downsample(downsample(X2(:,:,i), ds)', ds)';

end

% X = X2;
% o1 = 300;
% o2 = 100;
% X = X(:,o2:(o2+512),:);

n1      = size(X,1)                               ;
n2      = size(X,2)                               ;
% imshow(X);

%% Make masks and linear sampling operators  

% Each mask has iid entries following the octanary pattern in which the entries are 
% distributed as b1 x b2 where 
% b1 is uniform over {1, -1, i, -i} (phase) 
% b2 is equal to 1/sqrt(2) with prob. 4/5 and sqrt(3) with prob. 1/5 (magnitude)

randn('state',2014);
rand ('state',2014);

L = 21;                  % Number of masks  
Masks = zeros(n1,n2,L);  % Storage for L masks, each of dim n1 x n2

% Sample phases: each symbol in alphabet {1, -1, i , -i} has equal prob. 
for ll = 1:L, Masks(:,:,ll) = randsrc(n1,n2,[1i -1i 1 -1]); end

% Sample magnitudes and make masks 
temp = rand(size(Masks));
Masks = Masks .* ( (temp <= 0.2)*sqrt(3) + (temp > 0.2)/sqrt(2) );

% C = zeros(32*32);

% F1D_m = dftmtx(32);
% F1D_n = dftmtx(32);
% for i=1:L
%     M = Masks(:,:,i);
%     tmp = kron(F1D_n, F1D_m) .* diag(M(:));
%     C = C + tmp' * tmp;
% end
% C = C / L;
%

C = zeros(n1*n2,1);

F1D_m = dftmtx(n1);
F1D_n = dftmtx(n2);
D = (kron(diag(F1D_n), diag(F1D_m)));
for i=1:L
    M = Masks(:,:,i);
    tmp =  D(:) .* M(:);
    C = C + abs(tmp).^2;
end
C = C / L;

W = (C).^.5;
W = reshape(W, n1, n2);
Masksw = Masks ./ repmat((W), 1, 1, L);

% Make linear operators; 
A = @(I)  fft2(conj(Masks) .* reshape(repmat(I,[1 L]), size(I,1), size(I,2), L));  % Input is n1 x n2 image, output is n1 x n2 x L array
Aw = @(I)  fft2(conj(Masksw) .* reshape(repmat(I,[1 L]), size(I,1), size(I,2), L));  % Input is n1 x n2 image, output is n1 x n2 x L array
At = @(Y) sum(Masks .* ifft2(Y), 3) * size(Y,1) * size(Y,2);                       % Input is n1 x n2 x L array, output is n1 x n2 image
Atw = @(Y) sum(Masksw .* ifft2(Y), 3) * size(Y,1) * size(Y,2);                       % Input is n1 x n2 x L array, output is n1 x n2 image

%% Prepare structure to save intermediate results 

ttimes   = 10:10:200;        % Iterations at which we will save info 
ntimes = length(ttimes)+1;   % +1 because we will save info after the initialization 
Xhats    = cell(1,ntimes);
Xhatsbw    = cell(1,ntimes);
Xhatspol    = cell(1,ntimes);
Xhatsnes    = cell(1,ntimes);
for mm = 1:ntimes, Xhats{mm} = zeros(size(X));  end
for mm = 1:ntimes, Xhatsbw{mm} = zeros(size(X));  end
for mm = 1:ntimes, Xhatspol{mm} = zeros(size(X));  end
for mm = 1:ntimes, Xhatsnes{mm} = zeros(size(X));  end
Times    = zeros(3,ntimes);
Timesbw    = zeros(3,ntimes);
Timespol    = zeros(3,ntimes);
Timesnes    = zeros(3,ntimes);

%%

% tmp = zeros(32*32,32*32);
% C = zeros(32*32);
% for i=1:L
%      F1D_m = dftmtx(32);
%      F1D_n = dftmtx(32);
%     M = Masks(:,:,i);
%     tmp = kron(F1D_n, F1D_m) .* repmat(M(:)',32*32, 1);
%     C = C + tmp' * tmp;
% end
% C = C / L;
% C2 = sqrtm(inv(C));
% C2 = diag(C2);
% C2 = reshape(C2, n2, n1)';

%% Wirtinger flow  

npower_iter = 50;                   % Number of power iterations 
T = max(ttimes);                    % Max number of iterations
tau0 = 330;                         % Time constant for step size 
mu = @(t) min(1-exp(-t/tau0), 0.4); % Schedule for step size 


beta = (sqrt(log(n1*n2))-sqrt(log(2)))/(sqrt(log(n1*n2))+sqrt(2));


for rgb = 1:3, 
    fprintf('Color band %d\n', rgb)
    x = squeeze(X(:,:,rgb)); % Image x is n1 x n2 
    Y = abs(A(x)).^2;        % Measured data 
    Ysqrt = sqrt(Y);
    
    % Initialization
    z0 = randn(n1,n2); z0 = z0/norm(z0,'fro'); % Initial guess 
    tic
    for tt = 1:npower_iter, 
        z0 = At(A(z0)); z0 = z0/norm(z0,'fro');
    end
    Times(rgb,1) = toc;
    
    normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
    z = normest * z0;                   % Apply scaling 
    Xhats{1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; % Initial guess after global phase adjustment 
 
    % Loop    
    fprintf('Done with initialization, starting loop\n')
    tic
    for t = 1:T,
        Bz = A(z);
        C  = (abs(Bz).^2-Y) .* Bz;
        grad = At(C)/numel(C);                   % Wirtinger gradient            
        z   = z - mu(t)/normest^2 * grad;        % Gradient update 
        
        ind =  find(t == ttimes);                % Store results 
        if ~isempty(ind), 
             Xhats{ind+1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; 
             Times(rgb,ind+1) = toc;
        end
        
    end
    
    % Initialization
    z0 = randn(n1,n2); z0 = z0/norm(z0,'fro'); % Initial guess 
    tic
    for tt = 1:npower_iter, 
        z0 = At(A(z0)); z0 = z0/norm(z0,'fro');
    end
    Timespol(rgb,1) = toc;
    
    normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
    z = normest * z0;                   % Apply scaling
    zprev = z;
    Xhatspol{1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; % Initial guess after global phase adjustment 
    
    % Loop    
    fprintf('Done with initialization of Polyak, starting loop\n')
    tic
    for t = 1:T,
        Bz = A(z);
        C  = (abs(Bz).^2-Y) .* Bz;
        grad = At(C)/numel(C);                   % Wirtinger gradient            
        tmp   = z - mu(t)/normest^2 * grad + beta*(z - zprev);        % Gradient update 
        zprev = z;
        z = tmp;
        
        ind =  find(t == ttimes);                % Store results 
        if ~isempty(ind), 
             Xhatspol{ind+1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; 
             Timespol(rgb,ind+1) = toc;
        end
        
    end
    
    
    
    z0 = randn(n1,n2); z0 = z0/norm(z0,'fro'); % Initial guess 
    tic
    for tt = 1:npower_iter, 
        z0 = At(A(z0)); z0 = z0/norm(z0,'fro');
    end
    Timesnes(rgb,1) = toc;
    
    normest = sqrt(sum(Y(:))/numel(Y)); % Estimate norm to scale eigenvector  
    z = normest * z0;                   % Apply scaling 
    zprev = z;

    Xhatsnes{1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; % Initial guess after global phase adjustment
    
    % Loop    
    fprintf('Done with initialization of Nesterov, starting loop\n')
    tic
    % BW Loop
    for t = 1:T,
        Bz = A(z+ beta*(z - zprev));
        C  = (abs(Bz).^2-Y) .* Bz;
        grad = At(C)/numel(C);                   % Wirtinger gradient            
        tmp   = z - mu(t)/normest^2 * grad + beta*(z - zprev);        % Gradient update 
        zprev = z;
        z = tmp;
        
        ind =  find(t == ttimes);                % Store results 
        if ~isempty(ind), 
             Xhatsnes{ind+1}(:,:,rgb) = exp(-1i*angle(trace(x'*z))) * z; 
             Timesnes(rgb,ind+1) = toc;
        end
        
    end
    
end
fprintf('All done!\n')

%% Show some results 

iter = [0 ttimes];
Relerrs = zeros(1,ntimes);
Relerrspol = zeros(1,ntimes);
Relerrsnes = zeros(1,ntimes);
for mm = 1:ntimes; 
    fprintf('Mean running times after %d iterations: %.1f\n', iter(mm), mean(Times(:,mm)))
    Relerrs(mm) = norm(Xhats{mm}(:)-X(:))/norm(X(:)); 
    fprintf('Relative error after %d iterations: %f\n', iter(mm), Relerrs(mm))  
    fprintf('\n')
end

for mm = 1:ntimes; 
    fprintf('Mean running times after %d iterations: %.1f\n', iter(mm), mean(Timespol(:,mm)))
    Relerrspol(mm) = norm(Xhatspol{mm}(:)-X(:))/norm(X(:)); 
    fprintf('Relative error after %d iterations: %f\n', iter(mm), Relerrspol(mm))  
    fprintf('\n')
end

% for tt = 1:ntimes, 
%     figure; imshow(mat2gray(abs(Xhats{tt})),[]);
% end

for mm = 1:ntimes; 
    fprintf('Mean running times after %d iterations: %.1f\n', iter(mm), mean(Timesnes(:,mm)))
    Relerrsnes(mm) = norm(Xhatsnes{mm}(:)-X(:))/norm(X(:)); 
    fprintf('Relative error after %d iterations: %f\n', iter(mm), Relerrsnes(mm))  
    fprintf('\n')
end

% for tt = 1:ntimes, 
%     figure; imshow(mat2gray(abs(Xhatsbw{tt})),[]);
% end

%%
figure
plot(iter,log10(Relerrs), 'linewidth', 3)
hold on
plot(iter,log10(Relerrspol), 'linewidth', 3)
plot(iter,log10(Relerrsnes), 'linewidth', 3)
grid on
legend('WF', 'Polyak', 'Nesterov')
ylabel('$\log_{10}(|x^t - x_*| / |x_*|)$','Interpreter','latex')
xlabel('Iteration')
set(gca,'FontSize',18),set(gca,'FontName','Sans')

exportgraphics(gcf, 'errs_acc_pr.png', 'Resolution', 300)

%%

figure
plot(iter, mean(Times), 'linewidth', 3)
hold on
plot(iter, mean(Timespol), 'linewidth', 3)
plot(iter, mean(Timesnes), 'linewidth', 3)
grid on
ylabel('Time (s)')
xlabel('Iteration')
legend('WF', 'BWGD')
set(gca,'FontSize',18)
set(gca,'FontName','Sans')
exportgraphics(gcf, 'times_acc_pr.png', 'Resolution', 300)


%%


figure
plot(mean(Times),log10(Relerrsbw), 'linewidth', 3)
hold on
plot(mean(Timesbw),log10(Relerrs), 'linewidth', 3)
grid on
ylabel('log-Relative Error')
xlabel('Time (s)')
legend('BWGD', 'WF')
set(gca,'FontSize',18)
set(gca,'FontName','Times')
exportgraphics(gcf, 'errs_times_pr.png', 'Resolution', 300)

%%
idx = 14;
figure
imshow(mat2gray(abs(Xhats{idx})),[]);
imwrite(mat2gray(abs(Xhats{idx})),'wf_rec_1.jpg')
figure
imshow(mat2gray(abs(Xhatspol{idx})),[]);
imwrite(mat2gray(abs(Xhatspol{idx})),'pol_rec_1.jpg')
figure
imshow(mat2gray(abs(Xhatsnes{idx})),[]);
imwrite(mat2gray(abs(Xhatsnes{idx})),'nes_rec_1.jpg')
iter(idx+1)

%% Vectorized
%ntimes(7)
% tmp = zeros(32*32,32*32);
% C = zeros(32*32);
% for i=1:L
%      F1D_m = dftmtx(32);
%      F1D_n = dftmtx(32);
%     M = Masks(:,:,i);
%     tmp = kron(F1D_n, F1D_m) .* diag(M(:));
%     C = C + tmp' * tmp;
% end
% C = C / L;
% 
% 
% 
% figure
% plot(diag(C))
% hold on 
% plot(W(:))


% C2 = sqrtm(inv(C));



% C2 = diag(C2);
% C2 = reshape(C2, n2, n1)';


%2d dft, method 2: vectorize input, gen matrix by linear transformation ideas
%refer to sec 2.6 of Linear Algebra, Gilbert Strang
% x = X(:,:,1);
% % x = reshape(x,[],1);
% [m,n] = size(x);
% Fvec = zeros(m*n,m*n);
% 
% 
% F1D_m = dftmtx(m);
% F1D_n = dftmtx(n);
% X1 = F1D_m * x * F1D_n;
% 
% X2 = kron(F1D_n, F1D_m) * reshape(x,[],1);
% 
% M = Masks(:,:,1);
% 
% X2 = kron(F1D_n, F1D_m) * diag(conj(M(:))) * reshape(x,[],1);
% 
% X3 = A(x);
% X3 = X3(:,:,1);
% 
% norm(X2 - X3(:))
% 
% %%
% 

%% 
save('test.mat','Times', 'Timesbw','Relerrsbw','Relerrs','ntimes','ttimes')%,'Xhats','Xhatsbw','X')
%% 
% A = kron(F1D_n, F1D_m);
% A1 = kron(F1D_n, F1D_m) * diag(conj(M(:)));


% norm(X1(:) - X2(:))
% X2 = Fvec * x; %input,output are vectors
% X2 = reshape(X2,m,n); %convert back to image
%imagesc(abs(X1))

% x = reshape(x,m,n); %convert back to image