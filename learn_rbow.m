function [pz,W,loglik] = learn_rbow( x, L, options )

dbstop if error

if ~exist( 'options', 'var'); options = []; end
if ~isfield( options,'learn_W'); options.learn_W = 1; end
if ~isfield( options,'learn_pz'); options.learn_pz = 1; end
if ~isfield( options,'plot'); options.plot = 1; end

T = length(x);
[M,R] = size(x{1});
counts = zeros(M, T, R);
x = cat(2, x{:});

for r = 1:R
    counts(:,:,r) = x(:,r:R:end-(R-r));
end

[pz,W,loglik] = learn_rbow_inner( counts, L, options );

end

function [pz,W,loglik] = learn_rbow_inner( counts, L, options )

dbstop if error

if ~exist( 'options', 'var'); options = []; end
if ~isfield( options,'learn_W'); options.learn_W = 1; end
if ~isfield( options,'learn_pz'); options.learn_pz = 1; end
if ~isfield( options,'plot'); options.plot = 1; end

[Z,T,B] = size(counts);
pseudocounts = 2;
Sh = 100;

% Q_{i,r,zr}, L - zr, B - r, i - T. Reversed order in paper
% [initalize latent variables] for each region, pick max prob of model and set 1 
qz = rand(L,B,T); qz = double( bsxfun( @eq, qz, max(qz)) );

if isfield( options,'pz'); 
    pz = options.pz;
else
    pz = ( pseudocounts + sum(qz,3) ) / ( T + L*pseudocounts);
end

if isfield( options,'W');
    W = options.W;
else
    W = pseudocounts + squeeze( sum(sum( bsxfun( @times, qz, reshape( permute( counts, [3,2,1]),[1,B,T,Z])),3),2));
    W = bsxfun( @rdivide, W, sum(W,2));
end

max_iter = 100;
LL = zeros(1,max_iter);

for iter=1:max_iter
    
    % E-step
    for t=1:T
        lqz = log(pz) + squeeze( sum( bsxfun( @times, squeeze( counts(:,t,:) ), log( reshape(W.^(1/Sh)',[Z,1,L]) ) ),1) )';
        qz(:,:,t) = exp(  bsxfun( @minus, bsxfun( @minus, lqz, max( lqz) ), log( sum( exp( bsxfun( @minus, lqz, max( lqz) ) ) )) ) );
    end
    
    % M-step aall
    if options.learn_pz
        pz = ( pseudocounts + sum(qz,3) ) / ( T + L*pseudocounts);
    end
    
    % M-step wall
    if options.learn_W
        W = pseudocounts + squeeze( sum(sum( bsxfun( @times, qz, reshape( permute( counts, [3,2,1]),[1,B,T,Z])),3),2));
        W = bsxfun( @rdivide, W, sum(W,2));
    end
    
    loglik = zeros(1,T);
    for t=1:T
        pr(t) = sum(sum( qz(:,:,t).*log( pz) ));
        obs(t) = sum( sum( qz(:,:,t).* squeeze( sum( bsxfun( @times, squeeze( counts(:,t,:) ), log( reshape(W',[Z,1,L]) ) ),1) )'));
        ent(t) = sum( sum( qz(:,:,t).*log( qz(:,:,t))));
        loglik(t) = pr(t) + obs(t) - ent(t);
    end
    
    LL(iter) = sum( loglik );
    fprintf('Iter %d: log-likelihood is %f\n', iter, LL(iter));
    
    if options.plot
      Nplots = 3;    
      set(gcf,'Name','RBOW generative model');
      subplot(Nplots,1,1); imagesc(pz);         title('pz');
      subplot(Nplots,1,2); imagesc(W);        title('W');
      subplot(Nplots,1,3); plot(LL(1:end),'.-'); title('log-likelihood per iteration')
      drawnow;
    end
    
end

end
