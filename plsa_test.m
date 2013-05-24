function [Et,td1] = plsa_test(x,y,wt,td,epsilon)
C =length(td);
T = size(td{1},1);
[M,R] = size(x{1});
y = [y{:}]';

Et = cell(C,1);
td1 = cell(C,1);

for c = 1:C
    fprintf('--- testing category %d ---\n', c);
    xs = x(y==c);
    D = length(xs);
    xs = cat(2, xs{:});
    counts = zeros(M,D*R);    
    for r = 1:R
        counts(:,(r-1)*D+1:r*D) = xs(:,r:R:end-(R-r));
    end
    
    Et{c} = zeros(C,D*R);
    td1{c} = cell(C,1);
    for k = 1:C
        [~,td1{c}{k},Et{c}(k,:)] = plsa_encap(counts,T,epsilon,0,wt{k},0); %TODO
    end
end

function [wt,td,E] = plsa_encap(counts,T,epsilon,plo,wt,learn)
%
% [wt,td] = plsa(counts,T,epsilon,plo,wt,learn)
%
% Learns plsa model with T topics from the (W x D) matrix "counts" (words by documents) on basis of Maximum Likelihood
% Finds for each docuement mixing weights of T topics, and for each topic a distribution over the words.
%
% See T. Hofmann's paper at UAI'99 on the plsa model http://www.cs.brown.edu/people/th/papers/Hofmann-UAI99.pdf
%
% Inputs:
%
% counts  (W x D)  : matrix with count data of W words for D documents 
% T       (scalar) : desired number of topics 
% epsilon (scalar) : EM iterations terminate if the relative change in log-likelihood is under epsilon
% plo     (scalar) : plot log-likelihood progress and wt and td matrices if plo ~= 0
% wt      (W x T)  : optional, initializes word-topic distributions from wt (columns should normalize to one) 
% learn   (scalar) : keeps wt fixed if learn = 0;
%
% Outputs:
%
% wt      (W x T)  : word-topic distributions (columns are normalized)
% td      (T x D)  : topic-document distributions (columns are normalized)
% E	  (1 x D)  : document log-likelihoods: log p( words in doc | mixing weights over topics of the document )
%
% J.J. Verbeek, INIRA Rhone-Alpes, 2006-2007.
%

[W,D] = size(counts);
fprintf('--> PLSA started with %d topics, %d documents, and %d word vocabulary.\n',T,D,W);

if nargin<5; wt    = normalize( rand(W,T), 1); end
if nargin<6; learn = 1; end

td   = normalize( ones(T,D), 1);

tot  = sum(counts(:));
done = 0;
iter = 1;

E = sum( counts.*log( wt*td  +eps) ,1);
F = sum( E ) / tot;
    
while ~done; iter = iter+ 1; 

   td = normalize( td .* ( wt' * ( counts ./ ( wt * td + eps ) ) ), 1);
   
   if learn;
       wt = normalize( wt .* ( ( counts ./ ( wt * td + eps ) ) * td' ), 1);
   end
         
   E = sum( counts.*log( wt*td  +eps) ,1); 
   F_new = sum( E ) / tot; F =  [F F_new];
        
   rel_ch = (F(iter) - F(iter-1))/abs(F(iter-1)); if rel_ch< epsilon & (iter> 5 ); done=1;end
   
   fprintf('iteration %3d F = %f rel-ch = %.6f \n', iter, F(iter), rel_ch); 
   
   if plo;
      Nplots = 3;    
      set(gcf,'Name','Probabilistic Latent Semantic Analysis');
      subplot(Nplots,1,1); imagesc(td);         title('Distribution over topics per document');
      subplot(Nplots,1,2); imagesc(wt');        title('Distribution over words per topic');
      subplot(Nplots,1,3); plot(F(2:end),'.-'); title('Average data log-likelihood per iteration')
      drawnow;
   end
   
end
