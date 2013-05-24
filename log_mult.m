function L = log_mult(X, PROB)
% X and PROB are m-by-k, L will be m-by-1
% this function equivalent to log(mnpdf(X,PROB))
% but mnpdf can't get correct results for those underflow or upflow
% \log(\frac{card(X)!}{b_1! ... b_K!} \prod_{k=1}^K v_k^{b_k}) 
% = \sum_{i=1}^{card(b)} \log(i)
%   + \sum_{k=1}^K b_k\log(v_k) - \sum_{k=1}^K \sum_{i=1}^{b_k} \log(i)

    % *** TODO is some elements in PROB are zero, log(PROB) will be -inf
    %L = sum(log(1:sum(X,2)),2) + X*log(PROB)' - log_factorial(X);
    L = log_factorial(sum(X,2)) + dot(X,log(PROB),2) - log_factorial(X);
end

function res = log_factorial(b)
% b can be a scalar or a matrix with m-by-k
% output a vector with m-by-1
%     res = 0;
%     for i = 1:length(b)
%         res = res + sum(log(1:b(i)));
%     end
    
    res = sum(arrayfun(@(x)sum(log(1:x),2),b),2);
    
end