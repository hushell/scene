function scores = svm_eval(model, feat, bias)
    % scores = dot(model, feat, 1) + bias;
    scores = model.' * feat + bias;
end