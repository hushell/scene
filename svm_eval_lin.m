function scores = svm_eval_lin(model, feat)
    if 0
        test_labels = -1 .* ones(size(feat,2), 1);
        test_data = feat.';
        test_data = double(test_data);
        test_data = sparse(test_data);
    
        % '-b 1' only for logistic regression?
        % [l, a, scores] = predict(test_labels, test_data, model, '-b 1');
        [l, a, scores] = predict(test_labels, test_data, model);
    else
        scores = model.w' * feat + model.b' * ones(1, size(feat, 2));
end
