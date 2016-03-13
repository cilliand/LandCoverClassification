function score = getPdfScore(X, mu, sigma)
% Returns PDF score for a given vector.
% i.e. P(X,wi) = score.
    inner = double(X) - mu;
    inner = (inner*inv(sigma))*inner';
    inner = inner * -0.5;
    outer = 1/(sqrt(det(sigma)));
    score = outer*exp(inner);
end