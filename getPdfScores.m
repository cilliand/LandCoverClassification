function score = getPdfScores(X, mu, sigma)
inner = double(X) - mu;
inner = (inner*inv(sigma))*inner';
inner = inner * -0.5;
outer = 1/(sqrt(det(sigma)));
score = outer*exp(inner);
end