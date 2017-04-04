function xy = ellip2d(P,r,t,c)
% returns set of points in 2d such that
% x'*P*x + 2*r'x + t == c

% See: http://ivanpapusha.com/pdf/matrix_ident.pdf (Section 2.1)

assert(all(size(P) == [2, 2]));
assert(all(size(r) == [2, 1]));
assert(all(size(t) == [1, 1]));
assert(all(size(c) == [1, 1]));

% symmetrize quadratic form
P = (P + P')/2;

% normalize to plot x'*P*x + 2*r'*x == 1
P = P / (c-t);
r = r / (c-t);

% find center
xc = -(P\r); 

% find offsets
theta = linspace(0,2*pi,1000);
dxy = [cos(theta); sin(theta)];
[V,D] = eig(inv(P));
dxy = V*sqrt(D)*dxy;
xy = bsxfun(@plus, dxy, xc);

end
