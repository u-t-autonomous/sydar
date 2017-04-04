function [orderW,sorted] = resort(W, J)
% RESORT Summary of this function goes here
    [sorted, indices] = sort(J, 'descend');
    % get your output with
    orderW = W(indices,:);

end

