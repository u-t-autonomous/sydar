function [next_state] =  get_trans_ex(state, x)
% A: x_1 <= 1
% B: x_1 > 1
% C: norm(x)< 0.1
%
switch state
    case 0
        if x(1)> 1;
            next_state=2;
        else  
            if x(1)<=1;
                next_state = 1;
            else
                next_state= 0;
            end
        end
            case 1
                if x(1) <=1;
                    next_state = 1;
                else
                    next_state = 3;
                end
                case 2
                    if x(1) > 1
                        next_state = 2;
                    else
                        next_state = 3;
                    end
                    case 3
                        if norm(x) <0.01
                            disp('should terminates')
                            next_state= 3;
                        else
                            next_state = 3;
                        end
        end
        %
        %         if state == 0 &&  x(1) > 1; % visit B
        %     next_state = 2 ;
        % else
        %     if state ==2 && x(1) >1;
        %     next_state = 2;
        %
        %     else
        %
        % if state ==2 && x(1) <=1;
        %     next_state = 3;
        % else
        % if state == 0 && x(1) <=1; % visit A
        %     next_state = 1;
        % else
        % if state ==1 && x(1) <=1; % visit A
        %     next_state = 1;
        % else
        % if state ==1 && x(1) >1; % visit B
        %     next_state = 3;
        % else
        % if state ==3
        %     if norm(x) < 0.1;
        %     next_state = 4;
        %     return
        %     else
        %         next_state=3;
        %         return
        %     end
        % else
        %     if state == 4
        %     next_state=4;
        % end
end