function [value,isterminal,direction] = terminate_event(t,z,sys)
% set up the event, if the cost is larger than 10000, the system is not
% stable. terminate.
global q
%  if q==4
%      disp(q)
%  end
 value=1;
%      if double(z(3)> 10000000)
%          value = 0; %unstable
%      end
     if q==3 && norm(z(1:2,1))<0.01;
         value=0; % reach the target
     end
     if z(end) >sys.maxcost
         value=0;
     end

     isterminal = 1; % stop the integration  at value = 0;
     direction = 0 ; %     
end

