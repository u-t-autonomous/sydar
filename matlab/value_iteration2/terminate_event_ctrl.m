function [value,isterminal,direction] = terminate_event(t,z,sys)
% set up the event,  if the system converges to the origin (close), then
% stop.
     n= size(z,1);
     x= z(1:n-1,1);
     threshold = 0.001;
     value=1;
     if double( norm(x) < threshold)
         value = 0; % entering the origin, stop.
     end
     if double(z(3)> 10000)
         value = 0; %unstable
     end
     isterminal = 1; % stop the integration
     direction = 0 ; %
end

