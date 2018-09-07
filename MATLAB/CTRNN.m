%Run a CTRNN
dt=0.01;
T=300;
time= 0:dt:T;

%no nodes
N = 50
%set weights randon=mly, with zero mean and differing variances
%set weight to 0 with probability Pc, ie average connectivity is 1 - Pc
mean = 0;
vnc = .53;
Pc = 0.7;
runs = 50;

Pcs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
vncs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8];
for n=1:length(Pcs)
    for i=1:length(vncs)
        converged = 0;
        for j=1:runs
            W = randn(N)*vncs(i) + mean;
            I = (rand(size(W)) < Pcs(n));
            W(I) = 0;

            %initial conditions
            y=zeros(N,length(time));
            y(1,1) = -1;
            y(2,1) = 1;

            %biases are zeros to start
            theta = floor(rand(N,1)+0.5)*0.0;

            %random input
            Iconf= floor(rand(N,1)+0.5);
            pulse_time =50;%time
            pulse_dur=10;%duration
            Imag=1.0;%magnitude
            I=0;
            for k=2:length(time)

                y(:,k) = y(:,k-1) +dt*(-y(:,k-1)+tanh( W*y(:,k-1) +theta+I));

                if(time(k) > pulse_time && time(k) < pulse_time+pulse_dur)
                    I=Imag*Iconf;
                else
                    I=0;
                end 
            end

            if(all(abs(y(:, end))< 1e-5)) 
                converged = converged + 1;
            end
            figure(1);clf;
            plot(time,y)
        end
        numConvPerVnc(i) = converged;
    end
    numConvPerPC(n,:) = numConvPerVnc;
end