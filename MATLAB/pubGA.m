%The pub GA
clear;
NoIndividuals = 10; NoGenes = 20; NoGenerations = 100; mutationRate= 1/NoGenes;
% Crossover probability
NoCrossProb = 1;
CrossProb = 0.5;
NoRepetitions = 50;

%[meanFitnessNC,meanIterationsNC] = part1(NoIndividuals,NoGenes,NoGenerations,mutationRate,NoCrossProb,NoRepetitions);
%[meanFitness,meanIterations] = part1(NoIndividuals,NoGenes,NoGenerations,mutationRate,CrossProb,NoRepetitions);
%part2(NoIndividuals,NoGenes,NoGenerations,mutationRate,NoRepetitions);
part3(NoGenes,NoGenerations,mutationRate,NoRepetitions);

function [meanF,meanI] = part1(NoIndividuals,NoGenes,NoGenerations,mutationRate,Pc,NoRepetitions)

    for n=1:NoRepetitions
        %Make random set of genotypes, alleles are 1 and -1
        Pop = floor(rand(NoIndividuals,NoGenes)+0.5);
        Pop(Pop==0) = -1;

        %%Build randomly initialised interaction matrix of 1s and 0s
        W = floor(rand(NoGenes,NoGenes)+0.5);
        % Bit of a hacky method, set all 0s to -1s
        W(W==0) = -1;
        % Make into diagonal matrix
        W = W - tril(W,-1) + triu(W,1)';
        % Set diagonal to 0
        W(logical(eye(size(W))))=0;

        %temp = Pop(1,:) * W * transpose(Pop(1,:));

        %Evaluate initial Fit
        for i=1:NoIndividuals  
            Fit(i) = Pop(i,:) * W * transpose(Pop(i,:));
        end

        for i=1:NoGenerations
            for k=1:NoIndividuals

                % Tournament selection
                ind1 = floor(rand*NoIndividuals+1);
                ind2 = floor(rand*NoIndividuals+1);
                P1 = Pop(ind1,:);
                P2 = Pop(ind2,:);

                if( Fit(ind1)>Fit(ind2) )
                    Winner = P1;
                    Loser = P2; 
                    LoserInd = ind2;
                else
                    Winner = P2;
                    Loser = P1; 
                    LoserInd = ind1;
                end

                % Perform crossover and mutation
                for j=1:NoGenes
                    if( rand < Pc )
                        Pop(LoserInd,j) = Winner(j);
                    end
                    if( rand < mutationRate )
                        Pop(LoserInd,j) = -Loser(j);
                    end
                end

                % Pop(LoserInd,:) = Loser;

                %Calculate new genotype's fitness, add best fit of generation to bestFit
                Fit(LoserInd) = Pop(LoserInd,:) * W * transpose(Pop(LoserInd,:));

            end
            [maxFit,maxInd] = max(Fit);
            bestFit(i) = maxFit;
            bestGeno = Pop(maxInd,:);
        end
        finalFit = bestFit(end);
        meanFit(n) = finalFit;
        
        plot(bestFit);
        xlabel('Generations')
        ylabel('Fit')

        % Sliding window technique to find how quickly GA arrives at a maximum
        % Ref https://stackoverflow.com/questions/22664897/ask-for-matlab-code-to-detect-steady-state-of-data
        difs = abs(diff(bestFit));
        % Use sliding window to find windows of consecutive elements below threshold
        steady = nlfilter(difs, [1, 5], @(x)all(x == 0));
        % Find where steady state starts (1) and ends (-1)
        start = diff(steady);
        % Return indices of starting steady state
        ind = find(start == 1);
        % Maximum found by GA starts at the max steady state index
        if(max(ind))
                meanIters(n) = max(ind);
        else meanIters(n) = NoGenerations;
        end
    end
    meanF = sum(meanFit) / length(meanFit);
    meanI = sum(meanIters) / length(meanIters);
end

function part2(NoIndividuals,NoGenes,NoGenerations,mutationRate,NoRepetitions)
    
    CrossRange = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2];
    for n=1:length(CrossRange)
        for m=1:NoRepetitions
            %Make random set of genotypes, alleles are 1 and -1
            Pop = floor(rand(NoIndividuals,NoGenes)+0.5);
            Pop(Pop==0) = -1;

            %%Build randomly initialised interaction matrix of 1s and 0s
            W = floor(rand(NoGenes,NoGenes)+0.5);
            % Bit of a hacky method, set all 0s to -1s
            W(W==0) = -1;
            % Make into diagonal matrix
            W = W - tril(W,-1) + triu(W,1)';
            % Set diagonal to 0
            W(logical(eye(size(W))))=0;

            %temp = Pop(1,:) * W * transpose(Pop(1,:));

            %Evaluate initial Fit
            for i=1:NoIndividuals  
                Fit(i) = Pop(i,:) * W * transpose(Pop(i,:));
            end

            for i=1:NoGenerations
                for k=1:NoIndividuals

                    % Tournament selection
                    ind1 = floor(rand*NoIndividuals+1);
                    ind2 = floor(rand*NoIndividuals+1);
                    P1 = Pop(ind1,:);
                    P2 = Pop(ind2,:);

                    if( Fit(ind1)>Fit(ind2) )
                        Winner = P1;
                        Loser = P2; 
                        LoserInd = ind2;
                    else
                        Winner = P2;
                        Loser = P1; 
                        LoserInd = ind1;
                    end

                    % Perform crossover and mutation
                    for j=1:NoGenes
                        if( rand < CrossRange(n) )
                            Pop(LoserInd,j) = Winner(j);
                        end
                        if( rand < mutationRate )
                            Pop(LoserInd,j) = -Loser(j);
                        end
                    end

                    % Pop(LoserInd,:) = Loser;

                    %Calculate new genotype's fitness, add best fit of generation to bestFit
                    Fit(LoserInd) = Pop(LoserInd,:) * W * transpose(Pop(LoserInd,:));

                end
                [maxFit,maxInd] = max(Fit);
                bestFit(i) = maxFit;
                bestGeno = Pop(maxInd,:);
            end
            finalFit = bestFit(end);
           
            %plot(bestFit);
            %xlabel('Generations')
            %ylabel('Fit')

            % Sliding window technique to find how quickly GA arrives at a maximum
            % Ref https://stackoverflow.com/questions/22664897/ask-for-matlab-code-to-detect-steady-state-of-data
            difs = abs(diff(bestFit));
            % Use sliding window to find windows of consecutive elements below threshold
            steady = nlfilter(difs, [1, 5], @(x)all(x == 0));
            % Find where steady state starts (1) and ends (-1)
            start = diff(steady);
            % Return indices of starting steady state
            ind = find(start == 1);
            % Maximum found by GA starts at the max steady state index
            if(max(ind))
                meanIters(m) = max(ind);
            else meanIters(m) = NoGenerations;
            end
        end
        meanItersPerPc(n) = sum(meanIters) / length(meanIters);
    end
    plot(range,meanItersPerPc);
    xlabel('Fraction of genotype passed from winning parent to offspring');
    ylabel('Average num iterations taken to reach max');
end

function part3(NoGenes,NoGenerations,mutationRate,NoRepetitions)
    
    PopRange = [10,20,30,40,50,60,70,80,90,100];
    CrossProb = [1,0.5];
    ItersNCandC = zeros(2,length(PopRange));
    for cross=1:2
        meanItersPopRange = zeros(1,length(PopRange));
        for n=1:length(PopRange)

            for m=1:NoRepetitions
                %Make random set of genotypes, alleles are 1 and -1
                Pop = floor(rand(PopRange(n),NoGenes)+0.5);
                Pop(Pop==0) = -1;

                %%Build randomly initialised interaction matrix of 1s and 0s
                W = floor(rand(NoGenes,NoGenes)+0.5);
                % Bit of a hacky method, set all 0s to -1s
                W(W==0) = -1;
                % Make into diagonal matrix
                W = W - tril(W,-1) + triu(W,1)';
                % Set diagonal to 0
                W(logical(eye(size(W))))=0;

                %temp = Pop(1,:) * W * transpose(Pop(1,:));

                %Evaluate initial Fit
                for i=1:PopRange(n)  
                    Fit(i) = Pop(i,:) * W * transpose(Pop(i,:));
                end

                for i=1:NoGenerations
                    for k=1:PopRange(n)

                        % Tournament selection
                        ind1 = floor(rand*PopRange(n)+1);
                        ind2 = floor(rand*PopRange(n)+1);
                        P1 = Pop(ind1,:);
                        P2 = Pop(ind2,:);

                        if( Fit(ind1)>Fit(ind2) )
                            Winner = P1;
                            Loser = P2; 
                            LoserInd = ind2;
                        else
                            Winner = P2;
                            Loser = P1; 
                            LoserInd = ind1;
                        end

                        % Perform crossover and mutation
                        for j=1:NoGenes
                            if( rand < CrossProb(cross) )
                                Pop(LoserInd,j) = Winner(j);
                            end
                            if( rand < mutationRate )
                                Pop(LoserInd,j) = -Loser(j);
                            end
                        end

                        % Pop(LoserInd,:) = Loser;

                        %Calculate new genotype's fitness, add best fit of generation to bestFit
                        Fit(LoserInd) = Pop(LoserInd,:) * W * transpose(Pop(LoserInd,:));

                    end
                    [maxFit,maxInd] = max(Fit);
                    bestFit(i) = maxFit;
                    %bestGeno = Pop(maxInd,:);
                end
                finalFit = bestFit(end);
                meanFit(m) = finalFit;

                %plot(bestFit);
                %xlabel('Generations')
                %ylabel('Fit')

                % Sliding window technique to find how quickly GA arrives at a maximum
                % Ref https://stackoverflow.com/questions/22664897/ask-for-matlab-code-to-detect-steady-state-of-data
                difs = abs(diff(bestFit));
                % Use sliding window to find windows of consecutive elements below threshold
                steady = nlfilter(difs, [1, 4], @(x)all(x == 0));
                % Find where steady state starts (1) and ends (-1)
                start = diff(steady);
                % Return indices of starting steady state
                ind = find(start == 1);
                % Maximum found by GA starts at the max steady state index
                if(max(ind))
                    meanIters(m) = max(ind);
                else meanIters(m) = NoGenerations;
                end

            end
            meanItersPopRange(n) = sum(meanIters) / length(meanIters);
            meanFitPopRange(n) = sum(meanFit) / length(meanFit);
        end
%         TODO: work out how to get both nc and c onto one plot
        ItersNCandC(cross,:) = meanItersPopRange;
        meanFitNCandC(cross,:) = meanFitPopRange;
    end
    
    plot(PopRange,ItersNCandC(1,:),'color','b'); hold on;
    plot(PopRange,ItersNCandC(2,:),'color','r');
    xlabel('Population size');
    ylabel('Average num iterations taken to reach max');
    %plot(range,meanFitPopRange);
    %xlabel('Population size');
    %ylabel('Average max fitness');
    
    meanC = sum(ItersNCandC(1,:)) / length(ItersNCandC(1,:));
    meanNC = sum(ItersNCandC(2,:)) / length(ItersNCandC(2,:));
    disp(meanC);disp(meanNC);
    disp(meanFitNCandC);
    disp(sum(meanFitNCandC)/length(meanFitNCandC));
end