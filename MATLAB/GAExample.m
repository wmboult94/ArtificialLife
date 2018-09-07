%A simple GA
clear;
NoIndviduals = 100; NoGenes = 4; NoGenerations = 100;mutationRate= 1/NoGenes;
%make random set of binary genotypes
Pop = floor(rand(NoIndviduals,NoGenes)+0.5);

%Random target String
TargetString = floor(rand(1,NoGenes)+0.5);

%Interaction matrix
W = [0 1 -1 -1; 1 0 -1 -1; -1 -1 0 1; -1 -1 1 0];

%evaluate iniial Fit
for i=1:NoIndviduals
    %Fit is sum squared error between individual and target string
    Fit(i) = NoGenes - sum(abs(TargetString-Pop(i,:)));
end

for i=1:NoGenerations
    %initalise temproray Pop
    New_Pop = [];newFit = [];
    
    [rankId,rankFit] = sort(Fit);
    %the roullette wheel
    Wheel = cumsum(1:NoIndviduals); max_wheel = sum(1:NoIndviduals);
    
    %elitism (add the best individual to new poosuaklio
    
    [maxFit,maxInd] = max(Fit);
    bestFit(i) = maxFit;
    New_Pop(1,:) = Pop(maxInd,:);
    newFit(1) = maxFit;
    
    %roullette wheel selection
    
    %to turn of elitsim change 2->1 in the following line
    for j=2:NoIndviduals
        
        
        
        %pick first individual
        pick1=rand*max_wheel;ind1=1;
        while(pick1>Wheel(ind1))
            ind1 = ind1+1;
        end
        %pick second individual
        pick2=rand*max_wheel;ind2=1;
        while(pick2>Wheel(ind2))
            ind2 = ind2+1;
        end
        
        ind1 = rankFit(ind1);
        ind2 = rankFit(ind2);
        %create daughter with no crossover
%           maxInd = max([ind1 ind2]);
%           daughter = Pop((maxInd),:);
        %create daughter from crossover
        crossOverPoint = floor(rand*NoGenes+1);
        parent1 = Pop((ind1),:); parent2 = Pop((ind2),:);
        daughter = [parent1(1:crossOverPoint) parent2((crossOverPoint+1):end)];
        
        %mutate the daughter
        for k=1:NoGenes
            if(rand < mutationRate)
                daughter(k) = not(daughter(k));
            end
        end
        
        New_Pop(j,:) = daughter;
        
        %calculate new Fit
        newFit(j) = NoGenes - sum(abs(TargetString-daughter));
    end
    
     bestFit(i) = max(newFit);
      
    %set old Pop to new popualtion
    Pop = New_Pop;Fit= newFit;
end
plot(bestFit);
xlabel('Generations')
ylabel('Fit')