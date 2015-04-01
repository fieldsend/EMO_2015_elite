function [I_dom, X, Y_mo, Y_n, Y_dom, resamp_hist] = ...
    incremental_noisy_DEMO(evaluations,cost_function,domain_function,l,num_obj,p_cross,func_arg,update_type,search_proportion,check_num)

% function [I_dom, X, Y_mo, Y_n, Y_dom] = incremental_noisy_DEMO(evaluations,cost_function,domain_function,l,num_obj,std_mut,p_cross,func_arg,update_type,search_proportion)
%
% An iterative implementation of the DEMO algorithm to handle noisy multi-
% objective problems.
%
% Code relates to:
%
% T. Robic and B. Filipic (2005),
% DEMO: Differential evolution for multiobjective optimization.
% In Evolutionary Multi-Criterion Optimization, pages 520-533. Springer.
%
% J. E. Fieldsend (2015)
% Elite Accumulative Sampling Strategies for Noisy Multi-Objective Optimisation
% In Evolutionary Multi-Criterion Optimization, Part II, pages 172-186, LNCS 9019, Springer.
%
% Please cite the above work if you use this code
%
% Designed for real-valued optimisation problems where there is
% observational noise -- assumes noise is symmetric with the mean being the
% appropriate ML estimator (e.g. Guassian).
%
% The extension dynamically adapts the number of resamples it undertakes,
% based upon comparing the estimated Pareto fronts maintained at lagged
% time-steps.
%
% If your problem is constrained (beyond simply bounded), then set p_cross to
% 0.0 as currently the constraint and bound checks are only in the
% mutation operator
%
% Uses the single_link_guardian_iterative function, which is also available
% from my github page in the gecco_2014_changing_objectives repository
%
% INPUTS:
%
% evalutions = total number of function evaluations
% cost_function = string with name of multi-objective function to be
%          optimised, should take a design vector and the func_arg
%          structure as arguments and return a 1 by num_obj vector of
%          corresponding objective values (which are probably noisy)
% domain_function = string with name of function to check legality of
%          solution, should take a design vector and the func_arg structure
%          as arguments and return 1 if the design is legal, 0 otherwise
% l = number of design variables
% num_obj = number of objectives
% p_cross = probability of crossover
% func_arg = struture of cost function meta-parameters, also includes
%          domain bounds: fung_arg.upb, func_arg.lwb and func_arg.range
%          should hold the lower bounds, upper bounds and range of the
%          design variables in vectors
% update_type = value indicating accumulative update type used. A value of
%          1 will assign one elite resample for each one new design each
%          generation. A value of 2 increments the number of resamples taken
%          at each generation, if the front estimate has oscillated from
%          check_num time steps previously A value of 3 increments the
%          minimum number of resamples each elite member must have on
%          generation completion, incremented if the front estimate has
%          oscillated from check_num time steps previously. A value of 4
%          ensures the average number of resamples per elite member is
%          always higher than the average number averages across all time
%          steps.
%
% search_proportion = proportion of time steps where searching is an option
%          (rather than only refining elite members). Default to 1.0
% check_num = (minimum) number of time steps before each ossilation check
%
% OUTPUTS:
%
% I_dom = indices of estimated Pareto set members of X and corresponding
%          Pareto front members of Y_mo
% X = All design locations evaluated in the optimisation run
% Y_mo = (mean) objective vectors associated with the elements in X
% Y_n = number of reevaluations taken at each member of X in order to
%          generate Y_mo elements
% Y_dom = index of member of X which 'guards' each member (a value of 0
%          meaning it is not guarded, and therefore is in the estimated
%          Pareto set). See the single_link_guardian_iterative function
%          for further details
% resamp_hist = matrix holding evolution information. First column, number 
%          of resamples per gen, second column av number resamples per
%          elite member, third column av number resamples across all design
%          locations that have been evaluated, fourth column archive size,
%          fifth column, indicator of whether stats taken at end of a
%          generation.
%
%
% (c) Jonathan Fieldsend 2012-2015
% Version 1.0

if (evaluations<100)
    error('As the algorithm samples 100 points initially, evaluations must be at least 100');
end
if (l<1)
    error('Cannot have zero or fewer design variables');
end
if (num_obj<1)
    error('Cannot have zero or fewer objectives');
end
if (p_cross<0)
    error('Crossover probability should not be negative');
end
if (exist('search_proportion', 'var')==0)
    search_proportion = 1.00; %do not use any evaluations at end exclsuively to hone estimate
end
if (exist('check_num', 'var')==0)
    check_num = 100; % check advancement every check_num function evaluations
end
if (search_proportion<0)
    error('Cannot have a negative search proportion term');
end
if (search_proportion>1)
    error('Cannot have a search proportion term larger than 1');
end


pop_size = 100; % initial number of random designs/search population size
reps = 1; % number of reevaluations per iteration

% predetermine the number of designs visited, due to the multiplication
% through by the decimal search_proportion term and the ceil this may over
% estimate by a one or two, so at the end of the run we empty the unused
% final elements of the preallocated matrices
unique_locations = 100+ceil(((evaluations-100)/2)*search_proportion)+1;

% preallocate matrices for efficiency
X = zeros(unique_locations,l); % all locations evaled
Y_mo = zeros(unique_locations,num_obj); % mean criteria values associated with solutions
Y_n = zeros(unique_locations,1); % number of revaluations of this particular solution
Y_dom = zeros(unique_locations,1); % index of set member which dominates this one.
resamp_hist = zeros(unique_locations,5); % track the number of reevaluations over time and other statistics

% initialise holders for sets to use later
epsilons.m_curr_old = [];
epsilons.m_old_curr = [];
epsilons.a_curr_old = [];
epsilons.a_old_curr = [];
% now sample 100 points
% propogate first sample to initialise archive
X(1,:) = generate_legal_initial_solution(domain_function, l, func_arg);
Y_mo(1,:)=evaluate_f(cost_function,X(1,:),num_obj,1,func_arg);
Y_n(1) = 1;
I_dom = 1;
index = 2; % index of next element to add

% track initial Pareto front
old_front = Y_mo(I_dom,:);

% sample rest of initial population
for i=2:pop_size;
    x = generate_legal_initial_solution(domain_function, l, func_arg);
    [X, Y_mo, Y_n, Y_dom, index, I_dom] = evaluate_and_set_state(...
        cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n, Y_dom, index, -1, I_dom,i,reps);
end
Pop_indices = 1:pop_size; % search population indices
evals = pop_size; % keep track of evaluations used

resamp_hist(1:pop_size,:) = reps;
% OPTIMISATION LOOP
track=0;

old_n = 1; % after initialisation all archived points have been sampled 1 time
old_n_index = 1;
while (evals < evaluations) % while algorithm has not exhausted the number of function evaluations permitted
    % propose a new design if still in search mode
    if (evals<=evaluations*search_proportion)
        [x, ~, Pop_indices, varied_index] = incremental_DE_evolve(X, p_cross, func_arg, Pop_indices, pop_size);
        [X, Y_mo, Y_n, Y_dom, index, I_dom] = evaluate_and_set_state(...
            cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, -1,I_dom,evals,reps);            
        
        
        [Pop_indices] = incremental_DE_update(Pop_indices, Y_mo, I_dom,varied_index, index-1, num_obj); % update search population if required
        
        [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo);
        track = track+1;
    end
    
    % update estimate of an existing elite design
    
    if update_type==2 % do 'reps' reevaluations each generation
        for k=1:reps
            if evals<evaluations
                [~,II] = min(Y_n(I_dom));
                copy_index = I_dom(II(1));
                x = X(copy_index,:);
                % now see if membership of I_dom should be changed
                [X, Y_mo, Y_n,Y_dom, index, I_dom] = evaluate_and_set_state(...
                    cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, copy_index,I_dom,evals,reps);
                [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo);
            end
        end
    else
        % do a single reevaluation update type always for 1, 3 & 4
        if evals<evaluations
            [~,II] = min(Y_n(I_dom));
            copy_index = I_dom(II(1));
            x = X(copy_index,:);
            % now see if membership of I_dom should be changed
            [X, Y_mo, Y_n,Y_dom, index, I_dom] = evaluate_and_set_state(...
                cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, copy_index,I_dom,evals,reps);
            [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo);
        end
        if update_type==3
            % ensure all front members are evaluated at least reps times
            while min(Y_n(I_dom))<reps && evals<evaluations
                [~,II] = min(Y_n(I_dom));
                copy_index = I_dom(II(1));
                x = X(copy_index,:);
                % now see if membership of I_dom should be changed
                [X, Y_mo, Y_n,Y_dom, index, I_dom] = evaluate_and_set_state(...
                    cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, copy_index,I_dom,evals,reps);
                [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo);
            end
        end
        if update_type == 4
            while old_n >= mean(Y_n(I_dom)) % don't resample if average resample number is larger than average to this point
                [~,II] = min(Y_n(I_dom));
                copy_index = I_dom(II(1));
                x = X(copy_index,:);
                % now see if membership of I_dom should be changed
                [X, Y_mo, Y_n,Y_dom, index, I_dom] = evaluate_and_set_state(...
                    cost_function,x,num_obj,1,func_arg, X, Y_mo, Y_n,Y_dom,index, copy_index,I_dom,evals,reps);
                [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo);
            end
            if old_n_index==501;
                old_n_index = 1;
            end
            old_n = old_n + (mean(Y_n(I_dom))-old_n)/(index-1);
        end  
    end
    resamp_hist(evals-1,5) = 1; % track stats corresponding to end of generation
    
    % If check_num or more evals since last oscillation check, check for
    % oscillation
    if track >= check_num
        [epsilons, old_front, contrary] = ...
            check_advance(epsilons, old_front, Y_mo, I_dom);
        track = track - check_num;
        if contrary
            reps =  reps + 1;
        end
    end
end

% remove unused matrix elements
X(index:end,:) = [];
Y_mo(index:end,:) = [];
Y_dom(index:end) = [];
Y_n(index:end) = [];

%---------------
function [evals, resamp_hist] = update_stats(evals, resamp_hist, reps, Y_n, I_dom, index, Y_mo)

evals = evals+1;
resamp_hist(evals,1) = reps;
resamp_hist(evals,2) = mean(Y_n(I_dom));
resamp_hist(evals,3) = mean(Y_n(1:index-1));
resamp_hist(evals,4) = length(I_dom);

% can put plotting code in here using Y_mo if you want to see how the
% population evols, e.g. for two objective dimensions:

% scatter(Y_mo(1:index-1,1),Y_mo(1:index-1,2),20,Y_n(1:index-1),'fill'); colorbar; colormap cool
% hold on
% plot(Y_mo(I_dom,1),Y_mo(I_dom,2),'ko'); % mark Pareto set estimate
% plot(Y_mo(index-1,1),Y_mo(index-1,2),'kx'); % mark evaluated point
% hold off

%---------------
function [X, Y_mo, Y_n, Y_dom,index, I_dom] = evaluate_and_set_state(...
    cost_function,x,num_obj,initial_reevals,func_arg, X, Y_mo, Y_n, Y_dom, index, prev_index, I_dom,evals,reps)

% index  = index of new deign
% prev_index = if a a reassessment, previous location, otherwise set at -1

[x_objectives]=evaluate_f(cost_function,x,num_obj,initial_reevals,func_arg);

if (prev_index == -1) % if it is a new design
    X(index,:) = x;
    Y_mo(index,:) = x_objectives;
    Y_n(index) = 1;
    % update leading edge, and maintain single link guardian data structure
    [Y_mo,Y_dom,I_dom] = single_link_guardian_iterative(Y_mo,Y_dom,I_dom,index,0,[1 2 1 2 2 2]);
else % it is a reevaluation, so incrementally update mean estimate
    Y_n(prev_index) = Y_n(prev_index)+1;
    Y_mo(prev_index,:) = Y_mo(prev_index,:) + (x_objectives-Y_mo(prev_index,:))/Y_n(prev_index);
    % update leading edge, and maintain single link guardian data structure
    [Y_mo,Y_dom,I_dom] = single_link_guardian_iterative(Y_mo,Y_dom,I_dom,index,prev_index,[1 2 1 2 2 2]);
end

% print details every 500 evaluations
if (rem(evals,500)==0)
    fprintf('Evals %d elite %d av elite reevals %f av pop reevals %f reps %d designs %d\n', evals, length(I_dom), mean(Y_n(I_dom)), mean(Y_n(1:index-1)),reps,index);
end

% update index tracking number of evaluations so far
if prev_index ==-1
    index=index+1;
end

%--------------------------------------------------------------------------
function distances = calc_crowding_dist(X)

[n,m] = size(X);
distances = zeros(n,1);
for j=1:m
    [~,Im] = sort(X); % get index of sorted solutions on each dimension
    distances(Im(1)) = inf;
    distances(Im(end)) = inf;
    for i=2:n-1;
        distances(Im(i)) = distances(Im(i)) + (X(Im(i+1),j)-X(Im(i-1),j));
    end
end

%--------------------------------------------------------------------------
function x = generate_legal_initial_solution(domain_function, l, func_arg)

%generate an initial legal solution
illegal =1;
while (illegal)
    x = rand(1,l);
    x = x.*func_arg.range+func_arg.lwb; %put on data range
    if( feval(domain_function,x,func_arg) ) %check if legal
        illegal = 0;
    end
end

%--------------------------------------------------------------------------
function [results]=evaluate_f(cost_function,c,num_obj,num_reevaluations,func_arg)

% repeatedly evaluate the solution 'c' num_reevaluations times

results=zeros(num_reevaluations,num_obj); %preallocate matrix for efficiency
for i=1:num_reevaluations
    results(i,:)=feval(cost_function,c,num_obj,func_arg);
end

%--------------------------------------------------------------------------
function [epsilons,old_front,contrary] = ...
    check_advance(epsilons, old_front, Y_mo, I_dom)

% function compares Pareto front state between each check_num evaluations
[no,~] = size(old_front);
current_front = Y_mo(I_dom,:);
[nc,~] = size(current_front);
max_v = max(max(old_front), max(current_front)); % get max values of bounding box containing old_front and current_front
min_v = min(min(old_front), min(current_front)); % get min values of bounding box containing old_front and current_front
range = max_v-min_v;
normalised_old_front = old_front./repmat(range,no,1) + repmat(min_v,no,1);
normalised_current_front = current_front./repmat(range,nc,1) + repmat(min_v,nc,1);

epsilon_add_curr_dom_old = epsilon_indicator_additive(normalised_current_front,normalised_old_front);
epsilon_add_old_dom_curr = epsilon_indicator_additive(normalised_old_front,normalised_current_front);

epsilons.a_old_curr = [epsilons.a_old_curr epsilon_add_old_dom_curr];
epsilons.a_curr_old = [epsilons.a_curr_old epsilon_add_curr_dom_old];
contrary = 0;
if epsilon_add_old_dom_curr<epsilon_add_curr_dom_old
    contrary = 1;
end
% replace old front with current front to use next time
old_front = current_front;

%--------------------------------------------------------------------------
function epsilon=epsilon_indicator_additive(Y1,Y2)

% returns the additive epsilon needed for Y1 to dominate Y2

[n1,m1] = size(Y1);
[n2,m2] = size(Y2);
if (m1 ~= m2)
    error('Compared sets do not have the same number of objectives');
end
indicator = zeros(n2,1);

for i=1:n2
    t = max(Y1-repmat(Y2(i,:),n1,1),[],2); % get max shift value for member across objectives
    indicator(i) = min(t); %identify the minimum shift of Y1 in order for a member of it to dominate ith member of Y2
end

epsilon = max(indicator)+eps; % get minimum shift required to cover all members in Y2 by Y1
% using eps term as had some floating point
% precision errors when testing

%--------------------------------------------------------------------------
function [x, copy_index, Pop_indices, varied_index] = incremental_DE_evolve(X,p_cross,func_arg, Pop_indices, pop_size)

% incremental DEMO/parent
I = randperm(pop_size);
varied_index = I(1);
copy_index = Pop_indices(I(1));
x = X(copy_index,:); % always make base a non-dominated solution
% now select a, b and c, randomly 
a = X(Pop_indices(I(2)),:);
b = X(Pop_indices(I(3)),:);
c = X(Pop_indices(I(4)),:);


differential_weight = 0.5;
for i=1:length(x)
    if rand() < p_cross
        x(i) = a(i) + differential_weight*(b(i)-c(i));
    end
end
% ensure child is legal
x(x<func_arg.lwb) = func_arg.lwb(x<func_arg.lwb);
x(x>func_arg.upb) = func_arg.upb(x>func_arg.upb);

%--------------------------------------------------------------------------
function [Pop_indices, me] = incremental_DE_update(Pop_indices, Y_mo, I_dom, varied_index, child_index, num_obj)

pop_size = length(Pop_indices);

if sum(Y_mo(child_index,:)<=Y_mo(Pop_indices(varied_index),:))== num_obj % dominates parent - replace
    Pop_indices(varied_index) = child_index;
elseif sum(Y_mo(child_index,:)>=Y_mo(Pop_indices(varied_index),:))~= num_obj
    % mutually non-dominating, so need to analyse effect of using one or
    % other in set
    
    Pop_indices = [Pop_indices child_index]; % add to population
    % then calculate crowding and remove most crowded  element on worst
    % shell
    ranks = recursive_pareto_shell_with_duplicates(Y_mo(Pop_indices,:),0);
    % identify those in last shell
    last_shell_indices = find(ranks==max(ranks));
    distances = calc_crowding_dist(Y_mo(Pop_indices,:));
    [~, I_rem] = sort(distances(last_shell_indices));
    Pop_indices(last_shell_indices(I_rem(1)))=[]; % remove elements in last shell which have the smallest (i.e. closest) value
end % if neither case is true, then it is dominated by the parent, so the set is unchanged

% ensure that Pareto set members are not omitted from Population due to
% front oscillation over time
me =0;
missing_elite = setdiff(I_dom,Pop_indices);

if isempty(missing_elite)==0 % if some elite members are missing from Pop_indices
    me=1;
    %   display('DE population has oscillated -- fixing');
    Pop_indices = [Pop_indices missing_elite]; %add to population
    % then ranks and crowding and remove most crowded elements on worst
    % shell
    ranks = recursive_pareto_shell_with_duplicates(Y_mo(Pop_indices,:),0);
    new_Pop_indices = [];
    rank_index = 0;
    while length(new_Pop_indices) < pop_size;
        old_size = length(new_Pop_indices); % keep track of how big before spills over
        new_Pop_indices = [new_Pop_indices Pop_indices(ranks==(rank_index))];
        rank_index = rank_index+1;
    end
    % calculate distances for crowding
    distances = calc_crowding_dist(Y_mo(new_Pop_indices,:));
    
    % fill remaining elements by sampling from the rank_index shell
    % according to crowding distances
    if (length(new_Pop_indices)>pop_size)
        curr_pop_len = length(new_Pop_indices);
        remove_number = curr_pop_len-pop_size;
        % have too many, so sort based first on rank, and then on
        % distance, as we have an index into last shell added we can
        % simply focus on these though :)
        
        [~, I_rem] = sort(distances(old_size+1:curr_pop_len));
        % I_rem gives indices of those to remove from smallest to largest
        % of the last shell added, need to shift index by previous shells
        % added though
        I_rem = I_rem + old_size;
        new_Pop_indices(I_rem(1:remove_number))=[];
    end
    Pop_indices = new_Pop_indices;
end
