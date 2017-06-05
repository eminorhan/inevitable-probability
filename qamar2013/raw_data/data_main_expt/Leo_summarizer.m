clear all; close all;
% This takes James's Leo's "LEO_ALL_20111212.csv" and converts it to our format

% Import the file
newData1 = importdata('Leo_summary_raw.csv');
% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end

% Instead of writing a hash function, we use a more primitive approach
% to discern session numbers from the strings provided in "textdata"
compare = @(a,b) logical ( (numel(a)==numel(b)) * ( sum(a==b) == numel(a) ) );
session = zeros(1,size(data,1));
previous = textdata{2};
enumerate_sess = 1;
for i = 1:size(data,1)
    if ~compare(previous, textdata{i+1})
        enumerate_sess = enumerate_sess + 1;
    end
    session(i) = enumerate_sess;
    previous = textdata{i+1};
end

% Now define all the other variables
class = data(:,3)' - 1; % ensures class is 0/1
contrast = data(:,4)';
orientation = data(:,1)' - 270; % centers data to mean 0
%-%-% If "response" is given as correct/incorrect, convert to class %-%-%
%xnor = @(p,q) (p&q) == (p|q); % used in next line
%response = xnor(c0000000000000000lass,data(:,2)') ; % convert correct/incorrect to reported class
response = data(:,2)' - 1; % ensures response is 0/1


if 1 % Get rid of sessions that have less than 500 trials
    us = unique(session); idx1 = zeros(size(orientation));
    session_original = session;
    for i = 1:numel(us)
        if sum(session_original == us(i)) < 1000
            idx1 = idx1 + (session_original == us(i));
            session(session_original > us(i)) = session( session_original > us(i) ) - 1; % Ensures session labels are contiguous
        end
    end
    idx1 = ~logical(idx1);
else
    idx1 = ones(size(orientation)); % Disregard above
end

% Get rid of contrasts that have less than 500 trials
uc = unique(contrast); idx2 = zeros(size(orientation));
for i = 1:numel(uc)
    if sum(contrast == uc(i)) < 1000
        idx2 = idx2 + (contrast == uc(i));
    end
end
idx2 = ~logical(idx2);

class = class(idx1 & idx2);
contrast = contrast(idx1 & idx2);
response = response(idx1 & idx2);
session = session(idx1 & idx2);
orientation = orientation(idx1 & idx2);

if 0 % Use real sessions (0) or virtual (1)
    % Cut off trials to mod max(session)
    idx1 = 1:floor(numel(session)/max(session))*max(session);
    class = class(idx1);
    contrast = contrast(idx1);
    response = response(idx1);
    session = session(idx1);
    orientation = orientation(idx1);
    
    % Relabel with virtual session
    session = repmat(1:max(session),1,floor(numel(session)/max(session)));
    % Randomize virtual session labels
    session = session(randperm(numel(session)));
end

if 1 % Combine all sessions into one big dataset
    session = ones(size(session));
end

% Rid of unused variables and save
clear compare enumerate_sess i newData1 previous textdata vars data xnor idx1 idx2 us uc session_original
save Leo_summary


if 0
    % see plot counts per session
    us = unique(session); counters = zeros(size(us));
    for i = 1:numel(us)
        counters(i) = sum(session==us(i));
    end
    figure; plot(us,counters); xlabel('session'); ylabel('count');
end


if 0
    % see plot counts per contrast
    uc = unique(contrast); counterc = zeros(size(uc));
    for i = 1:numel(uc)
        counterc(i) = sum(contrast==uc(i));
    end
    figure; plot(uc,counterc); xlabel('contrast'); ylabel('count');
end


if 0
    % diagnostic plots
    [h1, x1] = hist(orientation(class==0),100); h1 = h1/sum(h1);
    [h2, x2] = hist(orientation(class==1),100); h2 = h2/sum(h2);
    
    figure;
    plot(x1, h1,'r'); hold on;
    plot(x2, h2); hold on;
    legend('Class 1','Class 2')
    
    [h1, x1] = hist(orientation(response==0),100); h1 = h1/sum(h1);
    [h2, x2] = hist(orientation(response==1),100); h2 = h2/sum(h2);
    
    figure;
    plot(x1, h1,'r'); hold on;
    plot(x2, h2); hold on;
    legend('Report Class 1','Report Class 2')
end