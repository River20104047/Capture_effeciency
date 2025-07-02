%% This is used to calculate area and overlapped area of sampling
% 2025/06/12 v01 By Zijiang Yang

% Prepare workspace
clc, clear, close all

tic

% Dredge width
L = 2.75;  % meters

% Grid interval 
m = 0.5;  % meters

% Control flag to show geo map
showOnGeoMap = true;  % Set to false to disable map display

% File name
filename = 'A02_Smoothed GPS data - test.xlsx';

% Sheet names
sheets = {'E01', 'E02', 'E03', 'E04', 'E05'}; % Test

% Initialize arrays to hold all lat/lon data for plotting
allLat = {};
allLon = {};

% Loop through each sheet
for i = 1:length(sheets)
    sheetName = sheets{i};
    
    % Read Latitude and Longitude from columns P and Q
    data = readtable(filename, 'Sheet', sheetName);
    
    % Extract columns
    lat = data{:, 'Latitude_Smoothed'};
    lon = data{:, 'Longitude_Smoothed'};

    % Store raw coordinates as CorXX
    eval(sprintf('Cor%02d = [lat, lon];', i));
    
    % Store for geo plotting
    allLat{i} = lat;
    allLon{i} = lon;
end

% Optional: plot all tracks on one geo map
if showOnGeoMap
    figure
    gx = geoaxes;           % create geographic axes
    hold(gx, 'on')          % allow multiple tracks

    for i = 1:length(allLat)
        geoplot(gx, allLat{i}, allLon{i}, 'LineWidth', 1.5, 'DisplayName', sprintf('Cor%02d', i));
    end

    title(gx, 'All Tracks on Geo Map')
    geobasemap(gx, 'streets')
    legend(gx, 'show')
end


%% Relative coordinates
% Combine all coordinates
allCoords = [];
for i = 1:length(sheets)
    varName = sprintf('Cor%02d', i);
    coords = eval(varName);
    allCoords = [allCoords; coords];  % Append all [lat, lon]
end

% Find min latitude and min longitude
minLat = min(allCoords(:,1));
minLon = min(allCoords(:,2));

% Convert L (in meters) to degree shifts
deltaLat = ceil(L) / 111000;  % ~111 km per degree latitude
deltaLon = ceil(L) / (111000 * cosd(minLat));  % Adjust for latitude

% Shifted SW point
swLat = minLat - deltaLat;
swLon = minLon - deltaLon;

% Plot the shifted SW point
if showOnGeoMap
    geoscatter(gx, swLat, swLon, 20, 'o', ...
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'white', ...
        'LineWidth', 0.5, ...
        'DisplayName', 'SW Corner');
end

% Find max latitude and max longitude
maxLat = max(allCoords(:,1));
maxLon = max(allCoords(:,2));

% Convert L to delta (already done above, reuse deltaLat and deltaLon)

% Shifted NE point
neLat = maxLat + deltaLat;
neLon = maxLon + deltaLon;

% Plot the shifted NE point
if showOnGeoMap
    geoscatter(gx, neLat, neLon, 20, 'o', ...
        'MarkerEdgeColor', 'black', ...
        'MarkerFaceColor', 'white', ...
        'LineWidth', 0.5, ...
        'DisplayName', 'NE Corner');
end


% Define origin in lat/lon (shifted SW point)
originLat = swLat;
originLon = swLon;


utmProj = projcrs(32654);  % WGS84 / UTM zone 54N

% Convert all CorXX to UTM first and store
allXY = [];  % For finding min X and Y
for i = 1:length(sheets)
    varName = sprintf('Cor%02d', i);
    coords = eval(varName);
    lat = coords(:,1);
    lon = coords(:,2);
    
    % Convert to UTM using correct zone
    [x, y] = projfwd(utmProj, lat, lon);
    
    allXY = [allXY; x, y];  % collect for global min

    % Temporarily store full UTM for shift later
    eval(sprintf('utmXY%02d = [x, y];', i));
end

% Get the true SW point in meters
originX = min(allXY(:,1)) - ceil(L);
originY = min(allXY(:,2)) - ceil(L);

% Compute relative XY coordinates
for i = 1:length(sheets)
    xy = eval(sprintf('utmXY%02d', i));
    x_rel = xy(:,1) - originX;
    y_rel = xy(:,2) - originY;
    eval(sprintf('xy%02d = [x_rel, y_rel];', i));
end

% Optional: Plot all relative coordinates in local XY system
figure
hold on
axis equal
grid on
xlabel('X (meters)')
ylabel('Y (meters)')
title('All Tracks in Local XY Coordinate System')

for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));
    plot(xy(:,1), xy(:,2), 'LineWidth', 1.5, 'DisplayName', sprintf('xy%02d', i));
end

legend show

%% Points matrix
% Convert shifted NE corner (lat/lon) to UTM
[neX, neY] = projfwd(utmProj, neLat, neLon);

% Compute relative NE corner in local coordinate system
xyNE = [neX - originX, neY - originY];

% Define x and y ranges based on xyNE
xRange = 0 : m : xyNE(1);
yRange = 0 : m : xyNE(2);

% Generate grid
[X, Y] = meshgrid(xRange, yRange);
gridPoints = [X(:), Y(:)];  % N x 2 matrix of (x, y)

% Plot everything
figure
hold on
axis equal
grid on
xlabel('X (m)')
ylabel('Y (m)')
title('Sampling Tracks with Grid Points')

% 🔹 Plot grid points first
scatter(gridPoints(:,1), gridPoints(:,2), 5, 'o', ...
    'MarkerFaceColor', [0.7 0.7 0.7], ...
    'MarkerEdgeColor', [0.5 0.5 0.5], ...
    'LineWidth', 0.5, ...
    'DisplayName', 'Grid Points');

% 🔹 Then plot traces (on top)
for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));
    plot(xy(:,1), xy(:,2), 'LineWidth', 1.2, 'DisplayName', sprintf('xy%02d', i));
end

legend show
grid off


% Plot polygonal swath of each trace with grid overlay
figure
hold on
axis equal
grid on
xlabel('X (m)')
ylabel('Y (m)')
title('Polygonal Swath Areas with Grid Overlay')

% 🔹 First: plot grid points (so they appear underneath)
scatter(gridPoints(:,1), gridPoints(:,2), 5, 'o', ...
    'MarkerFaceColor', [0.7 0.7 0.7], ...
    'MarkerEdgeColor', [0.5 0.5 0.5], ...
    'LineWidth', 0.5, ...
    'DisplayName', 'Grid Points');

% Dredge half-width (radius from centerline)
R = L / 2;

% 🔹 Then: plot swath polygons and traces
for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));  % xy = [x, y]

    % Create buffer polygon (swept area)
    swathPoly = polybuffer(xy, 'lines', R);

    % Plot swath with transparency
    plot(swathPoly, ...
        'FaceColor', 'blue', ...
        'FaceAlpha', 0.1, ...
        'EdgeColor', 'none', ...
        'DisplayName', sprintf('Swath %02d', i));

    % Overlay centerline trace
    plot(xy(:,1), xy(:,2), 'k-', 'LineWidth', 0.1);
end

legend show

%% Area calculation
% For each swath (swath01, swath02, ...):
% Create swathXX = [xpt, ypt, zpt]
% xpt, ypt: the coordinates of all grid points
% zpt: 1 if inside swath polygon, else 0

% Loop through each swath polygon
for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));  % trace
    swathPoly = polybuffer(xy, 'lines', R);  % buffer polygon

    % Check which grid points are inside the swath polygon
    inSwath = isinterior(swathPoly, gridPoints(:,1), gridPoints(:,2));  % logical array

    % Create swathXX = [xpt, ypt, zpt]
    xpt = gridPoints(:,1);
    ypt = gridPoints(:,2);
    zpt = double(inSwath);  % convert logical to numeric (0 or 1)

    % Store as swath01, swath02, etc.
    eval(sprintf('swath%02d = [xpt, ypt, zpt];', i));
end


% % Plot for checking/demonstration
% for i = 1:length(sheets)
%     % Get swathXX data
%     swath = eval(sprintf('swath%02d', i));
%     xpt = swath(:,1);
%     ypt = swath(:,2);
%     zpt = swath(:,3);
% 
%     % Define color based on zpt
%     colors = repmat([0.9 0.9 0.9], length(zpt), 1);  % default gray
%     colors(zpt == 1, :) = repmat([1 0 0], sum(zpt == 1), 1);  % red for covered points
% 
%     % Create figure
%     figure
%     scatter(xpt, ypt, 10, colors, 's', 'filled')  % 's' = square marker
%     axis equal
%     grid on
%     xlabel('X (m)')
%     ylabel('Y (m)')
%     title(sprintf('Swath Coverage for swath%02d', i))
% end


%% Cumutive area calculation
% Initialize cumulative swath
for i = 1:length(sheets)
    if i == 1
        % First cumulative swath is just swath01
        eval(sprintf('CumSwath%02d = swath%02d;', i, i));
    else
        % Previous cumulative swath
        prev = eval(sprintf('CumSwath%02d', i-1));
        current = eval(sprintf('swath%02d', i));

        % Sum the zpt columns; keep x and y from swath01
        xpt = current(:,1);
        ypt = current(:,2);
        zpt = prev(:,3) + current(:,3);

        eval(sprintf('CumSwath%02d = [xpt, ypt, zpt];', i));
    end
end

% Visualization (only plot the last one?)
% for i = 1:length(sheets)
for i = length(sheets):length(sheets)
    % Load cumulative swath
    cumSwath = eval(sprintf('CumSwath%02d', i));
    xpt = cumSwath(:,1);
    ypt = cumSwath(:,2);
    zpt = cumSwath(:,3);

    % Replace 0s with NaN so they're not plotted
    zpt(zpt == 0) = NaN;

    % Create figure
    figure
    scatter(xpt, ypt, 15, zpt, 's', 'filled')  % size=15, shape=square
    axis equal
    grid off
    xlabel('X (m)')
    ylabel('Y (m)')
    title(sprintf('Cumulative Swath Coverage: CumSwath%02d', i))

    colormap sky 
    colorbar
    zmax = max(zpt, [], 'omitnan');
    if isfinite(zmax)
        if zmax == 1
            caxis([1, 1.01])  % Slightly expand range so color axis works
        else
            caxis([1, zmax])
        end
    end
end

% Quantify overlap statistics per swath with respect to the cumulative coverage at that step
maxOverlap = length(sheets);  % max possible z level
swathSummary = zeros(maxOverlap, maxOverlap);  % rows = swath, cols = overlap levels

for i = 1:maxOverlap
    swath = eval(sprintf('swath%02d', i));
    cum = eval(sprintf('CumSwath%02d', i));

    swathMask = (swath(:,3) == 1);  % points this swath covers
    cumZ = cum(:,3);  % total coverage at this step

    for z = 1:maxOverlap
        % Count how many points in swath i have cumulative coverage z
        count = sum(swathMask & (cumZ == z));
        swathSummary(i, z) = count;
    end
end

% Create row and column labels
swathNames = arrayfun(@(i) sprintf('Swath%02d', i), 1:maxOverlap, 'UniformOutput', false);
zLabels = arrayfun(@(z) sprintf('n_%02d', z), 1:maxOverlap, 'UniformOutput', false);

% Add N total per swath
totalPerSwath = sum(swathSummary, 2);
swathTable = array2table([swathSummary, totalPerSwath], ...
    'VariableNames', [zLabels, {'N_total'}], ...
    'RowNames', swathNames);

% Display the table
disp(swathTable);

% Convert count table to area table
areaPerPoint = m^2;
swathArea = swathSummary * areaPerPoint;

% Create area table
areaTable = array2table([swathArea, sum(swathArea, 2)], ...
    'VariableNames', [zLabels, {'Area_total_m2'}], ...
    'RowNames', swathNames);

% Display area table
disp(areaTable);


%% Exporting
% Set output filename
outputFile = 'A04_Summary of area.xlsx';

% 1. Export relative coordinates to "Relative coor" sheet
% Build a combined table of xy01, xy02, ...
xyAll = [];
varNames = {};
maxLen = 0;

% First, find the max length
for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));
    maxLen = max(maxLen, size(xy,1));
end

% Now pad and assemble
for i = 1:length(sheets)
    xy = eval(sprintf('xy%02d', i));
    n = size(xy,1);
    
    xcol = NaN(maxLen,1);
    ycol = NaN(maxLen,1);
    xcol(1:n) = xy(:,1);
    ycol(1:n) = xy(:,2);
    
    xyAll = [xyAll, xcol, ycol];
    varNames{end+1} = sprintf('x%02d', i);
    varNames{end+1} = sprintf('y%02d', i);
end

% Convert to table and write
xyTable = array2table(xyAll, 'VariableNames', varNames);
writetable(xyTable, outputFile, 'Sheet', 'Relative coor')

% 2. Export swathTable to "Swath count"
writetable(swathTable, outputFile, 'Sheet', 'Swath count', 'WriteRowNames', true)

% 3. Export areaTable to "Swath area"
writetable(areaTable, outputFile, 'Sheet', 'Swath area', 'WriteRowNames', true)

% Set output folder and base filename
outputFolder = '.';  % or specify a subfolder like 'figures'
baseName = 'Figure';

% Get handles to all open figures
figHandles = findall(groot, 'Type', 'figure');

% Save each figure
for i = 1:length(figHandles)
    fig = figHandles(i);
    figFileName = fullfile(outputFolder, sprintf('%s_%02d.fig', baseName, i));
    savefig(fig, figFileName);
end



toc