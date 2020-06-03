
% loops to generate test results for python implemementation of triangular
% dislocations
% Eric Lindsey, 2020 - modified from 2018 version by Ben Thompson

space={'FS','HS'};
tests={'simple','complex'};
n=10; % evaluation points along each spatial dimension (total pts: n^3)

nu=0.25;
for i=1:length(space)
    % get the point coordinates and vectors
    x = linspace(-3, 3, n);
    y = linspace(-3, 3, n);
    if strcmp(space{i},'FS')
        z = linspace(-3, 3, n);
    elseif strcmp(space{i},'HS')
        z = linspace(-3, 0, n); % half-space difference
    end
    [X,Y,Z] = meshgrid(x, y, z);
    Xf = reshape(X, numel(X), 1);
    Yf = reshape(Y, numel(Y), 1);
    Zf = reshape(Z, numel(Z), 1);
    for j=1:length(tests)
        % get the triangle coordinates and slip vector
        if strcmp(tests{j},'simple')
            P1 = [0,0,0];
            P2 = [1,0,0];
            P3 = [0,1,0];
            if strcmp(space{i},'HS')
                % make the z coordinates negative for HS case
                P1(3) = -1;
                P2(3) = -1;
                P3(3) = -1;
            end
            slip = [1.0, 0, 0];
        elseif strcmp(tests{j},'complex')
            P1 = [0,0.1,0.1];
            P2 = [1,-0.2,-0.2];
            P3 = [1,1,0.3];
            if strcmp(space{i},'HS')
                % make the z coordinates negative for HS case
                P1(3) = -0.9;
                P2(3) = -1.2;
                P3(3) = -0.7;
            end
            slip = [1.3,1.4,1.5];
        end

        % run the test
        % 1 million TDE evaluations takes 1 second on 1 core on meade03
        disp(['running test: ' space{i} '_' tests{j}])
        if strcmp(space{i},'FS')
            tic;
            [UEf, UNf, UVf] = TDdispFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), nu);
            dtime=toc;
            tic;
            [Stress, Strain] = TDstressFS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
            stime=toc; 
        elseif strcmp(space{i},'HS')
            tic;
            [UEf, UNf, UVf] = TDdispHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), nu);
            dtime=toc;
            tic;
            [Stress, Strain] = TDstressHS(Xf, Yf, Zf, P1, P2, P3, slip(1), slip(2), slip(3), 1.0, 1.0);
            stime=toc; 
        end
        disp(['displ. time: ',num2str(dtime)])
        disp(['stress time: ',num2str(stime)])
        disp(' ')

        % save results for Python comparison
        filename=[space{i} '_' tests{j} '.mat'];
        format long;
        obs=[Xf,Yf,Zf];
        tri=[P1;P2;P3];
        save(filename, 'obs', 'tri', 'slip', 'nu', 'UEf', 'UNf', 'UVf', 'Stress', 'Strain','dtime','stime');

    end
end

%% plot result

UEf(isnan(UEf)) = 0;
UNf(isnan(UEf)) = 0;
UVf(isnan(UEf)) = 0;

UE = reshape(UEf, n, n, n);
%
f = figure(1) %'visible','off');
contourf(y, z, reshape(UE(:, round(n / 2) + 1, :), n, n));
xlabel('x');
ylabel('y');
colorbar();
%saveas(f, 'figure.png');
