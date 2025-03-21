%%Generate Training Data
numBoundaryConditionPoints = [25 25];  %边界点x(-5,5)

x0BC1 = -5*ones(1,numBoundaryConditionPoints(1));
x0BC2 = 5*ones(1,numBoundaryConditionPoints(2));

t0BC1 = linspace(0,pi/2,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,pi/2,numBoundaryConditionPoints(2));

u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
v0BC1 = zeros(1,numBoundaryConditionPoints(1));
v0BC2 = zeros(1,numBoundaryConditionPoints(2)); 

u0BC1 = u0BC2;
v0BC1 = v0BC2;

numInitialConditionPoints  = 50;  %初始时间点(0,pi/2)

x0IC = linspace(-5,5,numInitialConditionPoints);
t0IC = zeros(1,numInitialConditionPoints);

a = 0.7;
q = 2 * sqrt(1 - 2*a);
u0IC = 1 + (2*(1-2*a))./(sqrt(2*a)*cos(q.*x0IC)-1);
v = zeros(size(x));
v0IC = zeros(1,numInitialConditionPoints);

X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
v0 = [v0IC v0BC1 v0BC2];
%%
numInternalCollocationPoints = 20000;  %均匀采样

points = rand(numInternalCollocationPoints,2);

dataX = 10*points(:,1)-5;
dataT = pi/2*points(:,2);

%% 检测GPU可用性
executionEnvironment = "auto";
canUseGPU = (gpuDeviceCount > 0);
if canUseGPU
    executionEnvironment = "gpu";
    disp("使用GPU加速训练");
else
    disp("GPU不可用，使用CPU训练");
end
%%
%神经网络构建
numBlocks = 4;
fcOutputSize = 100;

fcBlock = [
    fullyConnectedLayer(fcOutputSize)
    tanhLayer];

layers = [
    featureInputLayer(2)
    repmat(fcBlock,[numBlocks 1])
    fullyConnectedLayer(2)];

net = dlnetwork(layers);

%%%网格参数
% net = 
%   dlnetwork with properties:
%          Layers: [18×1 nnet.cnn.layer.Layer]
%     Connections: [17×2 table]
%      Learnables: [18×3 table]
%           State: [0×3 table]
%      InputNames: {'input'}
%     OutputNames: {'fc_9'}
%     Initialized: 1

%学习对象转换为double
    net = dlupdate(@double,net);
    % net = dlupdate(@single,net);
%%
if canUseGPU
    net = dlupdate(@gpuArray,net);
end
%%
%训练选项
solverState = lbfgsState;
maxIterations = 10;  %训练步数
gradientTolerance = 1e-5;
stepTolerance = 1e-5;

%%
%训练网络
% 数据转化；将数据转移到GPU
if canUseGPU
    X = dlarray(gpuArray(dataX),"BC");
    T = dlarray(gpuArray(dataT),"BC");
    X0 = dlarray(gpuArray(X0),"CB");
    T0 = dlarray(gpuArray(T0),"CB");
    U0 = dlarray(gpuArray(U0),"CB");
else
    X = dlarray(dataX,"BC");
    T = dlarray(dataT,"BC");
    X0 = dlarray(X0,"CB");
    T0 = dlarray(T0,"CB");
    U0 = dlarray(U0,"CB");
end

%
accfun = dlaccelerate(@modelLoss1);
lossFcn = @(net) dlfeval(accfun,net,X,T,X0,T0,U0);
monitor = trainingProgressMonitor( ...
    Metrics="TrainingLoss", ...
    Info=["Iteration" "GradientsNorm" "StepNorm"], ...
    XLabel="Iteration");

iteration = 0;
while iteration < maxIterations && ~monitor.Stop
    iteration = iteration + 1;
    [net, solverState] = lbfgsupdate(net,lossFcn,solverState);

    updateInfo(monitor, ...
        Iteration=iteration, ...
        GradientsNorm=solverState.GradientsNorm, ...
        StepNorm=solverState.StepNorm);

    recordMetrics(monitor,iteration,TrainingLoss=solverState.Loss);

    monitor.Progress = 100*iteration/maxIterations;

    if solverState.GradientsNorm < gradientTolerance || ...
            solverState.StepNorm < stepTolerance || ...
            solverState.LineSearchStatus == "failed"
        break
    end

end

%%
tTest = [0.25 0.5 0.75 1];
numObservationsTest = numel(tTest);

szXTest = 1001;
XTest = linspace(-1,1,szXTest);
% 将测试数据转移到GPU（如果可用）
if canUseGPU
    XTest = dlarray(gpuArray(XTest),"CB");
else
    XTest = dlarray(XTest,"CB");
end

UPred = zeros(numObservationsTest,szXTest);
UTest = zeros(numObservationsTest,szXTest);

 for i = 1:numObservationsTest
     t = tTest(i);

    TTest = repmat(t,[1 szXTest]);
    if canUseGPU
            TTest = dlarray(gpuArray(TTest),"CB");
        else
            TTest = dlarray(TTest,"CB");
    end
    XTTest = cat(1,XTest,TTest);

    UPred(i,:) = forward(net,XTTest);
    UTest(i,:) = solveNLSE(extractdata(XTest),t,0.7);
end

err = norm(UPred - UTest) / norm(UTest);

figure
tiledlayout("flow")

for i = 1:numel(tTest)
    nexttile
    
    plot(XTest,UPred(i,:),"-",LineWidth=2);

    hold on
    plot(XTest, UTest(i,:),"--",LineWidth=2)
    hold off

    ylim([-1.1, 1.1])
    xlabel("x")
    ylabel("u(x," + t + ")")
end

legend(["Prediction" "Target"])

%%
