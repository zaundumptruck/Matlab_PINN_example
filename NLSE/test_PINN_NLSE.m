classdef test_PINN_NLSE < handle
    properties
        x0
        t0
        x_lb
        t_lb
        x_ub
        t_ub
        x_f
        t_f
        u0
        v0
        lb
        ub
        layers
        weights
        biases
        net
        options
    end
    
    methods
        function obj = test_PINN_NLSE(x0, u0, v0, tb, X_f, layers, lb, ub)
            % 构造函数，初始化网络参数
            X0 = [x0, zeros(size(x0))];  % 左边界坐标点(x,0)
            X_lb = [zeros(size(tb)) + lb(1), tb];  % 下边界坐标点(-5,t)
            X_ub = [zeros(size(tb)) + ub(1), tb];  % 上边界坐标点(5,t)
            
            obj.lb = lb;
            obj.ub = ub;
            
            obj.x0 = X0(:,1);
            obj.t0 = X0(:,2);
            
            obj.x_lb = X_lb(:,1);
            obj.t_lb = X_lb(:,2);
            
            obj.x_ub = X_ub(:,1);
            obj.t_ub = X_ub(:,2);
            
            obj.x_f = X_f(:,1);
            obj.t_f = X_f(:,2);
            
            obj.u0 = u0;
            obj.v0 = v0;
            
            obj.layers = layers;
            
            % 初始化网络
            obj.initialize_NN(layers);
            
            % 设置训练选项
            obj.options = trainingOptions('adam', ...
                'MaxEpochs', 50000, ...
                'InitialLearnRate', 0.01, ...
                'GradientThreshold', 1, ...
                'Plots', 'training-progress', ...
                'Verbose', false);
        end
        
        function initialize_NN(obj, layers)
            % 初始化网络结构
            layers_dlnet = [];
            for i = 1:length(layers)-1
                layers_dlnet = [layers_dlnet;
                    featureInputLayer(layers(1), 'Name', 'input');
                    fullyConnectedLayer(layers(i+1), 'Name', sprintf('fc%d',i));
                    tanhLayer('Name', sprintf('tanh%d',i))];
            end
            layers_dlnet = [layers_dlnet;
                fullyConnectedLayer(2, 'Name', 'output')];
            
            obj.net = dlnetwork(layers_dlnet);
        end
        
        function [u, v, u_x, v_x] = net_uv(obj, x, t)
            % 计算网络输出及其导数
            X = [x, t];
            X = 2.0 * (X - obj.lb) ./ (obj.ub - obj.lb) - 1.0;
            
            % 前向传播
            X = dlarray(X, 'CB');
            uv = forward(obj.net, X);
            uv = extractdata(uv);
            
            u = uv(:,1);
            v = uv(:,2);
            
            % 计算空间导数
            u_x = gradient(u, x);
            v_x = gradient(v, x);
        end
        
        function [f_u, f_v] = net_f_uv(obj, x, t)
            % 计算NLSE方程的残差
            [u, v, u_x, v_x] = obj.net_uv(x, t);
            
            % 计算时间和空间二阶导数
            u_t = gradient(u, t);
            u_xx = gradient(u_x, x);
            
            v_t = gradient(v, t);
            v_xx = gradient(v_x, x);
            
            % NLSE方程的残差
            f_u = u_t + 0.5*v_xx + (u.^2 + v.^2).*v;
            f_v = v_t - 0.5*u_xx - (u.^2 + v.^2).*u;
        end
        
        function train(obj)
            % 训练网络
            % 构建训练数据
            X_train = [obj.x0, obj.t0;
                obj.x_lb, obj.t_lb;
                obj.x_ub, obj.t_ub;
                obj.x_f, obj.t_f];
            
            % 训练循环
            for epoch = 1:obj.options.MaxEpochs
                % 计算损失函数
                [u0_pred, v0_pred] = obj.net_uv(obj.x0, obj.t0);
                [u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred] = obj.net_uv(obj.x_lb, obj.t_lb);
                [u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred] = obj.net_uv(obj.x_ub, obj.t_ub);
                [f_u_pred, f_v_pred] = obj.net_f_uv(obj.x_f, obj.t_f);
                
                loss = mean((obj.u0 - u0_pred).^2) + ...
                    mean((obj.v0 - v0_pred).^2) + ...
                    mean((u_lb_pred - u_ub_pred).^2) + ...
                    mean((v_lb_pred - v_ub_pred).^2) + ...
                    mean((u_x_lb_pred - u_x_ub_pred).^2) + ...
                    mean((v_x_lb_pred - v_x_ub_pred).^2) + ...
                    mean(f_u_pred.^2) + ...
                    mean(f_v_pred.^2);
                
                % 更新网络参数
                gradients = dlgradient(loss, obj.net.Learnables);
                obj.net.Learnables = dlupdate(@(p,g) p - obj.options.InitialLearnRate*g, ...
                    obj.net.Learnables, gradients);
                
                % 显示训练进度
                if mod(epoch, 10) == 0
                    fprintf('Epoch: %d, Loss: %.3e\n', epoch, loss);
                end
            end
        end
        
        function [u_pred, v_pred, f_u_pred, f_v_pred] = predict(obj, X_star)
            % 预测新数据点的解
            x_star = X_star(:,1);
            t_star = X_star(:,2);
            
            [u_pred, v_pred] = obj.net_uv(x_star, t_star);
            [f_u_pred, f_v_pred] = obj.net_f_uv(x_star, t_star);
        end
    end
end

% 主程序
function main()
    % 设置随机数种子
    rng(1234);
    
    % 定义问题参数
    noise = 0.0;
    lb = [-5.0, 0.0];
    ub = [5.0, pi/2];
    
    N0 = 50;
    N_b = 50;
    N_f = 20000;
    layers = [2, 100, 100, 100, 100, 2];
    
    % 加载数据
    data = load('../Data/NLS.mat');
    t = data.tt(:);
    x = data.x(:);
    Exact = data.uu;
    Exact_u = real(Exact);
    Exact_v = imag(Exact);
    Exact_h = sqrt(Exact_u.^2 + Exact_v.^2);
    
    [X, T] = meshgrid(x, t);
    
    X_star = [X(:), T(:)];
    u_star = Exact_u';
    v_star = Exact_v';
    h_star = Exact_h';
    
    % 采样训练数据
    idx_x = randperm(length(x), N0);
    x0 = x(idx_x);
    u0 = Exact_u(idx_x, 1);
    v0 = Exact_v(idx_x, 1);
    
    idx_t = randperm(length(t), N_b);
    tb = t(idx_t);
    
    % 使用Latin Hypercube采样生成配点
    X_f = lhsdesign(N_f, 2);
    X_f = lb + (ub - lb) .* X_f;
    
    % 创建和训练模型
    model = test_PINN_NLSE(x0, u0, v0, tb, X_f, layers, lb, ub);
    
    tic;
    model.train();
    training_time = toc;
    fprintf('Training time: %.4f\n', training_time);
    
    % 预测并计算误差
    [u_pred, v_pred, f_u_pred, f_v_pred] = model.predict(X_star);
    h_pred = sqrt(u_pred.^2 + v_pred.^2);
    
    error_u = norm(u_star(:) - u_pred(:)) / norm(u_star(:));
    error_v = norm(v_star(:) - v_pred(:)) / norm(v_star(:));
    error_h = norm(h_star(:) - h_pred(:)) / norm(h_star(:));
    
    fprintf('Error u: %e\n', error_u);
    fprintf('Error v: %e\n', error_v);
    fprintf('Error h: %e\n', error_h);
    
    % 可视化结果
    U_pred = griddata(X_star(:,1), X_star(:,2), u_pred, X, T);
    V_pred = griddata(X_star(:,1), X_star(:,2), v_pred, X, T);
    H_pred = griddata(X_star(:,1), X_star(:,2), h_pred, X, T);
    
    % 绘制结果
    plot_results(X, T, H_pred, x, t, Exact_h, x0, tb, lb, ub);
end

function plot_results(X, T, H_pred, x, t, Exact_h, x0, tb, lb, ub)
    % 创建图形
    figure('Position', [100, 100, 1000, 800]);
    
    % 绘制H_pred的热图
    subplot(2, 1, 1);
    imagesc([lb(2), ub(2)], [lb(1), ub(1)], H_pred');
    colorbar;
    hold on;
    
    % 绘制训练数据点
    X0 = [x0, zeros(size(x0))];
    X_lb = [ones(size(tb))*lb(1), tb];
    X_ub = [ones(size(tb))*ub(1), tb];
    X_u_train = [X0; X_lb; X_ub];
    plot(X_u_train(:,2), X_u_train(:,1), 'kx', 'MarkerSize', 4);
    
    % 绘制切片线
    line = linspace(min(x), max(x), 2);
    plot([t(75), t(75)], line, 'k--', 'LineWidth', 1);
    plot([t(100), t(100)], line, 'k--', 'LineWidth', 1);
    plot([t(125), t(125)], line, 'k--', 'LineWidth', 1);
    
    xlabel('t');
    ylabel('x');
    title('|h(t,x)|');
    axis tight;
    
    % 绘制切片对比图
    subplot(2, 3, 4);
    plot(x, Exact_h(:,75), 'b-', 'LineWidth', 2);
    hold on;
    plot(x, H_pred(75,:), 'r--', 'LineWidth', 2);
    xlabel('x');
    ylabel('|h(t,x)|');
    title(sprintf('t = %.2f', t(75)));
    axis square;
    xlim([-5.1, 5.1]);
    ylim([-0.1, 5.1]);
    legend('Exact', 'Prediction');
    
    subplot(2, 3, 5);
    plot(x, Exact_h(:,100), 'b-', 'LineWidth', 2);
    hold on;
    plot(x, H_pred(100,:), 'r--', 'LineWidth', 2);
    xlabel('x');
    ylabel('|h(t,x)|');
    title(sprintf('t = %.2f', t(100)));
    axis square;
    xlim([-5.1, 5.1]);
    ylim([-0.1, 5.1]);
    
    subplot(2, 3, 6);
    plot(x, Exact_h(:,125), 'b-', 'LineWidth', 2);
    hold on;
    plot(x, H_pred(125,:), 'r--', 'LineWidth', 2);
    xlabel('x');
    ylabel('|h(t,x)|');
    title(sprintf('t = %.2f', t(125)));
    axis square;
    xlim([-5.1, 5.1]);
    ylim([-0.1, 5.1]);
    
    % 保存图形
    saveas(gcf, './figures/retest/reNLS.png');
end