function U = solveNLSE(X, t, a)
    % 使用分步傅里叶法求解非线性薛定谔方程    
    % 设置参数
    N = length(X);          % 空间网格点数(10000)
    L = 2;                  % 空间区间长度 [-1,1]
    dx = L/N;              % 空间步长
    dt = 0.001;            % 时间步长
    nt = round(t/dt);      % 时间步数
    b = 1;

    x=(-n/2:n/2-1)*dx;     % 空间坐标
   
    % 设置初始条件 
    p = sqrt(8*a*(1-2*a));
    q = 2*sqrt(1-2*a);
    U = 1 + (2*(1-2*a))./(sqrt(2*a)*cos(q*X)-1);
    
    % 计算频率网格
    kx = (2*pi/L) * [0:N/2-1 -N/2:-1]'; % 频率网格
    w=fftshift(kx);
    
    % 分步傅里叶法求解
    for n = 1:nt
        z = n*dx;
        D=exp((-1/2.*1i.*(abs(w)).^b).*h/2);
        qstep1=ifft(D.*fft(U));
        N=exp(1i.*d.*(abs(qstep1).^2).*h
        qstep2=N.*qstep1;
        U=ifft(D.*fft(qstep2));

  l=floor(2+(j-1)/100);  %计算存储索引，每100步记录一次
  u1(l,:)=(abs(U).^2)';
% (abs(U).^2)' 计算当前光强分布并转置为行向量
%  u1(l,:)=将光强分布存储在矩阵u1中

        
        % 确保边界条件（与原代码保持一致）
        if any(abs(X) == 1)
            U(abs(X) == 1) = 0;
        end
    end
    
    U = real(U);
end