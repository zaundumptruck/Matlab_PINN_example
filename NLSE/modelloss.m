function [loss,gradients] = modelLoss(net,X,T,X0,T0,U0)

% Make predictions with the initial conditions.
XT = cat(1,X,T);
% 神经网络输出两个通道：实部和虚部
Uout = forward(net,XT);
% 将输出分离为实部和虚部
Ureal = Uout(:,1,:);
Uimag = Uout(:,2,:);
% 构建复数值
U = complex(Ureal, Uimag);

% Calculate derivatives with respect to X and T.
X = stripdims(X);
T = stripdims(T);
Ureal = stripdims(Ureal);
Uimag = stripdims(Uimag);
U = complex(Ureal, Uimag);

% 计算关于X的导数
UxReal = dljacobian(Ureal,X,1);
UxImag = dljacobian(Uimag,X,1);
Ux = complex(UxReal, UxImag);

% 计算关于T的导数
UtReal = dljacobian(Ureal,T,1);
UtImag = dljacobian(Uimag,T,1);
Ut = complex(UtReal, UtImag);

% 计算关于X的二阶导数
UxxReal = dldivergence(UxReal,X,1);
UxxImag = dldivergence(UxImag,X,1);
Uxx = complex(UxxReal, UxxImag);

% 计算非线性项 |U|^2*U
Uabs2 = Ureal.^2 + Uimag.^2;  % |U|^2
NonlinearTermReal = Uabs2 .* Ureal;
NonlinearTermImag = Uabs2 .* Uimag;
NonlinearTerm = complex(NonlinearTermReal, NonlinearTermImag);

% 计算mseF。强制执行非线性薛定谔方程：i*u_t + u_xx + |u|^2*u = 0
% 方程重写为：i*u_t + u_xx + |u|^2*u = 0
% 分解为实部和虚部
fReal = UtImag + UxxReal + NonlinearTermReal;
fImag = -UtReal + UxxImag + NonlinearTermImag;

% 计算残差的平方和
mseF = mean(fReal.^2 + fImag.^2);

% Calculate mseU. Enforce initial and boundary conditions.
XT0 = cat(1,X0,T0);
U0Pred = forward(net,XT0);
mseU = l2loss(U0Pred,U0);

% Calculated loss to be minimized by combining errors.
loss = mseF + mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end
