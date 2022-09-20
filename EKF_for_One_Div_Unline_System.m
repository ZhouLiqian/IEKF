%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 创建时间：2022/04/07
% 创建人：周立倩
% 程序功能：EKF滤波器设计
% 状态方程：x(k+1)=0.5*x(k)+2.5*x(k)/(1+x(k)^2)+8*cos(1.2*(k+1))+w(k)
% 观测方程：z(k+1)=x(k+1)^2/20+v(k+1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function EKF_for_One_Div_Unline_System
% 初始化
T=50;
Q=10;
R=1;
w=sqrt(Q)*randn(1,T);
v=sqrt(R)*randn(1,T);
% 状态方程
 x=zeros(1,T);
 y=zeros(1,T);
 x(1)=0.1;
 y(1)=x(1)^2/20+v(1);
for k=2:T
    x(k)=0.5*x(k-1)+2.5*x(k-1)/(1+x(k-1)^2)+8*cos(1.2*k)+w(k-1);% 真实状态值
    y(k)=x(k)^2/20+v(k);% 真实观测值
end
% EKF滤波算法
Xekf=zeros(1,T);% 后验估计值
P0=eye(1);
Xekf(1)=x(1);
for k=2:T
Xn=0.5*Xekf(k-1)+2.5*Xekf(k-1)/(1+Xekf(k-1)^2)+8*cos(1.2*k);% 状态预测
Zn=Xn^2/20;% 观测预测
F=0.5+2.5*(1-Xn^2)/(1+Xn^2)^2;% 状态转移矩阵
H=Xn/10;% 观测矩阵
P=F*P0*F'+Q;% 预测协方差矩阵
K=P*H'*inv(H*P*H'+R);% 卡尔曼增益
Xekf(k)=Xn+K*(y(k)-Zn);% 状态更新 Xn:先验估计值
P0=(eye(1)-K*H)*P;% 协方差更新
end
% 误差分析
Xstd=zeros(1,T);
for k=1:T
    Xstd(k)=abs(Xekf(k)-x(k));
end
% 画图
figure
hold on;
box on;
plot(x,'-ko','Markerface','r');
plot(Xekf,'-ks','Markerface','b');
xlabel('时间/s','FontSize',10);
ylabel('状态值x','FontSize',10);
title('EKF滤波结果','FontSize',16);
