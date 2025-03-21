clc;
clear all
global a
a=0.7;
z0=0.4;
b=1;
d=0;
%%-----------
n=2048;
hx=0.06;
x=(-n/2:n/2-1)*hx;
hw=2*pi/(n*hx);
w=fftshift((-n/2:n/2-1)*hw);
%%--------------
p=sqrt(8*a*(1-2*a));
q=2*sqrt(1-2*a);
 U=exp(1i*(z0))*(1+(2*(1-2*a)*cosh(p*(z0))+1i*p*sinh(p*(z0)))./(sqrt(2*a)*cos(q*x)-cosh(p*(z0))));%KM
cosh(p*z0)
sinh(p*z0)
% U=exp(-1i*z0)*(1-4*(1-2*1i*z0)./(1+4*(x.^2)+4*(z0.^2)));%PS
V=(angle(U));
u1(1,:)=(abs(U).^2)'; 
L=60;
nm=10000;
h=L/nm;
for j=1:nm
    j;
    z=j*h;
    D=exp((-1/2.*1i.*(abs(w)).^b).*h/2);
   qstep1=ifft(D.*fft(U));
     N=exp(1i.*d.*(abs(qstep1).^2).*h);
    qstep2=N.*qstep1;
     U=ifft(D.*fft(qstep2));
    l=floor(2+(j-1)/100);
    u1(l,:)=(abs(U).^2)';
end
s=max(u1(l,:))
ss=min(u1(l,:))
figure(1)
plot(x,u1(1,:),'r',x,u1(l,:),'b')
xlabel('x (a.u.)','FontSize',18);
ylabel('Intensity|U|^2 (a.u.)','FontSize',18);
figure(2)
z=0:h*100:L;
mesh(x,z,u1)
xlabel('x (a.u.)','FontSize',18);
ylabel('Distance z (a.u.)','FontSize',18);
zlabel('Intensity|U|^2 (a.u.)','FontSize',18); 
view(-50,30);
figure(3)
z=0:h*100:L;
mesh(x,z,u1)
view(0,90)                                                                                                                                                                                                                                      
xlabel('x (a.u.)','FontSize',18);
ylabel('Distance z (a.u.)','FontSize',18);
figure(4)
plot(x,u1(l,:),'b')
xlabel('x (a.u.)','FontSize',18);
ylabel('Intensity|U|^2 (a.u.)','FontSize',18);
figure(5)
plot(x,u1(1,:),'r')
xlabel('x (a.u.)','FontSize',18);
ylabel('Intensity|U|^2 (a.u.)','FontSize',18);  
figure(6)
z=0:h*100:L;
mesh(x,z,u1)
view(0,0)
figure(7)
plot(x,V,'r')
xlabel('x (a.u.)','FontSize',18);
ylabel('Phase (a.u.)','FontSize',18); 
save  'D:\matlabxiugai\expc00.dat' u1 -ascii
save  'D:\matlabxiugai\expc01.dat' x -ascii

