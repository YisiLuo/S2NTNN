function sol  = interpolation(Nhsi)
Y_tensor0 = Nhsi;
Omega = find(Y_tensor0 ~= 0);
A = Y_tensor0;
Ind = zeros(size(Y_tensor0));
Ind(Omega)  = 1;
B = padarray(A,[20,20,20],'symmetric','both');
C = padarray(Ind,[20,20,20],'symmetric','both');
%a0 = interpolate2(B,C);
a1 = interpolate(shiftdim(B,1),shiftdim(C,1));
a1(a1<0) = 0;
a1(a1>1) = 1;
a1 = a1(21:end-20,21:end-20,21:end-20);
a1 = shiftdim(a1,2);
a1(Omega) = Y_tensor0(Omega);

a2 = interpolate(shiftdim(B,2),shiftdim(C,2));
a2(a2<0) = 0;
a2(a2>1) = 1;
a2 = a2(21:end-20,21:end-20,21:end-20);
a2 = shiftdim(a2,1);
a2(Omega) = Y_tensor0(Omega);
a = 0.5*a1+0.5*a2;
sol = a;
