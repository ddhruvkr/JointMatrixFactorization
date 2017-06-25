R=importdata('E:\nidhi Data\imp_papers\2015_btech_8sem\Matrix_factorization_research_paper\code\Tr1.mat'); %size(R)=358070*3
Users=6040;
Items=1682;
LF=5;
alpha=0.001; beta=20;alpha1=0.02;
num_iterations=50;
Uf=randn(Users,LF);
Vf=randn(Items,LF); %1682*LF
Uf1=zeros(Users,LF);
Vf1=zeros(Items,LF);
bu=zeros(length(R),1);
bi=zeros(length(R),1);
count=0;es=0.0;err=0.0;lambda=0.01;lambda1=0.01;
global_rating1= mean(R(:,3));
for num_iterations = 1:num_iterations
for i=1:size(R)
    pred =0.0;
    U=R(i,1);
    I=R(i,2);
    r=R(i,3)-global_rating1;
   for k=1:LF
    pred = pred + (Uf(U,k)* Vf(I,k));
   end
    err=r-pred+global_rating1+bu(i)+bi(i);
   % fprintf(1, '\n Before Training num_iterations %d RMSE %6.4f', num_iterations , err);
   bu(i)= bu(i)+alpha*(err-lambda1*bu(i));
   bi(i)= bi(i)+alpha*(err-lambda1*bi(i));
    es=es+err*err;
    count=count+1;
   for k=1:LF
        Uf1(U,k)=Uf(U,k)+alpha*((err * Vf(I,k))-(lambda*Uf(U,k)));
        Vf1(I,k)=Vf(I,k)+alpha*((err * Uf(U,k))-(lambda*Vf(I,k)));
   end
   
  for k=1:LF
  Uf(U,k)=Uf1(U,k);
  Vf(I,k)= Vf1(I,k);
end
end
%%%%%%%%%%%%RMSE on training dataset%%%%%%%%%%%%%%%%%%%%%%%%%%%
es=sqrt(es/(count*1.0));
fprintf(1, '\n After Trainingnum_iterations %d RMSE %6.4f', num_iterations , es);
end
Ts=importdata('E:\nidhi Data\imp_papers\2015_btech_8sem\Matrix_factorization_research_paper\code\Tr1.mat');
es1=0.0; 
for j=1:length(Ts)
    pred1=0.0; 
    U1=Ts(j,1);
    I1=Ts(j,2);
    r1=Ts(j,3)-global_rating1;
    for k=1:LF
        pred1=pred1+Uf(U1,k)*Vf(I1,k);
    end
   if (pred1>5)
     pred1=5;
   else if(pred1<0)
    pred1=0;
       end
   end
    err=r1-pred1+global_rating1+bu(i)+bi(i);
    es1=es1+err*err;
   dlmwrite('C:\Users\iiita\Desktop\nid3.csv',pred1,'-append');
end
es1=sqrt(es1/length(Ts));
fprintf(1, '\n After Testing num_iterations %d RMSE %6.4f', num_iterations , es1);



