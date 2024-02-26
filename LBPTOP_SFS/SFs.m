function SFsVector = SFs(Histogram)
% Histogram is the image after RILBPTOP
% clc
% clear
% load datahistogram

[x,y]=size(Histogram);
xx=reshape(double(Histogram),[1,x*y]);

SFsVector(1)=var(xx);
SFsVector(2)=kurtosis(xx);
SFsVector(3)=skewness(xx);

end