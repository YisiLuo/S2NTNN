function Out=interpolate(I, Mask)
      [n1, n2, n3]=size (Mask);
      Out=zeros(n1, n2, n3) ;
      x=1:n1 ;y=1:n2;
      [x, y]=meshgrid(x, y) ;
      for i=1 :n3
      mask=Mask(:, :, i) ;
      temp=I(:, :, i) ;
      [X, Y]=find(mask== 1) ;
      X=double(X) ;Y=double(Y) ;
      Out2(:, :,i) = griddata(X, Y, temp(mask == 1), x, y) ;
      end

      %1_ recover=I_ recover' ;

      Out=permute(Out2,[2 1 3]);
      Out (isnan (Out))=128;
      end