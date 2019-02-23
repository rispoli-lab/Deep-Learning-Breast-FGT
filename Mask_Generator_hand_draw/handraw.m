
function [cropedMask,maskIm] = handraw(inputIm,Bmax)
figure(10001)
 imagesc(inputIm);
 title('Hand labeling')
  axis tight; axis equal
 h = imfreehand;
 position = h.getPosition();
 cropedMask = poly2mask(position(:,1),position(:,2),(size(inputIm,2)),size(inputIm,2));
 %e = impoly(gca,position);
%cropedMask = createMask(e,h_im);
   cropedMask = uint8(cropedMask);
  cropedMask(cropedMask ==0) = Bmax;
  cropedMask(cropedMask ==1) = 0;
maskIm = inputIm.*cropedMask;
maskIm(isnan(maskIm))=0;
close(figure(10001))
end