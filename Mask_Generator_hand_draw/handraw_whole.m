function [cropedMask] = handraw_whole(inputIm)
figure(10001)
imagesc(inputIm);
 title('Hand labeling')
  axis tight; axis equal
 h = imfreehand;
 position = h.getPosition();
 cropedMask = poly2mask(position(:,1),position(:,2),(size(inputIm,2)),size(inputIm,2));
close(figure(10001))
end