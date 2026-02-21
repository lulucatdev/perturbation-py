var k c;
varexo eps;

model(linear);
  k(+1) = 0.9*k + 0.01*eps;
  c - 0.95*c(+1) - 0.1*k = 0;
end;

initval;
  k = 0;
  c = 0;
  eps = 0;
end;

steady;
check;

shocks;
  var eps; stderr 1;
end;

stoch_simul(order=1, irf=0, noprint, nograph, nocorr, nomoments);
