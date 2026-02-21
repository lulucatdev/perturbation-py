var k c;
varexo eps;
predetermined_variables k;

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
matlab;
  save('ghx_test.txt','oo_.dr.ghx','-ascii');
  save('ghu_test.txt','oo_.dr.ghu','-ascii');
  save('order_test.txt','oo_.dr.order_var','-ascii');
  save('state_test.txt','oo_.dr.state_var','-ascii');
end;
