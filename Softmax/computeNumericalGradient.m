function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 
EPSILON = 0.0001 ;
tmp = diag(zeros(size(theta)));
tmp_plus = tmp + diag(ones(size(theta))*EPSILON);
tmp_mins = tmp - diag(ones(size(theta))*EPSILON);
theta_plus = repmat(theta,1,size(theta)) + tmp_plus;
theta_mins = repmat(theta,1,size(theta)) + tmp_mins;
Cell_plus = num2cell(theta_plus,1);
Cell_mins = num2cell(theta_mins,1); 
res_plus = cellfun(J,Cell_plus);
res_mins = cellfun(J,Cell_mins);
numgrad = (res_plus - res_mins)/(2 * EPSILON);
numgrad = numgrad.';
% size(numgrad)
%% ---------------------------------------------------------------
end
