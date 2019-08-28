classdef PhaseShiftersLayer < nnet.layer.Layer
    
    %#ok<*PROPLC>
    
    properties
        n
        k
    end
    
    properties (Learnable)
        % Layer learnable parameters
        theta
    end
    
    methods
        function layer = PhaseShiftersLayer(inputSize, outputSize)
            if mod(inputSize, 2)
                error('Input size of PhaseShiftersLayer sould be even');
            elseif mod(outputSize, 2)
                error('Output size of PhaseShiftersLayer sould be even');
            end
            
            % Set layer name.
            layer.Name = 'PhaseShiftersLayer';
            
            % Set layer description.
            layer.Description = "Phase shifters layer";
            
            
            layer.n = inputSize / 2;
            layer.k = outputSize / 2;
            layer.theta = ones(layer.k * layer.n, 1);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Write parameters conviniently
            k = layer.k;
            n = layer.n;
            theta = vec2mat(layer.theta, n);
            
            Z = single(zeros(2*k, size(X,2)));
            for jj = 1:size(Z, 2)
                for ii = 1:k
                    Z(ii,jj) = sum(X(1:n,jj)  .* cos(theta(ii, :))' ...
                                 - X(n+1:end,jj) .* sin(theta(ii,:))');
                    Z(ii+k,jj) = sum(X(1:n,jj)  .* sin(theta(ii, :))' ...
                                   + X(n+1:end,jj) .* cos(theta(ii,:))');
                end
            end
        end
        
        function [dLdX, dLdtheta] = backward(layer, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value which can be used in
            %                             backward propagation [unused]
            % Output:
            %         dLdX              - Derivative of the loss with
            %                             respect to the input data
            %         dLdAlpha          - Derivatives of the loss with
            %                             respect to alpha

            % Write parameters conviniently
            k = layer.k;
            n = layer.n;
            theta = vec2mat(layer.theta, n);
            
            dLdX = single(zeros(size(X)));
            for jj = 1:size(dLdX, 2)
                for mm = 1:n
                    dZidXm = cos(theta(:, mm));
                    dZikdXm = sin(theta(:, mm));
                    dZidXmn = -sin(theta(:, mm));
                    dZikdXmn = cos(theta(:, mm));
                    dLdX(mm,jj) = sum(dLdZ(1:k,jj) .* dZidXm + ...
                                      dLdZ(k+1:end,jj) .* dZikdXm);
                    dLdX(mm+n,jj) = sum(dLdZ(1:k,jj) .* dZidXmn + ...
                                        dLdZ(k+1:end,jj) .* dZikdXmn);
                end
            end
            
            dLdtheta = single(zeros(k*n, 1));
            for jj = 1:size(dLdX, 2)
                for ii = 1:k
                    for mm = 1:n
                        dZidTheta = - X(mm,jj) * sin(theta(ii,mm)) ...
                                    - X(mm+n,jj) * cos(theta(ii,mm));
                        dZikdTheta = X(mm,jj) * cos(theta(ii,mm)) ...
                                   - X(mm+n,jj) * sin(theta(ii,mm));
                        dLdtheta(mm*ii, jj) = dLdZ(ii,jj) * dZidTheta + ...
                                              dLdZ(ii+k,jj) * dZikdTheta;
                    end
                end
            end
            dLdtheta = sum(dLdtheta, 2);
        end
    end
end