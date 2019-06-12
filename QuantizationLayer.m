classdef QuantizationLayer < nnet.layer.Layer
    
    % TODO   : ***__ Change tanh sum function to a*tanh(c*(x-b))
    % TODO   : **___ Recheck if the derivation are performed correctly
    
    properties
        quantizersNum
        codewordsNum
        a
        c
    end
    
    properties (Learnable)
        % Layer learnable parameters
        b
    end
    
    methods
        function layer = QuantizationLayer(quantizers, codewords)
            % Set number of inputs.
            layer.quantizersNum = quantizers;
            layer.codewordsNum = codewords;
            
            % Set layer name.
            layer.Name = 'Quantization';
            
            % Set layer description.
            layer.Description = "Learning quantization layer with " ...
                + quantizers + " quantizers of " + codewords + ...
                " codewords each";
            
            % Initialize layer weights
            layer.a = ones(1, codewords);
            % FIXME: b and c should not be multiplied
            layer.b = 10*linspace(-1, 1, codewords);
            layer.c = 100*ones(1, codewords);
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

            Z = single(zeros(size(X)));
            for jj = 1:size(Z, 2)
                for ii = 1:size(Z, 1)
                    Z(ii,jj) = sum(layer.a .* tanh(layer.c*X(ii,jj) - layer.b));
                end
            end
        end
        
        function [dLdX, dLdb] = backward(layer, X, ~, dLdZ, ~)
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

            dLdX = single(zeros(size(X)));
            for jj = 1:size(dLdX, 2)
                for ii = 1:size(dLdX, 1)
                    dZidXi = sum(layer.a .* layer.c .* (1-(tanh(layer.c * X(ii,jj) - layer.b)).^2));
                    dLdX(ii,jj) = dLdZ(ii,jj) * dZidXi;
                end
            end
            
            dLdb = single(zeros(size(layer.b)));
            for jj = 1:size(dLdX, 2)
                for ii = 1:size(dLdb, 2)
                    dZdbi = dLdZ(:,jj) .* ((tanh(layer.c(ii)*X(:,jj) - layer.b(ii))).^2 - 1);
                    dLdb(ii) = layer.a(ii) * sum(dZdbi);
                end
            end
        end
    end
end